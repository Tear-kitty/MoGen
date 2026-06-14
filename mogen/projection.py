from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input
    FLASH_ATTN_AVAILABLE = True
except Exception:  # pragma: no cover - flash-attn is installed in the runtime env, not in CI.
    flash_attn_func = None
    flash_attn_varlen_func = None
    index_first_axis = None
    pad_input = None
    FLASH_ATTN_AVAILABLE = False


class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, min(dim * mult * 2, 8192)),
            GEGLU(),
            nn.Linear(min(dim * mult, 8192 // 2), dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _unpad_input(hidden_states: torch.Tensor, attention_mask: torch.Tensor):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def efficient_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask_q: Optional[torch.Tensor] = None,
    mask_kv: Optional[torch.Tensor] = None,
    dropout: float = 0.0,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    backend: str = "flash",
) -> torch.Tensor:
    """Attention over tensors shaped [B, N, H, D].

    `backend="flash"` uses the external flash-attn package and supports varlen
    masks. `backend="sdpa"` uses PyTorch's scaled_dot_product_attention and is
    convenient for inference.
    """
    backend = backend.lower()
    if backend == "flash-attn":
        backend = "flash"
    if backend not in {"flash", "sdpa"}:
        raise ValueError(f"Unknown attention backend: {backend}")

    bsz, q_len, heads, head_dim = q.shape
    kv_len = k.shape[1]

    if causal and not (q_len == 1 or q_len == kv_len):
        raise ValueError("Causal mask only supports self-attention with N == 1 or N == M.")

    if backend == "flash":
        if not FLASH_ATTN_AVAILABLE:
            raise ImportError("flash-attn is not available. Install flash-attn or use --attention_backend sdpa.")
        if mask_q is None and mask_kv is None:
            return flash_attn_func(q, k, v, dropout, causal=causal, window_size=window_size)

        if mask_q is None:
            mask_q = torch.ones(bsz, q_len, dtype=torch.bool, device=q.device)
        if mask_kv is None:
            mask_kv = torch.ones(bsz, kv_len, dtype=torch.bool, device=q.device)

        # flash-attn varlen requires at least one key/value token per row. Rows
        # with no real control tokens are made safe upstream and overwritten to
        # zero after attention.
        q_unpad, indices_q, cu_q, max_q = _unpad_input(q, mask_q)
        k_unpad, indices_kv, cu_kv, max_kv = _unpad_input(k, mask_kv)
        v_unpad = index_first_axis(v.reshape(-1, heads, head_dim), indices_kv)
        out = flash_attn_varlen_func(
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_kv,
            max_seqlen_q=max_q,
            max_seqlen_k=max_kv,
            dropout_p=dropout,
            causal=causal,
            window_size=window_size,
        )
        return pad_input(out, indices_q, bsz, q_len)

    # PyTorch SDPA path. It can be faster/easier for inference and does not
    # require the external flash-attn package.
    q_sdpa = q.transpose(1, 2)  # [B, H, N, D]
    k_sdpa = k.transpose(1, 2)
    v_sdpa = v.transpose(1, 2)
    attn_mask = None
    if mask_kv is not None:
        attn_mask = mask_kv[:, None, None, :].to(torch.bool)
    out = F.scaled_dot_product_attention(
        q_sdpa, k_sdpa, v_sdpa, attn_mask=attn_mask, dropout_p=dropout, is_causal=causal
    )
    out = out.transpose(1, 2).contiguous()
    if mask_q is not None:
        out = out * mask_q[:, :, None, None].to(dtype=out.dtype)
    return out


class CrossAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        input_dim: Optional[int] = None,
        context_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        dropout: float = 0.0,
        attention_backend: str = "flash",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim if input_dim is not None else hidden_dim
        self.context_dim = context_dim if context_dim is not None else hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        self.num_heads = num_heads
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout
        self.attention_backend = attention_backend

        self.q_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.context_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.context_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.output_dim)

    def set_attention_backend(self, backend: str) -> None:
        self.attention_backend = backend

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask_q: Optional[torch.Tensor] = None,
        mask_kv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, q_len, channels = x.shape
        kv_len = context.shape[1]
        q = self.q_proj(x).reshape(bsz, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(context).reshape(bsz, kv_len, self.num_heads, self.head_dim)
        v = self.v_proj(context).reshape(bsz, kv_len, self.num_heads, self.head_dim)
        x = efficient_attention(q, k, v, mask_q, mask_kv, dropout=self.dropout, backend=self.attention_backend)
        x = x.reshape(bsz, -1, channels)
        return self.out_proj(x)


class CrossAttBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, gradient_checkpointing: bool = True, attention_backend: str = "flash"):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.ln1 = nn.LayerNorm(dim)
        self.att = CrossAttention(dim, num_heads, attention_backend=attention_backend)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim)

    def set_attention_backend(self, backend: str) -> None:
        self.att.set_attention_backend(backend)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        mask_q: Optional[torch.Tensor] = None,
        mask_kv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, x, c, mask_q, mask_kv, use_reentrant=False)
        return self._forward(x, c, mask_q, mask_kv)

    def _forward(self, x: torch.Tensor, c: torch.Tensor, mask_q: Optional[torch.Tensor], mask_kv: Optional[torch.Tensor]):
        x = x + self.att(self.ln1(x), c, mask_q, mask_kv)
        x = x + self.mlp(self.ln2(x))
        return x


class SelfAttBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, gradient_checkpointing: bool = True, attention_backend: str = "flash"):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.ln1 = nn.LayerNorm(dim)
        self.att = CrossAttention(dim, num_heads, attention_backend=attention_backend)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim)

    def set_attention_backend(self, backend: str) -> None:
        self.att.set_attention_backend(backend)

    def forward(
        self,
        x: torch.Tensor,
        mask_q: Optional[torch.Tensor] = None,
        mask_kv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, x, mask_q, mask_kv, use_reentrant=False)
        return self._forward(x, mask_q, mask_kv)

    def _forward(self, x: torch.Tensor, mask_q: Optional[torch.Tensor], mask_kv: Optional[torch.Tensor]):
        x = x + self.att(self.ln1(x), self.ln1(x), mask_q, mask_kv)
        x = x + self.mlp(self.ln2(x))
        return x


class BoxCoordinateEncoder(nn.Module):
    """Encode LabelMe boxes as explicit coordinate tokens.

    Each real box becomes one token with normalized geometry, Fourier features,
    a frozen SDXL text_encoder_2 label embedding, a small hashed-label residual,
    order embedding, and count embedding. This keeps instance geometry explicit
    while giving category labels a pretrained semantic prior.
    """

    def __init__(
        self,
        cross_attention_dim: int = 1536,
        heads: int = 32,
        max_boxes: int = 64,
        label_hash_size: int = 4096,
        label_dim: int = 128,
        label_text_dim: int = 1280,
        fourier_bands: int = 6,
        attention_backend: str = "flash",
    ):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.max_boxes = max_boxes
        self.label_hash_size = label_hash_size
        self.label_dim = label_dim
        self.label_text_dim = label_text_dim
        self.num_box_features = 10
        self.fourier_bands = fourier_bands
        self.register_buffer("fourier_freqs", 2.0 ** torch.arange(fourier_bands, dtype=torch.float32), persistent=False)

        input_dim = self.num_box_features * (1 + 2 * fourier_bands) + label_dim
        # self.label_embed = nn.Embedding(label_hash_size, label_dim)
        self.label_text_proj = nn.Sequential(
            nn.LayerNorm(label_text_dim),
            nn.Linear(label_text_dim, label_dim),
        )
        # A small learned residual keeps backward compatibility with hashed-label
        # checkpoints without making the random hash id the primary label signal.
        # self.label_hash_scale = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))
        self.order_embed = nn.Embedding(max_boxes, cross_attention_dim)
        self.count_embed = nn.Embedding(max_boxes + 1, cross_attention_dim)
        self.input_mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, cross_attention_dim),
            nn.GELU(),
            nn.Linear(cross_attention_dim, cross_attention_dim),
        )
        self.layers = nn.ModuleList(
            [SelfAttBlock(cross_attention_dim, heads, True, attention_backend) for _ in range(4)]
        )
        self.out = nn.Sequential(
            nn.LayerNorm(cross_attention_dim),
            nn.Linear(cross_attention_dim, cross_attention_dim),
        )

    def set_attention_backend(self, backend: str) -> None:
        for layer in self.layers:
            layer.set_attention_backend(backend)

    def forward(
        self,
        box_features: torch.Tensor,
        box_label_ids: torch.Tensor,
        box_token_mask: torch.Tensor,
        box_label_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if box_features.ndim != 3 or box_features.shape[-1] != self.num_box_features:
            raise ValueError(
                f"Expected box_features [B, N, {self.num_box_features}], got {tuple(box_features.shape)}"
            )
        _bsz, num_boxes, _ = box_features.shape
        if num_boxes > self.max_boxes:
            box_features = box_features[:, : self.max_boxes]
            box_label_ids = box_label_ids[:, : self.max_boxes]
            box_token_mask = box_token_mask[:, : self.max_boxes]
            if box_label_embeds is not None:
                box_label_embeds = box_label_embeds[:, : self.max_boxes]
            num_boxes = self.max_boxes

        dtype = self.input_mlp[1].weight.dtype
        device = box_features.device
        box_features = box_features.to(dtype=dtype)
        box_token_mask = box_token_mask.to(device=device, dtype=torch.bool)
        box_label_ids = box_label_ids.to(device=device, dtype=torch.long).clamp(0, self.label_hash_size - 1)

        freqs = self.fourier_freqs.to(device=device, dtype=dtype)
        angles = box_features[..., None] * freqs * (2.0 * math.pi)
        fourier = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).flatten(-2)
        if box_label_embeds is not None:
            label_embeds = self.label_text_proj(box_label_embeds.to(dtype=dtype))
        else:
            semantic_unused_link = box_features.new_tensor(0.0)
            for param in self.label_text_proj.parameters():
                semantic_unused_link = semantic_unused_link + param.sum() * 0.0

            label_embeds = box_features.new_zeros(
                box_features.shape[0],
                box_features.shape[1],
                self.label_embed_dim,
            ) + semantic_unused_link.to(dtype=dtype) 
        tokens = torch.cat([box_features, fourier, label_embeds], dim=-1)
        tokens = self.input_mlp(tokens)

        order_ids = torch.arange(num_boxes, device=device).clamp(max=self.max_boxes - 1)
        tokens = tokens + self.order_embed(order_ids)[None, :, :].to(dtype=dtype)
        counts = box_token_mask.sum(dim=1).clamp(max=self.max_boxes)
        tokens = tokens + self.count_embed(counts)[:, None, :].to(dtype=dtype)
        tokens = tokens * box_token_mask[:, :, None].to(dtype=dtype)

        # Flash-attention varlen kernels need at least one valid token per row.
        safe_mask = box_token_mask.clone()
        empty_rows = ~safe_mask.any(dim=1)
        if empty_rows.any():
            safe_mask[empty_rows, 0] = True
            tokens = tokens.clone()
            tokens[empty_rows] = 0

        for layer in self.layers:
            tokens = layer(tokens, safe_mask, safe_mask)
        tokens = self.out(tokens)
        if empty_rows.any():
            tokens = tokens.clone()
            tokens[empty_rows] = 0
        tokens = tokens * box_token_mask[:, :, None].to(dtype=dtype)
        return tokens


class MoGenProjection(nn.Module):
    """The original MoGen projection architecture, cleaned up but unchanged.

    Two modes are supported:
    - train_text=True: text-only stage, producing 64 phrase tokens.
    - train_text=False: control stage, producing 256 control tokens from any
      subset of structure, box, and appearance references.
    """

    def __init__(
        self,
        cross_attention_dim: int = 1536,
        clip_embeddings_dim: int = 512,
        heads: int = 32,
        train_text: bool = True,
        max_appearance_refs: int = 15,
        attention_backend: str = "flash",
        use_global_text_in_control: bool = True,
        global_text_scale: float = 1.0,
        max_boxes: int = 64,
        box_label_hash_size: int = 4096,
    ):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.clip_embeddings_dim = clip_embeddings_dim
        self.image_dim = 1536
        self.train_text = train_text
        self.max_appearance_refs = max_appearance_refs
        self.attention_backend = attention_backend
        self.use_global_text_in_control = use_global_text_in_control
        self.global_text_scale = float(global_text_scale)
        self.max_boxes = max_boxes
        self.box_label_hash_size = box_label_hash_size

        # Text token width after SDXL text_encoder + text_encoder_2 concatenation.
        self.cross_attention_dim = 2048
        self.heads = int(self.cross_attention_dim / 64)

        if self.train_text or self.use_global_text_in_control:
            self.model_global = SelfAttBlock(self.cross_attention_dim, self.heads, True, attention_backend)
            self.model_global_out = nn.Sequential(
                nn.LayerNorm(self.cross_attention_dim),
                nn.Linear(self.cross_attention_dim, self.cross_attention_dim),
            )

        if self.train_text:
            self.phrase_query = nn.Parameter(torch.randn(1, 64, self.cross_attention_dim) / self.cross_attention_dim**0.5)
            self.phrase_model = CrossAttBlock(self.cross_attention_dim, self.heads, True, attention_backend)
            self.phrase_self = SelfAttBlock(self.cross_attention_dim, self.heads, True, attention_backend)
            self.phrase_out = nn.Sequential(
                nn.LayerNorm(self.cross_attention_dim),
                nn.Linear(self.cross_attention_dim, self.cross_attention_dim),
            )
        else:
            self.cross_attention_dim = cross_attention_dim
            self.box_encoder = BoxCoordinateEncoder(
                cross_attention_dim=cross_attention_dim,
                heads=heads,
                max_boxes=max_boxes,
                label_hash_size=box_label_hash_size,
                attention_backend=attention_backend,
            )
            self.structure_query = nn.Parameter(
                torch.randn(1, 64, cross_attention_dim) / cross_attention_dim**0.5
            )
            self.structure_cross_attention = CrossAttBlock(cross_attention_dim, heads, True, attention_backend)
            self.structure_out = nn.Sequential(
                nn.LayerNorm(self.cross_attention_dim),
                nn.Linear(self.cross_attention_dim, cross_attention_dim),
            )
            self.object_pos_dim = 128
            self.object_out = nn.Sequential(
                nn.LayerNorm(self.cross_attention_dim + self.object_pos_dim),
                nn.Linear(self.cross_attention_dim + self.object_pos_dim, cross_attention_dim),
            )
            self.object_pos = nn.Embedding(max_appearance_refs, self.object_pos_dim)
            self.condition_query = nn.Parameter(torch.randn(1, 256, self.cross_attention_dim) / self.cross_attention_dim**0.5)
            self.condition_integrate = CrossAttBlock(cross_attention_dim, heads, True, attention_backend)
            self.condition_self = nn.Sequential(
                *[SelfAttBlock(cross_attention_dim, heads, True, attention_backend) for _ in range(2)]
            )
            self.condition_out = nn.Sequential(
                nn.LayerNorm(cross_attention_dim),
                nn.Linear(self.cross_attention_dim, 2048),
            )

    def set_attention_backend(self, backend: str) -> None:
        self.attention_backend = backend
        for module in self.modules():
            if hasattr(module, "set_attention_backend") and module is not self:
                module.set_attention_backend(backend)

    def enhance_text(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "model_global"):
            return text_embeddings
        extra = self.model_global(text_embeddings)
        extra = self.model_global_out(extra)
        return text_embeddings + self.global_text_scale * extra

    def project_text_branch(self, text_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.train_text:
            raise RuntimeError("project_text_branch is only available in text mode.")
        embeds = text_embeddings
        text_embeddings = self.enhance_text(text_embeddings)
        query = self.phrase_query.repeat(embeds.shape[0], 1, 1)
        condition = self.phrase_model(query, embeds)
        condition = self.phrase_self(condition)
        condition = self.phrase_out(condition)
        return text_embeddings, condition

    def _unused_module_link(self, *modules_or_tensors) -> torch.Tensor:
        link = None
        for module_or_tensor in modules_or_tensors:
            if isinstance(module_or_tensor, torch.Tensor):
                parameters = (module_or_tensor,)
            else:
                parameters = module_or_tensor.parameters()
            for param in parameters:
                term = param.sum() * 0.0
                link = term if link is None else link + term
        if link is None:
            return torch.tensor(0.0, device=self.condition_query.device, dtype=self.condition_query.dtype)
        return link

    def _infer_control_batch_shape(self, batch: dict) -> Tuple[int, torch.device]:
        for key in ("appearance", "structure", "box_features", "drop_control_embeds"):
            if key in batch and torch.is_tensor(batch[key]):
                return int(batch[key].shape[0]), batch[key].device
        raise ValueError("Control mode requires at least one control tensor or drop_control_embeds in batch.")

    def _dino_dtype(self, dino_v2: Optional[nn.Module]) -> torch.dtype:
        if dino_v2 is None:
            return self.condition_query.dtype
        try:
            return next(dino_v2.parameters()).dtype
        except StopIteration:
            return self.condition_query.dtype

    def pack_control_embeddings(self, batch: dict, dino_v2: Optional[nn.Module]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode and pack any available subset of structure, box, appearance controls.

        No control type is assumed to exist for a batch item. A row may contain only
        structure, only box, only appearance, any combination of them, or no active
        control after CFG dropout. The returned mask is built explicitly instead of
        inferring validity from nonzero embeddings.
        """
        bsz, device = self._infer_control_batch_shape(batch)
        proj_dtype = self.condition_query.dtype
        dino_dtype = self._dino_dtype(dino_v2)

        drop_control = batch.get("drop_control_embeds")
        if drop_control is None:
            drop_control = torch.zeros(bsz, device=device, dtype=torch.bool)
        else:
            drop_control = drop_control.to(device=device, dtype=torch.bool)

        # ------------------------------ structure ------------------------------
        used_structure = False
        structure_embeddings = torch.zeros(bsz, 0, self.cross_attention_dim, device=device, dtype=proj_dtype)
        active_structure = torch.zeros(bsz, device=device, dtype=torch.bool)
        if "structure" in batch:
            has_structure = batch.get("has_structure")
            if has_structure is None:
                has_structure = torch.ones(bsz, device=device, dtype=torch.bool)
            else:
                has_structure = has_structure.to(device=device, dtype=torch.bool)
            active_structure = has_structure & ~drop_control
            if active_structure.any():
                if dino_v2 is None:
                    raise ValueError("Structure controls require dino_v2.")
                structure_ref = batch["structure"].to(device=device, dtype=dino_dtype)
                with torch.no_grad():
                    valid_structure_raw = dino_v2(structure_ref[active_structure]).last_hidden_state.to(dtype=proj_dtype)
                structure_query = self.structure_query.repeat(valid_structure_raw.shape[0], 1, 1).to(
                    device=device,
                    dtype=proj_dtype,
                )
                valid_structure_embeddings = self.structure_cross_attention(structure_query, valid_structure_raw)
                valid_structure_embeddings = self.structure_out(valid_structure_embeddings).to(dtype=proj_dtype)
                structure_embeddings = torch.zeros(
                    bsz,
                    valid_structure_embeddings.shape[1],
                    valid_structure_embeddings.shape[2],
                    device=device,
                    dtype=proj_dtype,
                )
                structure_embeddings[active_structure] = valid_structure_embeddings
                structure_embeddings = structure_embeddings * active_structure[:, None, None].to(dtype=proj_dtype)
                used_structure = True

        # ---------------------------------- box ---------------------------------
        used_box = False
        box_embeddings = torch.zeros(bsz, 0, self.cross_attention_dim, device=device, dtype=proj_dtype)
        box_token_mask_active = torch.zeros(bsz, 0, device=device, dtype=torch.bool)
        active_box = torch.zeros(bsz, device=device, dtype=torch.bool)
        has_box_tensors = all(key in batch for key in ("box_features", "box_label_ids", "box_token_mask"))
        if has_box_tensors:
            raw_box_mask = batch["box_token_mask"].to(device=device, dtype=torch.bool)
            has_box_mask = batch.get("has_box_mask")
            if has_box_mask is None:
                has_box_mask = raw_box_mask.any(dim=1)
            else:
                has_box_mask = has_box_mask.to(device=device, dtype=torch.bool)
            active_box = has_box_mask & raw_box_mask.any(dim=1) & ~drop_control
            if active_box.any():
                box_token_mask_active = raw_box_mask & active_box[:, None]
                box_label_embeds = batch.get("box_label_embeds")
                if box_label_embeds is not None:
                    box_label_embeds = box_label_embeds.to(device=device, dtype=proj_dtype)
                box_embeddings = self.box_encoder(
                    batch["box_features"].to(device=device),
                    batch["box_label_ids"].to(device=device),
                    box_token_mask_active,
                    box_label_embeds=box_label_embeds,
                ).to(dtype=proj_dtype)
                used_box = True

        # ------------------------------- appearance -----------------------------
        used_appearance = False
        appearance_embeddings = torch.zeros(bsz, 0, self.cross_attention_dim, device=device, dtype=proj_dtype)
        appearance_token_mask = torch.zeros(bsz, 0, device=device, dtype=torch.bool)
        active_appearance = torch.zeros(bsz, device=device, dtype=torch.bool)
        if "appearance" in batch:
            appearance = batch["appearance"]
            _bsz, num_refs, channels, height, width = appearance.shape
            if _bsz != bsz:
                raise ValueError(f"appearance batch size {_bsz} does not match inferred batch size {bsz}")

            appearance_num = batch.get("appearance_num")
            if appearance_num is None:
                appearance_num = torch.full((bsz,), num_refs, device=device, dtype=torch.long)
            else:
                appearance_num = appearance_num.to(device=device, dtype=torch.long).clamp(min=0, max=num_refs)
            has_appearance = batch.get("has_appearance")
            if has_appearance is None:
                has_appearance = appearance_num > 0
            else:
                has_appearance = has_appearance.to(device=device, dtype=torch.bool)
            active_appearance = has_appearance & (appearance_num > 0) & ~drop_control

            class_counts = batch.get("appearance_class_counts")
            instance_class_ids = batch.get("appearance_instance_class_ids")
            instance_ids = batch.get("appearance_instance_ids")

            if class_counts is not None and instance_class_ids is not None and instance_ids is not None:
                class_counts = class_counts.to(device=device, dtype=torch.long)[:, :num_refs].clamp(min=0)
                appearance_class_num = batch.get("appearance_class_num")
                if appearance_class_num is None:
                    appearance_class_num = (class_counts > 0).sum(dim=1)
                else:
                    appearance_class_num = appearance_class_num.to(device=device, dtype=torch.long).clamp(min=0, max=num_refs)

                class_slot_ids = torch.arange(num_refs, device=device).unsqueeze(0)
                class_mask = (class_slot_ids < appearance_class_num.unsqueeze(1)) & (class_counts > 0)
                class_mask = class_mask & active_appearance[:, None]
                valid_class_flat_mask = class_mask.reshape(-1)

                if valid_class_flat_mask.any():
                    if dino_v2 is None:
                        raise ValueError("Appearance controls require dino_v2.")
                    flat_appearance = appearance.to(device=device, dtype=dino_dtype).reshape(bsz * num_refs, channels, height, width)
                    with torch.no_grad():
                        valid_appearance = dino_v2(flat_appearance[valid_class_flat_mask]).last_hidden_state.to(dtype=proj_dtype)
                    seq_len = valid_appearance.shape[1]
                    feature_dim = valid_appearance.shape[2]
                    class_raw = torch.zeros(
                        bsz * num_refs,
                        seq_len,
                        feature_dim,
                        device=device,
                        dtype=proj_dtype,
                    )
                    class_raw[valid_class_flat_mask] = valid_appearance
                    class_raw = class_raw.reshape(bsz, num_refs, seq_len, feature_dim)

                    instance_class_ids = instance_class_ids.to(device=device, dtype=torch.long)[:, :num_refs]
                    instance_class_ids = instance_class_ids.clamp(min=0, max=max(0, num_refs - 1))
                    instance_ids = instance_ids.to(device=device, dtype=torch.long)[:, :num_refs]
                    instance_ids = instance_ids.clamp(min=0, max=self.max_appearance_refs - 1)

                    instance_slot_mask = torch.arange(num_refs, device=device).unsqueeze(0) < appearance_num.unsqueeze(1)
                    instance_slot_mask = instance_slot_mask & active_appearance[:, None]
                    instance_slot_mask = instance_slot_mask & (class_counts.gather(1, instance_class_ids) > 0)

                    gather_index = instance_class_ids[:, :, None, None].expand(-1, -1, seq_len, feature_dim)
                    appearance_raw = torch.gather(class_raw, dim=1, index=gather_index)
                    ref_pos = self.object_pos(instance_ids).unsqueeze(2).expand(-1, -1, seq_len, -1).to(dtype=proj_dtype)
                    appearance_raw = torch.cat([appearance_raw, ref_pos], dim=-1)

                    appearance_token_mask = instance_slot_mask[:, :, None].expand(-1, -1, seq_len).reshape(bsz, num_refs * seq_len)
                    appearance_raw = appearance_raw.reshape(bsz, num_refs * seq_len, appearance_raw.shape[-1])
                    appearance_embeddings = self.object_out(appearance_raw).to(dtype=proj_dtype)
                    appearance_embeddings = appearance_embeddings * appearance_token_mask[:, :, None].to(dtype=proj_dtype)
                    used_appearance = True
            else:
                # Backward-compatible path for older batches/checkpoints that only
                # carried repeated appearance tensors and a total appearance_num.
                ref_mask = torch.arange(num_refs, device=device).unsqueeze(0) < appearance_num.unsqueeze(1)
                ref_mask = ref_mask & active_appearance[:, None]
                valid_flat_mask = ref_mask.reshape(-1)

                if valid_flat_mask.any():
                    if dino_v2 is None:
                        raise ValueError("Appearance controls require dino_v2.")
                    flat_appearance = appearance.to(device=device, dtype=dino_dtype).reshape(bsz * num_refs, channels, height, width)
                    with torch.no_grad():
                        valid_appearance = dino_v2(flat_appearance[valid_flat_mask]).last_hidden_state.to(dtype=proj_dtype)
                    seq_len = valid_appearance.shape[1]
                    feature_dim = valid_appearance.shape[2]
                    appearance_raw = torch.zeros(
                        bsz * num_refs,
                        seq_len,
                        feature_dim,
                        device=device,
                        dtype=proj_dtype,
                    )
                    appearance_raw[valid_flat_mask] = valid_appearance
                    appearance_raw = appearance_raw.reshape(bsz, num_refs, seq_len, feature_dim)
                    ref_pos = self.object_pos(torch.arange(num_refs, device=device).unsqueeze(0).expand(bsz, -1))
                    ref_pos = ref_pos.unsqueeze(2).expand(-1, -1, seq_len, -1).to(dtype=proj_dtype)
                    appearance_raw = torch.cat([appearance_raw, ref_pos], dim=-1)
                    appearance_token_mask = ref_mask[:, :, None].expand(-1, -1, seq_len).reshape(bsz, num_refs * seq_len)
                    appearance_raw = appearance_raw.reshape(bsz, num_refs * seq_len, appearance_raw.shape[-1])
                    appearance_embeddings = self.object_out(appearance_raw)
                    appearance_embeddings = appearance_embeddings * appearance_token_mask[:, :, None].to(dtype=proj_dtype)
                    used_appearance = True

        unused_modules = []
        if not used_structure:
            unused_modules.extend([self.structure_query, self.structure_cross_attention, self.structure_out])
        if not used_box:
            unused_modules.append(self.box_encoder)
        if not used_appearance:
            unused_modules.extend([self.object_out, self.object_pos])
        unused_control_grad_link = self._unused_module_link(*unused_modules)

        items = []
        masks = []
        has_any_control = []
        max_tokens = 0
        for idx in range(bsz):
            parts = []
            part_masks = []

            if active_structure[idx] and structure_embeddings.shape[1] > 0:
                part = structure_embeddings[idx]
                parts.append(part)
                part_masks.append(torch.ones(part.shape[0], device=device, dtype=torch.bool))

            if active_box[idx] and box_embeddings.shape[1] > 0:
                valid = box_token_mask_active[idx]
                if valid.any():
                    part = box_embeddings[idx][valid]
                    parts.append(part)
                    part_masks.append(torch.ones(part.shape[0], device=device, dtype=torch.bool))

            if active_appearance[idx] and appearance_embeddings.shape[1] > 0:
                valid = appearance_token_mask[idx]
                if valid.any():
                    part = appearance_embeddings[idx][valid]
                    parts.append(part)
                    part_masks.append(torch.ones(part.shape[0], device=device, dtype=torch.bool))

            if parts:
                item = torch.cat(parts, dim=0)
                item_mask = torch.cat(part_masks, dim=0)
                has_any_control.append(bool(item_mask.any().item()))
            else:
                item = torch.zeros(0, self.cross_attention_dim, device=device, dtype=proj_dtype)
                item_mask = torch.zeros(0, device=device, dtype=torch.bool)
                has_any_control.append(False)

            items.append(item)
            masks.append(item_mask)
            max_tokens = max(max_tokens, int(item.shape[0]))

        # Keep one fully masked dummy token when all controls are absent/dropped so
        # zero-gradient links remain connected under DDP/Accelerate.
        max_tokens = max(max_tokens, 1)
        padded_items = []
        padded_masks = []
        for item, item_mask in zip(items, masks):
            pad_len = max_tokens - item.shape[0]
            if pad_len > 0:
                item = torch.cat(
                    [item, torch.zeros(pad_len, self.cross_attention_dim, device=device, dtype=proj_dtype)],
                    dim=0,
                )
                item_mask = torch.cat([item_mask, torch.zeros(pad_len, device=device, dtype=torch.bool)], dim=0)
            padded_items.append(item[:max_tokens])
            padded_masks.append(item_mask[:max_tokens])

        control_embeddings = torch.stack(padded_items, dim=0) + unused_control_grad_link
        mask_kv = torch.stack(padded_masks, dim=0)
        has_any_control_tensor = torch.tensor(has_any_control, device=device, dtype=torch.bool)
        return control_embeddings, mask_kv, has_any_control_tensor

    def project_control_branch(
        self,
        text_embeddings: torch.Tensor,
        control_embeddings: torch.Tensor,
        mask_kv: torch.Tensor,
        has_any_control: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.train_text:
            raise RuntimeError("project_control_branch is only available in control mode.")

        if self.use_global_text_in_control:
            text_embeddings = self.enhance_text(text_embeddings)

        bsz = text_embeddings.shape[0]
        query = self.condition_query.repeat(bsz, 1, 1)

        if has_any_control is None:
            has_any_control = mask_kv.any(dim=1)
        if not has_any_control.any():
            condition = torch.zeros(bsz, query.shape[1], 2048, device=text_embeddings.device, dtype=text_embeddings.dtype)
            unused_condition_link = (
                control_embeddings.sum() * 0.0
                + self.condition_query.sum() * 0.0
                + self._unused_module_link(self.condition_integrate, self.condition_self, self.condition_out)
            )
            condition = condition + unused_condition_link
            return text_embeddings, condition

        safe_mask = mask_kv.clone()
        empty_rows = ~safe_mask.any(dim=1)
        if empty_rows.any():
            safe_mask[empty_rows, 0] = True
            control_embeddings = control_embeddings.clone()
            control_embeddings[empty_rows] = 0

        condition = self.condition_integrate(query, control_embeddings, None, safe_mask)
        condition = self.condition_self(condition)
        condition = self.condition_out(condition)
        if empty_rows.any():
            condition = condition.clone()
            condition[empty_rows] = 0
        return text_embeddings, condition

    def forward(self, text_embeddings: torch.Tensor, dino_v2: Optional[nn.Module] = None, batch: Optional[dict] = None):
        if self.train_text:
            return self.project_text_branch(text_embeddings)
        if batch is None:
            raise ValueError("Control mode requires a batch with control tensors.")
        control_embeddings, mask_kv, has_any_control = self.pack_control_embeddings(batch, dino_v2)
        return self.project_control_branch(text_embeddings, control_embeddings, mask_kv, has_any_control)
