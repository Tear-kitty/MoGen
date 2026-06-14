from __future__ import annotations

from typing import Iterable, List, Optional

import torch
from PIL import Image

from .adapter_setup import set_mogen_attention_processors
from .checkpointing import load_mogen_checkpoint
from .constants import (
    CONTROL_NUM_TOKENS,
    CONTROL_TARGET_BLOCKS,
    DEFAULT_NEGATIVE_PROMPT,
    BOX_LABEL_HASH_SIZE,
    MAX_APPEARANCE_REFS,
    MAX_BOXES,
    TEXT_NUM_TOKENS,
    TEXT_TARGET_BLOCKS,
)
from .projection import MoGenProjection
from .utils import get_generator


class MoGenAdapterXL:
    """Inference wrapper around an SDXL pipeline and MoGen projector."""

    def __init__(
        self,
        sd_pipe,
        checkpoint_path: str,
        device: str | torch.device,
        mode: str,
        target_blocks: Optional[Iterable[str]] = None,
        num_tokens: Optional[int] = None,
        attention_backend: str = "sdpa",
        dtype: torch.dtype = torch.float16,
        strict_load: bool = True,
        use_global_text_in_control: bool = True,
        global_text_scale: float = 1.0,
        max_boxes: int = MAX_BOXES,
        box_label_hash_size: int = BOX_LABEL_HASH_SIZE,
    ):
        if mode not in {"text", "control"}:
            raise ValueError("mode must be 'text' or 'control'")
        self.device = torch.device(device)
        self.mode = mode
        self.dtype = dtype
        self.checkpoint_path = checkpoint_path
        self.num_tokens = num_tokens if num_tokens is not None else (TEXT_NUM_TOKENS if mode == "text" else CONTROL_NUM_TOKENS)
        self.target_blocks = list(target_blocks or (TEXT_TARGET_BLOCKS if mode == "text" else CONTROL_TARGET_BLOCKS))

        self.pipe = sd_pipe.to(self.device)
        self.attention_modules = set_mogen_attention_processors(
            self.pipe,
            target_blocks=self.target_blocks,
            num_tokens=self.num_tokens,
            train_text=(mode == "text"),
            device=self.device,
            dtype=dtype,
        )
        self.projector = MoGenProjection(
            1536,
            512,
            32,
            train_text=(mode == "text"),
            max_appearance_refs=MAX_APPEARANCE_REFS,
            attention_backend=attention_backend,
            use_global_text_in_control=use_global_text_in_control,
            global_text_scale=global_text_scale,
            max_boxes=max_boxes,
            box_label_hash_size=box_label_hash_size,
        ).to(self.device, dtype=dtype)
        load_mogen_checkpoint(checkpoint_path, self.projector, self.attention_modules, strict=strict_load)
        self.projector.eval()
        self.attention_modules.eval()

    def set_scale(self, scale: float) -> None:
        from .attention_processor import MoGenAttnProcessor2_0

        for processor in self.pipe.unet.attn_processors.values():
            if isinstance(processor, MoGenAttnProcessor2_0):
                processor.scale = scale

    def _repeat_condition(self, tensor: Optional[torch.Tensor], num_samples: int) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        if tensor.ndim != 3:
            raise ValueError(f"Expected condition tensor [B, N, C], got {tuple(tensor.shape)}")
        if tensor.shape[0] != 1:
            raise ValueError("Current inference wrapper expects one prompt/control set at a time.")
        return tensor.to(self.device, dtype=self.dtype).repeat(num_samples, 1, 1)

    def _pack_inference_controls(
        self,
        num_samples: int,
        structure: Optional[torch.Tensor] = None,
        box: Optional[torch.Tensor] = None,
        appearance: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        parts = []
        for tensor in (
            self._repeat_condition(structure, num_samples),
            self._repeat_condition(box, num_samples),
            self._repeat_condition(appearance, num_samples),
        ):
            if tensor is not None:
                parts.append(tensor)

        if not parts:
            # Empty controls for the unconditional CFG branch.
            control = torch.zeros(num_samples, 1, 1536, device=self.device, dtype=self.dtype)
            mask = torch.zeros(num_samples, 1, device=self.device, dtype=torch.bool)
            has = torch.zeros(num_samples, device=self.device, dtype=torch.bool)
            return control, mask, has

        control = torch.cat(parts, dim=1)
        mask = control.abs().sum(dim=-1) != 0
        has = mask.any(dim=1)
        return control, mask, has

    @torch.inference_mode()
    def generate(
        self,
        prompt: str | List[str],
        negative_prompt: Optional[str | List[str]] = None,
        structure: Optional[torch.Tensor] = None,
        box: Optional[torch.Tensor] = None,
        appearance: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        num_samples: int = 4,
        seed: Optional[int] = None,
        guidance_scale: float = 5.0,
        num_inference_steps: int = 30,
        **kwargs,
    ) -> List[Image.Image]:
        self.set_scale(scale)

        if isinstance(prompt, str):
            prompt = [prompt]
        if len(prompt) != 1:
            raise ValueError("This wrapper currently supports one prompt per call; use num_samples for multiple images.")
        if negative_prompt is None:
            negative_prompt = DEFAULT_NEGATIVE_PROMPT
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(
            prompt,
            device=self.device,
            num_images_per_prompt=num_samples,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )

        prompt_embeds = prompt_embeds.to(self.device, dtype=self.dtype)
        negative_prompt_embeds = negative_prompt_embeds.to(self.device, dtype=self.dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(self.device, dtype=self.dtype)
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(self.device, dtype=self.dtype)

        if self.mode == "text":
            prompt_embeds, positive_condition = self.projector.project_text_branch(prompt_embeds)
            _, negative_condition = self.projector.project_text_branch(negative_prompt_embeds)
        else:
            if structure is None and box is None and appearance is None:
                raise ValueError("Control mode inference needs at least one of --structure_image, --box_json, --appearance_dir.")
            control, mask, has = self._pack_inference_controls(num_samples, structure, box, appearance)
            prompt_embeds, positive_condition = self.projector.project_control_branch(prompt_embeds, control, mask, has)
            empty_control, empty_mask, empty_has = self._pack_inference_controls(num_samples)
            _, negative_condition = self.projector.project_control_branch(
                negative_prompt_embeds, empty_control, empty_mask, empty_has
            )

        prompt_embeds = torch.cat([prompt_embeds, positive_condition], dim=1)
        negative_prompt_embeds = torch.cat([negative_prompt_embeds, torch.zeros_like(negative_condition)], dim=1)

        generator = get_generator(seed, self.device)
        return self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images
