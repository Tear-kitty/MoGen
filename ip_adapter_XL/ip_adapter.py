import os
from typing import List
import torch.nn.functional as F

import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

import matplotlib.pyplot as plt
import seaborn as sns
import math
import torch.nn as nn

from .utils import is_torch2_available, get_generator

from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
FLASH_ATTN_AVAILABLE = True

from einops import rearrange

if is_torch2_available():
    from .attention_processor import (
        AttnProcessor2_0 as AttnProcessor,
    )
    from .attention_processor import (
        CNAttnProcessor2_0 as CNAttnProcessor,
    )
    from .attention_processor import (
        IPAttnProcessor2_0 as IPAttnProcessor,
    )
# else:
#     from .attention_processor import AttnProcessor, CNAttnProcessor, IPAttnProcessor
from .resampler import Resampler

# class WindowAwareLinearProjection(torch.nn.Module):
#     def __init__(self, text_embeddings_dim: int, window_size: int):
#         super().__init__()

#         self.emb_dim = text_embeddings_dim

#         self.projection = torch.nn.Conv1d(
#             in_channels=text_embeddings_dim,
#             out_channels=text_embeddings_dim,
#             kernel_size=window_size, 
#             padding='same',
#             padding_mode='zeros'
#         )

#         self.projection.weight.data.zero_()
#         self.projection.bias.data.zero_()
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         assert x.shape[2] == self.emb_dim

#         return x + self.projection(x.permute(0, 2, 1)).permute(0, 2, 1)
 
class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds)
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

# class WindowAwareLinearProjection(torch.nn.Module): #纯文本
#     """Projection Model"""

#     def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=512, heads=32):
#         super().__init__()

#         self.generator = None
#         self.cross_attention_dim = cross_attention_dim
#         self.clip_embeddings_dim = clip_embeddings_dim
#         self.heads = heads

#         self.proj_Q = torch.nn.Linear(cross_attention_dim, clip_embeddings_dim)
#         self.proj_K = torch.nn.Linear(cross_attention_dim, clip_embeddings_dim)
#         self.proj_V = torch.nn.Linear(cross_attention_dim, cross_attention_dim)
#         self.attention_norm = torch.nn.LayerNorm(cross_attention_dim)
#         self.ffn = nn.Sequential(
#             nn.Linear(cross_attention_dim, cross_attention_dim*4),
#             GEGLU(),
#             nn.Linear(cross_attention_dim*2, cross_attention_dim)
#         )

#         self.proj_out = torch.nn.Linear(cross_attention_dim, cross_attention_dim)

#         self.norm_out = torch.nn.LayerNorm(cross_attention_dim)

#     def forward(self, text_embeddings):
#         embeds = text_embeddings

#         batch_size = embeds.shape[0]

#         query = self.proj_Q(embeds).view(batch_size, -1, self.heads, int(self.clip_embeddings_dim/self.heads)).transpose(1, 2)
#         key = self.proj_K(embeds).view(batch_size, -1, self.heads, int(self.clip_embeddings_dim/self.heads)).transpose(1, 2)
#         value = self.proj_V(embeds).view(batch_size, -1, self.heads, int(self.cross_attention_dim/self.heads)).transpose(1, 2)

#         clip_extra_context_tokens = F.scaled_dot_product_attention(
#             query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
#         )
#         clip_extra_context_tokens = clip_extra_context_tokens.transpose(1, 2).reshape(batch_size, -1, self.heads * int(self.cross_attention_dim/self.heads))
        
#         clip_extra_context_tokens = embeds + clip_extra_context_tokens
#         # embeds = clip_extra_context_tokens
        
#         clip_extra_context_tokens = self.attention_norm(clip_extra_context_tokens)

#         clip_extra_context_tokens = self.ffn(clip_extra_context_tokens)
        

#         clip_extra_context_tokens = self.norm_out(clip_extra_context_tokens)
#         clip_extra_context_tokens = self.proj_out(clip_extra_context_tokens)
        
#         clip_extra_context_tokens = clip_extra_context_tokens.to(query.dtype)

#         text_embeddings = 1.0*clip_extra_context_tokens + text_embeddings
#         return clip_extra_context_tokens

class ZeroConv2d(nn.Conv2d):
    def __init__(self, in_ch, out_ch):
        super().__init__(in_ch, out_ch, kernel_size=1, bias=True)
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.img_size = img_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.grid_size = (img_size[0] // self.patch_size[0], img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=self.patch_size,
                              stride=self.patch_size)

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.proj(x)           # (B, C, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        return x
    
class HintEncoder(nn.Module):
    def __init__(self, in_ch=3, base=64, img_size=512, patch_size=16, embed_dim=768):
        super().__init__()
        # 512->512
        self.zero_conv_in = ZeroConv2d(in_ch, base)

        self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=base,
                embed_dim=embed_dim,
            )
        
        self.pos_embeddings = nn.Parameter(torch.randn(1, 1024, embed_dim) / embed_dim ** 0.5)

        self.self_attn = SelfAttBlock(embed_dim, int(embed_dim/64), True)
        self.to_embed = nn.Linear(embed_dim, 1536)
        self.norm = nn.LayerNorm(1536)

    def forward(self, x):
        x = self.zero_conv_in(x)
        x = self.patch_embed(x) 

        pos_embeds = self.pos_embeddings.repeat(x.shape[0],1,1)
        x = x + pos_embeds  
        x = self.self_attn(x) 

        x = self.norm(self.to_embed(x))
        return x

import random
from flash_attn import flash_attn_func, flash_attn_varlen_func
class WindowAwareLinearProjection(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=512, heads=32, train_text=True):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_embeddings_dim = clip_embeddings_dim
        self.heads = int(cross_attention_dim/64)
        self.image_dim = 1536
        self.train_text = train_text

        self.cross_attention_dim=2048
        self.heads = int(self.cross_attention_dim/64)
        self.model_global = SelfAttBlock(self.cross_attention_dim, self.heads, True)
        self.model_global_out = nn.Sequential(
            nn.LayerNorm(self.cross_attention_dim),
            nn.Linear(self.cross_attention_dim, self.cross_attention_dim)
        )
        if self.train_text:
            self.phrase_query = nn.Parameter(torch.randn(1, 128, self.cross_attention_dim))
            self.phrase_model = CrossAttBlock(self.cross_attention_dim, self.heads, True)
            self.phrase_self = SelfAttBlock(self.cross_attention_dim, self.heads, True)
            self.phrase_out = nn.Sequential(
                nn.LayerNorm(self.cross_attention_dim),
                nn.Linear(self.cross_attention_dim, self.cross_attention_dim)
            )

        if not self.train_text:
            self.cross_attention_dim = cross_attention_dim
            self.box_encoder = HintEncoder(in_ch=3, base=64, img_size=512, patch_size=16, embed_dim=768)
            self.condition_query = nn.Parameter(torch.randn(1, 512, self.cross_attention_dim))
            self.condition_integrate = CrossAttBlock(cross_attention_dim, heads, True)
            self.condition_self = SelfAttBlock(cross_attention_dim, heads, True)
            self.condition_out = nn.Sequential(
                nn.LayerNorm(cross_attention_dim),
                nn.Linear(self.cross_attention_dim, 2048)
            )
    
    def forward(self, text_embeddings, dino_v2, batch=None, training=True, uncondition=False):
        #text_embeds
        embeds = text_embeddings
        text_embeddings_ori = text_embeddings

        batch_size = embeds.shape[0]

        if self.train_text:
            clip_extra_context_tokens = self.model_global(embeds)
            clip_extra_context_tokens = self.model_global_out(clip_extra_context_tokens)
            text_embeddings = 1.0*clip_extra_context_tokens + text_embeddings

            query = self.phrase_query.repeat(batch_size, 1, 1)
            condition_embeds = self.phrase_model(query, embeds)
            condition_embeds = self.phrase_self(condition_embeds)
            condition_embeds = self.phrase_out(condition_embeds)
            
            return text_embeddings, condition_embeds
        else:
            with torch.no_grad():
                clip_extra_context_tokens = self.model_global(embeds)
                clip_extra_context_tokens = self.model_global_out(clip_extra_context_tokens)
                text_embeddings = 1.0*clip_extra_context_tokens + text_embeddings

        if not self.train_text:
            mask_q = None
            mask_kv = None

            box_mask = batch['box_mask']
            box_embeddings = self.box_encoder(box_mask)
            
            structure_ref = batch['structure']
            sturcture_ref_embeds = dino_v2(structure_ref).last_hidden_state

            b, n, c, h, w = batch['appearance'].shape
            appearance_ref = batch['appearance'].reshape(b*n,c,h,w)
            image_reference_embeds = dino_v2(appearance_ref).last_hidden_state
            image_reference_embeds = image_reference_embeds.reshape(b,n,261,1536)
            mask = torch.arange(image_reference_embeds.size(1)).unsqueeze(0).to(image_reference_embeds.device) < batch['appearance_num'].unsqueeze(1)
            mask = mask.unsqueeze(-1).unsqueeze(-1)
            image_reference_embeds = image_reference_embeds * mask
            image_reference_embeds = image_reference_embeds.reshape(b,n*261,1536)

            b, n, c = box_embeddings.shape
            choices = torch.randint(0, 5, (b,)) 
            concatenated_tensors = []
            for i in range(b):
                choice = choices[i]
                if choice == 0:
                    pad = torch.zeros(box_embeddings[i].shape[0]+image_reference_embeds[i].shape[0], c).to(sturcture_ref_embeds.device)
                    concatenated_tensors.append(torch.cat([sturcture_ref_embeds[i], pad], dim=0))
                elif choice == 1:
                    pad = torch.zeros(sturcture_ref_embeds[i].shape[0], c).to(sturcture_ref_embeds.device)
                    concatenated_tensors.append(torch.cat([box_embeddings[i], image_reference_embeds[i], pad], dim=0))
                elif choice == 2:
                    pad = torch.zeros(sturcture_ref_embeds[i].shape[0]+image_reference_embeds[i].shape[0], c).to(sturcture_ref_embeds.device)
                    concatenated_tensors.append(torch.cat([box_embeddings[i], pad], dim=0))
                elif choice == 3:
                    pad = torch.zeros(box_embeddings[i].shape[0], c).to(sturcture_ref_embeds.device)
                    concatenated_tensors.append(torch.cat([sturcture_ref_embeds[i], image_reference_embeds[i], pad], dim=0))
                elif choice == 4:
                    pad = torch.zeros(box_embeddings[i].shape[0]+sturcture_ref_embeds[i].shape[0], c).to(sturcture_ref_embeds.device)
                    concatenated_tensors.append(torch.cat([image_reference_embeds[i], pad], dim=0))

            contro_embeds = torch.stack(concatenated_tensors)
            mask_kv = (contro_embeds.abs().sum(dim=-1) != 0)

            query = self.condition_query.repeat(batch_size, 1, 1)
            condition_embeddings = self.condition_integrate(query, contro_embeds, mask_q, mask_kv)
            condition_embeddings = self.condition_self(condition_embeddings)
            condition_embeddings = self.condition_out(condition_embeddings)
           
            return text_embeddings, condition_embeddings
    
from torch.utils.checkpoint import checkpoint
class ResCrossAttBlock(nn.Module):
    def __init__(self, dim, num_heads, gradient_checkpointing=True):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.ln1 = nn.LayerNorm(dim)
        self.att = CrossAttention(dim, num_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim)
        self.ln3 = nn.LayerNorm(dim)
        self.se_att = SelfAttention(dim, num_heads)
    
    def forward(self, x, c):
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, x, c, use_reentrant=False) #checkpoint() 通过 重新计算前向传播的部分计算图，减少了中间激活值的存储需求，从而降低显存占用。
        else:
            return self._forward(x, c)
        
    def _forward(self, x, c):
        x = x + self.att(self.ln1(x), c)
        x = self.ln3(x)
        x = x + self.se_att(x, x)
        x = x + self.mlp(self.ln2(x))
        return x

class MaskAttBlock(nn.Module):
    def __init__(self, dim, num_heads, gradient_checkpointing=True):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.ln1 = nn.LayerNorm(dim)
        self.att = CrossAttention(dim, num_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim)
    
    def forward(self, x, c, mask=None):
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, x, c, mask=mask, use_reentrant=False) #checkpoint() 通过 重新计算前向传播的部分计算图，减少了中间激活值的存储需求，从而降低显存占用。
        else:
            return self._forward(x, c)
        
    def _forward(self, x, c, mask):
        x = x + self.att(self.ln1(x), self.ln1(c), mask)
        x = x + self.mlp(self.ln2(x))
        return x
    
class CrossAttBlock(nn.Module):
    def __init__(self, dim, num_heads, gradient_checkpointing=True):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.ln1 = nn.LayerNorm(dim)
        self.att = CrossAttention(dim, num_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim)
        # self.ln3 = nn.LayerNorm(dim)
    
    def forward(self, x, c, mask_q=None, mask_kv=None):
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, x, c, mask_q, mask_kv, use_reentrant=False) #checkpoint() 通过 重新计算前向传播的部分计算图，减少了中间激活值的存储需求，从而降低显存占用。
        else:
            return self._forward(x, c)
        
    def _forward(self, x, c, mask_q, mask_kv):
        x = x + self.att(self.ln1(x), c, mask_q, mask_kv)
        x = x + self.mlp(self.ln2(x))
        return x

class SelfAttBlock(nn.Module):
    def __init__(self, dim, num_heads, gradient_checkpointing=True):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.ln1 = nn.LayerNorm(dim)
        self.att = CrossAttention(dim, num_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim)
    
    def forward(self, x, mask_q=None, mask_kv=None):
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, x, mask_q, mask_kv, use_reentrant=False) #checkpoint() 通过 重新计算前向传播的部分计算图，减少了中间激活值的存储需求，从而降低显存占用。
        else:
            return self._forward(x)
        
    def _forward(self, x, mask_q, mask_kv):
        x = x + self.att(self.ln1(x), self.ln1(x), mask_q, mask_kv)
        x = x + self.mlp(self.ln2(x))
        return x

def unpad_input(hidden_states, attention_mask):
    """
    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices of non-masked tokens from the flattened input sequence.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()  #attention_mask展平后所有非0位置的index
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    # TD [2022-03-04] We don't want to index with a bool mask, because Pytorch will expand the
    # bool mask, then call nonzero to get the indices, then index with those. The indices is @dim
    # times larger than it needs to be, wasting memory. It's faster and more memory-efficient to
    # index with integer indices. Moreover, torch's index is a bit slower than it needs to be,
    # so we write custom forward and backward to make it a bit faster.
    return (
        index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )

def attention(q, k, v, mask_q=None, mask_kv=None, dropout=0, causal=False, window_size=(-1, -1), backend='flash-attn'):
    # q: (B, N, H, D)
    # k: (B, M, H, D)
    # v: (B, M, H, D)
    # mask_q: (B, N)
    # mask_kv: (B, M)
    # return: (B, N, H, D)

    B, N, H, D = q.shape
    M = k.shape[1]

    # kiui.lo(q, k, v)

    if causal: 
        assert N == 1 or N == M, 'Causal mask only supports self-attention'

    ### unmasked case (usually inference)
    ### will ignore window_size except flash-attn impl. Only provide the effective window!
    if mask_q is None and mask_kv is None:
        if backend == 'flash-attn' and FLASH_ATTN_AVAILABLE:
            return flash_attn_func(q, k, v, dropout, causal=causal, window_size=window_size) # [B, N, H, D]
        else: # naive implementation
            q = q.transpose(1, 2).reshape(B * H, N, D)
            k = k.transpose(1, 2).reshape(B * H, M, D)
            v = v.transpose(1, 2).reshape(B * H, M, D)
            w = torch.bmm(q, k.transpose(1, 2)) / (D ** 0.5) # [B*H, N, M]
            if causal and N > 1:
                causal_mask = torch.full((N, M), float('-inf'), device=w.device, dtype=w.dtype)
                causal_mask = torch.triu(causal_mask, diagonal=1)
                w = w + causal_mask.unsqueeze(0)
            w = F.softmax(w, dim=-1)
            if dropout > 0:
                w = F.dropout(w, p=dropout)
            out = torch.bmm(w, v) # [B*H, N, D]
            out = out.reshape(B, H, N, D).transpose(1, 2).contiguous() # [B, N, H, D]
            return out
    
    ### at least one of q or kv is masked (training)
    ### only support flash-attn for now...
    if mask_q is None:
        mask_q = torch.ones(B, N, dtype=torch.bool, device=q.device)
    elif mask_kv is None:
        mask_kv = torch.ones(B, M, dtype=torch.bool, device=q.device)

    if FLASH_ATTN_AVAILABLE:
        # unpad (gather) input
        # mask_q: [B, N], first row has N1 1s, second row has N2 1s, ...
        # indices: [Ns,], Ns = N1 + N2 + ...
        # cu_seqlens_q: [B+1,], (0, N1, N1+N2, ...), cu=cumulative
        # max_len_q: scalar, max(N1, N2, ...)
        q, indices_q, cu_seqlens_q, max_len_q = unpad_input(q, mask_q)
        k, indices_kv, cu_seqlens_kv, max_len_kv = unpad_input(k, mask_kv)
        v = index_first_axis(v.reshape(-1, H, D), indices_kv) # same indice as k

        # call varlen_func
        out = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_kv,
            max_seqlen_q=max_len_q,
            max_seqlen_k=max_len_kv,
            dropout_p=dropout,
            causal=causal,
            window_size=window_size,
        )

        # pad (put back) output
        out = pad_input(out, indices_q, B, N)
        return out
    else:
        raise NotImplementedError('masked attention requires flash_attn!')
    
class CrossAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, input_dim=None, context_dim=None, output_dim=None, dropout=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim if input_dim is not None else hidden_dim
        self.context_dim = context_dim if context_dim is not None else hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0, 'hidden_dim must be divisible by num_heads'
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.context_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.context_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.output_dim)
    
    def forward(self, x, context, mask_q=None, mask_kv=None):
        # x: [B, N, C]
        # context: [B, M, C']
        # mask_q: [B, N]
        # mask_kv: [B, M]
        B, N, C = x.shape
        M = context.shape[1]
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim)#.transpose(1, 2)
        k = self.k_proj(context).reshape(B, M, self.num_heads, self.head_dim)#.transpose(1, 2)
        v = self.v_proj(context).reshape(B, M, self.num_heads, self.head_dim)#.transpose(1, 2)
        x = attention(q,k,v,mask_q,mask_kv) 
        x = x.reshape(B, -1, C)
        x = self.out_proj(x)
        return x
    
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, input_dim=None, context_dim=None, output_dim=None, dropout=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim if input_dim is not None else hidden_dim
        self.context_dim = context_dim if context_dim is not None else hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0, 'hidden_dim must be divisible by num_heads'
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.context_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.context_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.output_dim)
    
    def forward(self, x, context, mask_q=None, mask_kv=None):
        # x: [B, N, C]
        # context: [B, M, C']
        # mask_q: [B, N]
        # mask_kv: [B, M]
        B, N, C = x.shape
        M = context.shape[1]
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
        ) # [B, N, H, D]
        x = x.transpose(1, 2).reshape(B, -1, C)
        x = self.out_proj(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, min(dim * mult * 2, 8192)),
            GEGLU(),
            nn.Linear(min(dim * mult, 8192//2), dim)
        )
    def forward(self, x):
        return self.net(x)
    
class MLPProjModel(torch.nn.Module):
    """SD model with image prompt"""
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024):
        super().__init__()
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            torch.nn.GELU(),
            torch.nn.Linear(clip_embeddings_dim, cross_attention_dim),
            torch.nn.LayerNorm(cross_attention_dim)
        )
        
    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens

#only dino image
# class IPAdapter:
#     def __init__(self, sd_pipe, ip_ckpt, device, num_tokens=24, target_blocks=["block"]):
#         self.device = device
#         self.ip_ckpt = ip_ckpt
#         self.num_tokens = num_tokens
#         self.target_blocks = target_blocks

#         self.pipe = sd_pipe.to(self.device)
#         self.set_ip_adapter()

#         # image proj model
#         self.text_embedding_projector = self.init_proj()

#         self.load_ip_adapter()

#     def init_proj(self):
#         text_embedding_projector = WindowAwareLinearProjection(2048, 512, 32).to(self.device, dtype=torch.float16)
#         return text_embedding_projector
    
#     def set_ip_adapter(self):
#         unet = self.pipe.unet
#         attn_procs = {}
#         for name in unet.attn_processors.keys():
#             cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
#             if name.startswith("mid_block"):
#                 hidden_size = unet.config.block_out_channels[-1]
#             elif name.startswith("up_blocks"):
#                 block_id = int(name[len("up_blocks.")])
#                 hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
#             elif name.startswith("down_blocks"):
#                 block_id = int(name[len("down_blocks.")])
#                 hidden_size = unet.config.block_out_channels[block_id]
#             if cross_attention_dim is None:
#                 attn_procs[name] = AttnProcessor()
#             else:
#                 selected = False
#                 for block_name in self.target_blocks:
#                     if block_name in name:
#                         selected = True
#                         break
#                 if selected:
#                     attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=self.num_tokens, skip=False).to(self.device, dtype=torch.float16)
#                 else:
#                     attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=self.num_tokens, skip=True).to(self.device, dtype=torch.float16)
#         unet.set_attn_processor(attn_procs)
#         if hasattr(self.pipe, "controlnet"):
#             if isinstance(self.pipe.controlnet, MultiControlNetModel):
#                 for controlnet in self.pipe.controlnet.nets:
#                     controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
#             else:
#                 self.pipe.controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))

#     def load_ip_adapter(self):
#         if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
#             state_dict = {"image_proj": {}, "ip_adapter": {}}
#             with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
#                 for key in f.keys():
#                     if key.startswith("image_proj."):
#                         state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
#                     elif key.startswith("ip_adapter."):
#                         state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
#         else:
#             state_dict = torch.load(self.ip_ckpt, map_location="cpu")
#         self.text_embedding_projector.load_state_dict(state_dict["text_embedding_projector"], strict=True)
#         ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
#         ip_layers.load_state_dict(state_dict["ip_adapter"], strict=True)
#         print("successfully load weights from cpkt")

#     @torch.inference_mode()
#     def get_image_embeds(self, pil_image=None, clip_image_embeds=None, content_prompt_embeds=None, image_prompt_embeds_ = None, face_images=None):
#         if pil_image is not None:
#             if isinstance(pil_image, Image.Image):
#                 pil_image = [pil_image]
#             clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
#             clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
#             clip_image_last_hidden_states = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).last_hidden_state
#         else:
#             clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
        
#         if face_images is not None:
#             clip_face_image_embeds = []
#             clip_face_image_last_hidden_states = []
#             for face_image  in face_images:
#                 clip_face_image = self.clip_image_processor(images=face_image, return_tensors="pt").pixel_values
#                 clip_face_image_embed = self.image_encoder(clip_face_image.to(self.device, dtype=torch.float16)).image_embeds.squeeze(0)
#                 clip_face_image_last_hidden_state = self.image_encoder(clip_face_image.to(self.device, dtype=torch.float16)).last_hidden_state.squeeze(0)
#                 clip_face_image_embeds.append(clip_face_image_embed)
#                 clip_face_image_last_hidden_states.append(clip_face_image_last_hidden_state)
#             clip_face_image_embeds = torch.stack(clip_face_image_embeds).unsqueeze(0)
#             clip_face_image_last_hidden_states = torch.stack(clip_face_image_last_hidden_states).unsqueeze(0)

#         if content_prompt_embeds is not None:
#             clip_image_embeds = clip_image_embeds - content_prompt_embeds
        
#         face_number_per_image = 4
#         clip_image_embeds_2, mask = self.image_adjust_model(clip_image_embeds, image_prompt_embeds_, clip_image_last_hidden_states, clip_face_image_embeds, clip_face_image_last_hidden_states, face_number_per_image, None)
#         # attn_map = clip_image_embeds_2
#         # tensor = attn_map.cpu().numpy()
#         # plt.figure(figsize=(8, 6))
#         # sns.heatmap(tensor, cmap="viridis", annot=False, cbar=True)
#         # plt.title('Compensation Features with Different LayerNorm Position')
#         # plt.savefig('Compensation Features with Different LayerNorm Position.png')
#         # plt.title('Original Image Features')
#         # plt.savefig('Original Image Features.png')

#         clip_image_embeds = clip_image_embeds + self.adjust_scale * clip_image_embeds_2
#         clip_image_embeds = clip_image_embeds + torch.randn(1, 1280).to('cuda')
#         clip_image_embeds = clip_image_embeds.to(clip_image_embeds_2.dtype)

#         image_prompt_embeds_layout = self.image_proj_model(clip_image_embeds) 
#         #mask1 = (torch.rand(image_prompt_embeds_layout.shape) > 0.5).to(image_prompt_embeds_layout.dtype).to('cuda')
#         # image_prompt_embeds_layout = image_prompt_embeds_layout + torch.randn(1, 4, 2048).to('cuda')
#         image_prompt_embeds_layout = torch.cat([image_prompt_embeds_layout, image_prompt_embeds_layout], dim=1)

#         uncond_image_prompt_embeds_layout = self.image_proj_model(torch.zeros_like(clip_image_embeds))
#         uncond_image_prompt_embeds_layout = torch.cat([uncond_image_prompt_embeds_layout, uncond_image_prompt_embeds_layout], dim=1)

#         return image_prompt_embeds_layout, uncond_image_prompt_embeds_layout

#     def set_scale(self, scale):
#         for attn_processor in self.pipe.unet.attn_processors.values():
#             if isinstance(attn_processor, IPAttnProcessor):
#                 attn_processor.scale = scale
    
#     def set_scale_lora(self, scale):
#         for attn_processor in self.pipe.unet.attn_processors.values():
#             if isinstance(attn_processor, IPAttnProcessor):
#                 attn_processor.scale_lora = scale

#     def generate(
#         self,
#         pil_image=None,
#         clip_image_embeds=None,
#         prompt=None,
#         negative_prompt=None,
#         scale=1.0,
#         num_samples=4,
#         seed=None,
#         guidance_scale=7.5,
#         num_inference_steps=30,
#         neg_content_emb=None,
#         **kwargs,
#     ):
#         self.set_scale(scale)

#         if pil_image is not None:
#             num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
#         else:
#             num_prompts = clip_image_embeds.size(0)

#         if prompt is None:
#             prompt = "best quality, high quality"
#         if negative_prompt is None:
#             negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

#         if not isinstance(prompt, List):
#             prompt = [prompt] * num_prompts
#         if not isinstance(negative_prompt, List):
#             negative_prompt = [negative_prompt] * num_prompts

#         image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
#             pil_image=pil_image, clip_image_embeds=clip_image_embeds, content_prompt_embeds=neg_content_emb
#         )
#         bs_embed, seq_len, _ = image_prompt_embeds.shape
#         image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
#         image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
#         uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
#         uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

#         with torch.inference_mode():
#             prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
#                 prompt,
#                 device=self.device,
#                 num_images_per_prompt=num_samples,
#                 do_classifier_free_guidance=True,
#                 negative_prompt=negative_prompt,
#             )
#             prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
#             negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

#         generator = get_generator(seed, self.device)

#         images = self.pipe(
#             prompt_embeds=prompt_embeds,
#             negative_prompt_embeds=negative_prompt_embeds,
#             guidance_scale=guidance_scale,
#             num_inference_steps=num_inference_steps,
#             generator=generator,
#             **kwargs,
#         ).images

#         return images


# class IPAdapterXL(IPAdapter):
#     """SDXL"""

#     def generate(
#         self,
#         pil_image=None,
#         face_images=None,
#         image_prompt=None,
#         prompt=None,
#         negative_prompt=None,
#         scale=1.0,
#         scale_lora=1.0,
#         adjust_scale=1.0,
#         num_samples=4,
#         seed=None,
#         num_inference_steps=30,
#         neg_content_emb=None,
#         neg_content_prompt=None,
#         neg_content_scale=1.0,
#         **kwargs,
#     ):
#         self.set_scale(scale)

#         num_prompts = 1 #if isinstance(pil_image, Image.Image) else len(pil_image)

#         if prompt is None:
#             prompt = "best quality, high quality"
#         if negative_prompt is None:
#             negative_prompt = "text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry, bad anatomy"

#         if not isinstance(prompt, List):
#             prompt = [prompt] * num_prompts
#         if not isinstance(negative_prompt, List):
#             negative_prompt = [negative_prompt] * num_prompts

#         image_prompt_embeds = pil_image
#         bs_embed, seq_len, _ = image_prompt_embeds.shape
#         image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
#         image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
#         uncond_image_prompt_embeds = torch.zeros_like(pil_image)
#         uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
#         uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

#         # image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image)
#         # bs_embed, seq_len, _ = image_prompt_embeds.shape
#         # image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
#         # image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
#         # uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
#         # uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

#         with torch.inference_mode():
#             (
#                 prompt_embeds,
#                 negative_prompt_embeds,
#                 pooled_prompt_embeds,
#                 negative_pooled_prompt_embeds,
#             ) = self.pipe.encode_prompt(
#                 prompt,
#                 num_images_per_prompt=num_samples,
#                 do_classifier_free_guidance=True,
#                 negative_prompt=negative_prompt,
#             )

#             image_prompt_embeds = image_prompt_embeds.to(prompt_embeds.dtype)
#             prompt_embeds, image_prompt_embeds = self.text_embedding_projector(prompt_embeds, image_prompt_embeds)
#             prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
#              # uncond_image_prompt_embeds = torch.zeros_like(image_prompt_embeds)
#             uncond_image_prompt_embeds = uncond_image_prompt_embeds.to(prompt_embeds.dtype)
#             _, uncond_image_prompt_embeds = self.text_embedding_projector(negative_prompt_embeds, uncond_image_prompt_embeds)
#             negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

#         self.generator = get_generator(seed, self.device)
        
#         images = self.pipe(
#             prompt_embeds=prompt_embeds,
#             negative_prompt_embeds=negative_prompt_embeds,
#             pooled_prompt_embeds=pooled_prompt_embeds,
#             negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
#             num_inference_steps=num_inference_steps,
#             generator=self.generator,
#             **kwargs,
#         ).images
#         return images

class IPAdapter:
    def __init__(self, sd_pipe, ip_ckpt, device, num_tokens=24, target_blocks=["block"], use_control=False):
        self.device = device
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens
        self.target_blocks = target_blocks
        self.use_control = use_control

        self.pipe = sd_pipe.to(self.device)
        self.set_ip_adapter()

        # image proj model
        self.text_embedding_projector = self.init_proj()

        self.load_ip_adapter()

    def init_proj(self):
        if self.use_control:
            text_embedding_projector = WindowAwareLinearProjection(1536, 512, 32, False).to(self.device, dtype=torch.float16)
        else:
            text_embedding_projector = WindowAwareLinearProjection(1536, 512, 32, True).to(self.device, dtype=torch.float16)
        return text_embedding_projector
    
    def set_ip_adapter(self):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                selected = False
                for block_name in self.target_blocks:
                    if block_name in name:
                        selected = True
                        break
                if selected:
                    attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=self.num_tokens, skip=False).to(self.device, dtype=torch.float16)
                else:
                    attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=self.num_tokens, skip=True).to(self.device, dtype=torch.float16)
        unet.set_attn_processor(attn_procs)
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
            else:
                self.pipe.controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))

    def load_ip_adapter(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        self.text_embedding_projector.load_state_dict(state_dict["text_embedding_projector"], strict=True)
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"], strict=True)
        print("successfully load weights from cpkt")

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None, content_prompt_embeds=None, image_prompt_embeds_ = None, face_images=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
            clip_image_last_hidden_states = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).last_hidden_state
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
        
        if face_images is not None:
            clip_face_image_embeds = []
            clip_face_image_last_hidden_states = []
            for face_image  in face_images:
                clip_face_image = self.clip_image_processor(images=face_image, return_tensors="pt").pixel_values
                clip_face_image_embed = self.image_encoder(clip_face_image.to(self.device, dtype=torch.float16)).image_embeds.squeeze(0)
                clip_face_image_last_hidden_state = self.image_encoder(clip_face_image.to(self.device, dtype=torch.float16)).last_hidden_state.squeeze(0)
                clip_face_image_embeds.append(clip_face_image_embed)
                clip_face_image_last_hidden_states.append(clip_face_image_last_hidden_state)
            clip_face_image_embeds = torch.stack(clip_face_image_embeds).unsqueeze(0)
            clip_face_image_last_hidden_states = torch.stack(clip_face_image_last_hidden_states).unsqueeze(0)

        if content_prompt_embeds is not None:
            clip_image_embeds = clip_image_embeds - content_prompt_embeds
        
        face_number_per_image = 4
        clip_image_embeds_2, mask = self.image_adjust_model(clip_image_embeds, image_prompt_embeds_, clip_image_last_hidden_states, clip_face_image_embeds, clip_face_image_last_hidden_states, face_number_per_image, None)
        # attn_map = clip_image_embeds_2
        # tensor = attn_map.cpu().numpy()
        # plt.figure(figsize=(8, 6))
        # sns.heatmap(tensor, cmap="viridis", annot=False, cbar=True)
        # plt.title('Compensation Features with Different LayerNorm Position')
        # plt.savefig('Compensation Features with Different LayerNorm Position.png')
        # plt.title('Original Image Features')
        # plt.savefig('Original Image Features.png')

        clip_image_embeds = clip_image_embeds + self.adjust_scale * clip_image_embeds_2
        clip_image_embeds = clip_image_embeds + torch.randn(1, 1280).to('cuda')
        clip_image_embeds = clip_image_embeds.to(clip_image_embeds_2.dtype)

        image_prompt_embeds_layout = self.image_proj_model(clip_image_embeds) 
        #mask1 = (torch.rand(image_prompt_embeds_layout.shape) > 0.5).to(image_prompt_embeds_layout.dtype).to('cuda')
        # image_prompt_embeds_layout = image_prompt_embeds_layout + torch.randn(1, 4, 2048).to('cuda')
        image_prompt_embeds_layout = torch.cat([image_prompt_embeds_layout, image_prompt_embeds_layout], dim=1)

        uncond_image_prompt_embeds_layout = self.image_proj_model(torch.zeros_like(clip_image_embeds))
        uncond_image_prompt_embeds_layout = torch.cat([uncond_image_prompt_embeds_layout, uncond_image_prompt_embeds_layout], dim=1)

        return image_prompt_embeds_layout, uncond_image_prompt_embeds_layout

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale
    
    def set_scale_lora(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale_lora = scale

    def generate(
        self,
        pil_image=None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        neg_content_emb=None,
        **kwargs,
    ):
        self.set_scale(scale)

        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = clip_image_embeds.size(0)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=clip_image_embeds, content_prompt_embeds=neg_content_emb
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images


class IPAdapterXL(IPAdapter):
    """SDXL"""

    def generate(
        self,
        pil_image=None,
        face_images=None,
        image_prompt=None,
        prompt=None,
        appearance=None,
        box=None,
        box_mask=None,
        negative_prompt=None,
        scale=1.0,
        text_scale=1.0,
        num_samples=4,
        seed=None,
        num_inference_steps=30,
        neg_content_emb=None,
        neg_content_prompt=None,
        neg_content_scale=1.0,
        **kwargs,
    ):
        self.set_scale(scale)

        num_prompts = 1 #if isinstance(pil_image, Image.Image) else len(pil_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry, bad anatomy"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )

            embeds = prompt_embeds
            clip_extra_context_tokens = self.text_embedding_projector.model_global(prompt_embeds)
            clip_extra_context_tokens = self.text_embedding_projector.model_global_out(clip_extra_context_tokens)
            prompt_embeds = text_scale*clip_extra_context_tokens + prompt_embeds
            
            if box is None and pil_image is None and appearance is None:
                query = self.text_embedding_projector.phrase_query.repeat(num_samples, 1, 1)
                condition_embeddings = self.text_embedding_projector.phrase_model(query, embeds)
                condition_embeddings = self.text_embedding_projector.phrase_self(condition_embeddings)
                condition_embeddings = self.text_embedding_projector.phrase_out(condition_embeddings)
            else:
                # image_prompt_embeds = image_prompt_embeds.to(prompt_embeds.dtype)
                mask_q = None
                mask_kv = None
                if box is not None:
                    box = box.half().cuda().unsqueeze(0)
                    box = box.repeat(num_samples, 1, 1, 1)
                    box_embeddings = self.text_embedding_projector.box_encoder(box)
                else:
                    box_embeddings=None

                if pil_image is not None:
                    sturcture_ref_embeds = pil_image.repeat(num_samples, 1, 1)
                else:
                    sturcture_ref_embeds=None

                if appearance is not None:
                    image_reference_embeds = appearance.repeat(num_samples, 1, 1)
                else:
                    image_reference_embeds=None

                c = 1536
                concatenated_tensors = []
                if sturcture_ref_embeds is not None:
                    concatenated_tensors.append(sturcture_ref_embeds)
                if box_embeddings is not None:
                    concatenated_tensors.append(box_embeddings)
                if image_reference_embeds is not None:
                    concatenated_tensors.append(image_reference_embeds)

                contro_embeds = torch.cat(concatenated_tensors, dim=1)
                pad = torch.zeros(num_samples, 2851-contro_embeds.shape[1], c).to(prompt_embeds.device)
                if pad.numel() != 0:
                    contro_embeds = torch.cat([contro_embeds, pad],dim=1)
                contro_embeds = contro_embeds.to(prompt_embeds.dtype)
                
                mask_kv = (contro_embeds.abs().sum(dim=-1) != 0)
                query = self.text_embedding_projector.condition_query.repeat(num_samples, 1, 1)
                condition_embeddings = self.text_embedding_projector.condition_integrate(query, contro_embeds, mask_q, mask_kv)
                condition_embeddings = self.text_embedding_projector.condition_self(condition_embeddings)
                condition_embeddings = self.text_embedding_projector.condition_out(condition_embeddings)

            prompt_embeds = torch.cat([prompt_embeds, condition_embeddings], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, torch.zeros_like(condition_embeddings)], dim=1)

        self.generator = get_generator(seed, self.device)
        
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=self.generator,
            **kwargs,
        ).images
        return images
    
class IPAdapterPlus(IPAdapter):
    """IP-Adapter with fine-grained features"""

    def init_proj(self):
        image_proj_model = Resampler(
            dim=self.pipe.unet.config.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds


class IPAdapterFull(IPAdapterPlus):
    """IP-Adapter with full features"""

    def init_proj(self):
        image_proj_model = MLPProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.hidden_size,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model


class IPAdapterPlusXL(IPAdapter):
    """SDXL"""

    def init_proj(self):
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds

    def generate(
        self,
        pil_image,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)

        num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images
