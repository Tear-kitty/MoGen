from __future__ import annotations

from typing import Iterable, Tuple

import torch
from diffusers.pipelines.controlnet import MultiControlNetModel

from .attention_processor import AttnProcessor2_0, ControlNetAttnProcessor2_0, MoGenAttnProcessor2_0


def hidden_size_for_unet_attention(unet, name: str) -> int:
    if name.startswith("mid_block"):
        return unet.config.block_out_channels[-1]
    if name.startswith("up_blocks"):
        block_id = int(name[len("up_blocks.")])
        return list(reversed(unet.config.block_out_channels))[block_id]
    if name.startswith("down_blocks"):
        block_id = int(name[len("down_blocks.")])
        return unet.config.block_out_channels[block_id]
    raise ValueError(f"Unsupported UNet attention processor name: {name}")


def is_selected_block(name: str, target_blocks: Iterable[str]) -> bool:
    return any(block_name in name for block_name in target_blocks)


def set_mogen_attention_processors(
    pipe_or_unet,
    target_blocks: Iterable[str],
    num_tokens: int,
    train_text: bool,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.nn.ModuleList:
    """Install MoGen processors into an SDXL UNet and return them as a ModuleList."""
    unet = getattr(pipe_or_unet, "unet", pipe_or_unet)
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor2_0()
            continue

        hidden_size = hidden_size_for_unet_attention(unet, name)
        selected = is_selected_block(name, target_blocks)
        proc = MoGenAttnProcessor2_0(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            num_tokens=num_tokens,
            skip=not selected,
            train_text=train_text,
        )
        if selected:
            layer_name = name.split(".processor")[0]
            init_state = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            proc.load_state_dict(init_state, strict=False)
        attn_procs[name] = proc.to(device=device, dtype=dtype)
    unet.set_attn_processor(attn_procs)

    pipe = pipe_or_unet if hasattr(pipe_or_unet, "unet") else None
    if pipe is not None and hasattr(pipe, "controlnet"):
        if isinstance(pipe.controlnet, MultiControlNetModel):
            for controlnet in pipe.controlnet.nets:
                controlnet.set_attn_processor(ControlNetAttnProcessor2_0(num_tokens=num_tokens))
        else:
            pipe.controlnet.set_attn_processor(ControlNetAttnProcessor2_0(num_tokens=num_tokens))

    return torch.nn.ModuleList(unet.attn_processors.values())
