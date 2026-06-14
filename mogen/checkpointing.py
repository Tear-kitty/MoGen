from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch

PROJECTOR_KEYS = ("mogen_projector", "text_embedding_projector", "projector")
ATTENTION_KEYS = ("mogen_attention", "adapter_modules", "ip_adapter")


def _torch_load(path: str | Path) -> Dict[str, Any]:
    return torch.load(str(path), map_location="cpu")


def pick_state_dict(raw: Mapping[str, Any], keys: tuple[str, ...]) -> Optional[Mapping[str, torch.Tensor]]:
    for key in keys:
        if key in raw and isinstance(raw[key], Mapping):
            return raw[key]
    return None


def load_attention_state(
    checkpoint_path: str | Path,
    attention_modules: torch.nn.ModuleList,
    strict: bool = False,
) -> torch.nn.modules.module._IncompatibleKeys:
    raw = _torch_load(checkpoint_path)
    state = pick_state_dict(raw, ATTENTION_KEYS)
    if state is None:
        # Some checkpoints may already be the attention module state_dict.
        state = raw
    return attention_modules.load_state_dict(state, strict=strict)


def load_projector_state(
    checkpoint_path: str | Path,
    projector: torch.nn.Module,
    strict: bool = False,
) -> torch.nn.modules.module._IncompatibleKeys:
    raw = _torch_load(checkpoint_path)
    state = pick_state_dict(raw, PROJECTOR_KEYS)
    if state is None:
        state = raw
    return projector.load_state_dict(state, strict=strict)


def load_mogen_checkpoint(
    checkpoint_path: str | Path,
    projector: torch.nn.Module,
    attention_modules: torch.nn.ModuleList,
    strict: bool = True,
) -> Dict[str, Any]:
    raw = _torch_load(checkpoint_path)
    projector_state = pick_state_dict(raw, PROJECTOR_KEYS)
    attention_state = pick_state_dict(raw, ATTENTION_KEYS)
    if projector_state is None or attention_state is None:
        # Backward compatible path for old checkpoints that used only two keys.
        if projector_state is None:
            projector_state = raw.get("text_embedding_projector")
        if attention_state is None:
            attention_state = raw.get("ip_adapter")
    if projector_state is None or attention_state is None:
        raise KeyError(
            f"{checkpoint_path} does not contain MoGen checkpoint keys. "
            f"Expected one of {PROJECTOR_KEYS} and one of {ATTENTION_KEYS}."
        )
    proj_result = projector.load_state_dict(projector_state, strict=strict)
    attn_result = attention_modules.load_state_dict(attention_state, strict=strict)
    return {"projector": proj_result, "attention": attn_result, "metadata": raw.get("metadata", {})}


def load_matching_from_checkpoint(
    checkpoint_path: str | Path,
    projector: torch.nn.Module,
    attention_modules: torch.nn.ModuleList,
) -> Dict[str, Any]:
    """Non-strict initialization used by the control stage from a text-stage ckpt."""
    raw = _torch_load(checkpoint_path)
    results: Dict[str, Any] = {}
    projector_state = pick_state_dict(raw, PROJECTOR_KEYS)
    if projector_state is not None:
        results["projector"] = projector.load_state_dict(projector_state, strict=False)
    attention_state = pick_state_dict(raw, ATTENTION_KEYS)
    if attention_state is not None:
        results["attention"] = attention_modules.load_state_dict(attention_state, strict=False)
    return results


def save_mogen_checkpoint(
    checkpoint_path: str | Path,
    training_module: torch.nn.Module,
    accelerator,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    module = accelerator.unwrap_model(training_module)
    payload = {
        "mogen_projector": module.projector.state_dict(),
        "mogen_attention": module.attention_modules.state_dict(),
        "metadata": metadata or {},
    }
    accelerator.save(payload, str(checkpoint_path))
