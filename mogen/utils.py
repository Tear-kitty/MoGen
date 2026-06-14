from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch

IMG_EXTENSIONS = (
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".ppm", ".PPM",
    ".bmp", ".BMP", ".tif", ".tiff", ".TIF", ".TIFF",
)

_NUM_MAP = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
}
_NUM_RE = re.compile(r"^\s*(" + "|".join(_NUM_MAP.keys()) + r")\b", re.IGNORECASE)


def is_image_file(path: str | os.PathLike[str]) -> bool:
    return str(path).endswith(IMG_EXTENSIONS)


def safe_stem(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^\w\-.\u4e00-\u9fff]+", "_", text.strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return (text or "sample")[:max_len]


def extract_count_from_name(path: str | os.PathLike[str], default: int = 1) -> int:
    stem = Path(path).stem
    m = _NUM_RE.match(stem)
    return _NUM_MAP[m.group(1).lower()] if m else default


def extract_count_words_and_numbers(paths: Iterable[str | os.PathLike[str]]) -> Tuple[List[str], List[int]]:
    words: List[str] = []
    nums: List[int] = []
    for path in paths:
        stem = Path(path).stem
        m = _NUM_RE.match(stem)
        if m:
            word = m.group(1).lower()
            words.append(word + " ")
            nums.append(_NUM_MAP[word])
        else:
            words.append("one ")
            nums.append(1)
    return words, nums


def get_generator(seed: Optional[int], device: str | torch.device) -> Optional[torch.Generator]:
    if seed is None:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def str2bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected a boolean value, got: {value!r}")


def dtype_from_string(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32", "full"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def latest_checkpoint_dir(output_dir: str | os.PathLike[str], mode: Optional[str] = None) -> Optional[str]:
    root = Path(output_dir)
    if not root.exists():
        return None
    candidates = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if mode is not None and not name.startswith(f"{mode}-checkpoint-"):
            continue
        if "checkpoint-" not in name:
            continue
        try:
            step = int(name.rsplit("-", 1)[-1])
        except ValueError:
            continue
        candidates.append((step, child))
    if not candidates:
        return None
    return str(max(candidates, key=lambda item: item[0])[1])
