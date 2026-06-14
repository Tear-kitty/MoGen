from __future__ import annotations

import io
import random
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageFilter


def elastic_deform_pil(
    image: Image.Image,
    alpha: float = 18.0,
    sigma: float = 6.0,
    p: float = 0.15,
    rng: Optional[np.random.Generator] = None,
) -> Image.Image:
    """Apply a mild local warp to a PIL image.

    The default is intentionally conservative: it is strong enough to reduce
    overfitting to exact control edges, but not strong enough to change object
    identity or global layout.
    """
    rng = rng or np.random.default_rng()
    if rng.random() > p:
        return image

    image_rgb = image.convert("RGB")
    arr = np.asarray(image_rgb)
    h, w = arr.shape[:2]
    dx = rng.normal(0, 1, (h, w)).astype(np.float32)
    dy = rng.normal(0, 1, (h, w)).astype(np.float32)
    dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    warped = cv2.remap(arr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return Image.fromarray(warped)


def degrade_image_pil(
    image: Image.Image,
    p: float = 0.35,
    min_downsample_ratio: float = 0.35,
    blur_prob: float = 0.60,
    jpeg_prob: float = 0.35,
    noise_prob: float = 0.20,
    local_warp_prob: float = 0.15,
) -> Image.Image:
    """Global quality degradation for structure/appearance references.

    The recipe mixes low-resolution resize, blur, mild JPEG artifacts, light noise,
    and optional local warp. It keeps the semantic content while making references
    less pixel-perfect, which usually improves robustness of DINO-based controls.
    """
    if random.random() > p:
        return image.convert("RGB")

    img = image.convert("RGB")
    w, h = img.size

    # Low-resolution bottleneck followed by upsampling.
    ratio = random.uniform(min_downsample_ratio, 0.75)
    small_w = max(8, int(w * ratio))
    small_h = max(8, int(h * ratio))
    img = img.resize((small_w, small_h), Image.Resampling.BICUBIC)
    img = img.resize((w, h), Image.Resampling.BICUBIC)

    if random.random() < blur_prob:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.4, 1.6)))

    if random.random() < jpeg_prob:
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=random.randint(35, 75))
        buffer.seek(0)
        img = Image.open(buffer).convert("RGB")

    if random.random() < noise_prob:
        arr = np.asarray(img).astype(np.float32)
        noise = np.random.normal(0, random.uniform(1.5, 5.0), arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    img = elastic_deform_pil(
        img,
        alpha=random.uniform(8.0, 22.0),
        sigma=random.uniform(4.0, 8.0),
        p=local_warp_prob,
    )
    return img.convert("RGB")


def maybe_corrupt_count(
    count: int,
    error_prob: float = 0.20,
    max_count: int = 15,
) -> int:
    """Occasionally replace an appearance repeat count by a nearby wrong count."""
    count = max(1, min(max_count, int(count)))
    if random.random() >= error_prob:
        return count

    offsets = [-2, -1, 1, 2]
    weights = [0.15, 0.35, 0.35, 0.15]
    candidates = [max(1, min(max_count, count + off)) for off in offsets]
    candidates = [c for c in candidates if c != count]
    if not candidates:
        return count

    # Recompute weights after clipping/deduplication.
    weighted = []
    for off, weight in zip(offsets, weights):
        c = max(1, min(max_count, count + off))
        if c != count:
            weighted.append((c, weight))
    total = sum(w for _, w in weighted)
    r = random.random() * total
    acc = 0.0
    for value, weight in weighted:
        acc += weight
        if r <= acc:
            return value
    return weighted[-1][0]
