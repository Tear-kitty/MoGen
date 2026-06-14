from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

PALETTE = [
    (255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 128, 128), (255, 128, 0), (0, 128, 255),
    (128, 0, 128), (0, 128, 0), (128, 0, 0), (0, 0, 128),
    (128, 128, 0), (0, 128, 128), (192, 192, 192), (64, 64, 64),
    (255, 64, 64), (64, 255, 64),
]


def label_to_color(label: str | int | None) -> Tuple[int, int, int]:
    if label is None:
        return PALETTE[0]
    if isinstance(label, int):
        return PALETTE[label % len(PALETTE)]
    h = int(hashlib.md5(str(label).encode()).hexdigest(), 16)
    return PALETTE[h % len(PALETTE)]


def create_canvas(height: int = 512, width: int = 512) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def draw_one_box(
    canvas: np.ndarray,
    box: Sequence[float],
    label: str | int | None = None,
    thickness: int = 3,
    fill: bool = False,
    clamp: bool = True,
) -> np.ndarray:
    h, w, _ = canvas.shape
    left, top, right, bottom = box
    if clamp:
        left = max(0, min(w - 1, int(round(left))))
        right = max(0, min(w - 1, int(round(right))))
        top = max(0, min(h - 1, int(round(top))))
        bottom = max(0, min(h - 1, int(round(bottom))))
    if right <= left or bottom <= top:
        return canvas
    color = label_to_color(label)
    if fill:
        cv2.rectangle(canvas, (left, top), (right, bottom), color, -1)
    cv2.rectangle(canvas, (left, top), (right, bottom), color, thickness)
    return canvas


def _to_xyxy(points: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    (x1, y1), (x2, y2) = points
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 1e-9 else 0.0


def jitter_labelme_box_points(
    points: Sequence[Sequence[float]],
    img_w: int,
    img_h: int,
    p: float = 0.50,
    shift_sigma: float = 0.010,
    scale_sigma: float = 0.030,
    aspect_sigma: float = 0.030,
    iou_min: float = 0.75,
    min_size: float = 8.0,
    max_tries: int = 30,
) -> List[List[float]]:
    """Small, IoU-constrained box jitter for robustness."""
    ox1, oy1, ox2, oy2 = _to_xyxy(points)
    ow, oh = ox2 - ox1, oy2 - oy1
    if ow < 1.0 or oh < 1.0 or random.random() > p:
        return [[ox1, oy1], [ox2, oy2]]

    ocx, ocy = (ox1 + ox2) * 0.5, (oy1 + oy2) * 0.5
    base = min(ow, oh)
    original = (ox1, oy1, ox2, oy2)

    for _ in range(max_tries):
        dx = random.gauss(0.0, shift_sigma * base)
        dy = random.gauss(0.0, shift_sigma * base)
        scale = math.exp(random.gauss(0.0, scale_sigma))
        aspect = math.exp(random.gauss(0.0, aspect_sigma))
        nw = max(ow * scale * aspect, min_size)
        nh = max(oh * scale / aspect, min_size)
        ncx, ncy = ocx + dx, ocy + dy
        nx1 = _clamp(ncx - nw * 0.5, 0.0, float(img_w))
        nx2 = _clamp(ncx + nw * 0.5, 0.0, float(img_w))
        ny1 = _clamp(ncy - nh * 0.5, 0.0, float(img_h))
        ny2 = _clamp(ncy + nh * 0.5, 0.0, float(img_h))
        candidate = (min(nx1, nx2), min(ny1, ny2), max(nx1, nx2), max(ny1, ny2))
        if candidate[2] - candidate[0] < min_size or candidate[3] - candidate[1] < min_size:
            continue
        if _iou_xyxy(original, candidate) >= iou_min:
            return [[candidate[0], candidate[1]], [candidate[2], candidate[3]]]
    return [[ox1, oy1], [ox2, oy2]]


def render_labelme_box_mask(
    json_path: str | Path,
    output_size: int = 512,
    jitter: bool = False,
    jitter_prob: float = 0.50,
    thickness: int = 5,
) -> Image.Image:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_w = int(data.get("imageWidth") or data.get("imageHeight") or output_size)
    img_h = int(data.get("imageHeight") or data.get("imageWidth") or output_size)
    canvas = create_canvas(output_size, output_size)

    for label_idx, shape in enumerate(data.get("shapes", [])):
        points = shape.get("points")
        if not points or len(points) < 2:
            continue
        if jitter:
            points = jitter_labelme_box_points(points, img_w=img_w, img_h=img_h, p=jitter_prob)
        x1, y1, x2, y2 = _to_xyxy(points)
        box = [
            x1 * output_size / max(1, img_w),
            y1 * output_size / max(1, img_h),
            x2 * output_size / max(1, img_w),
            y2 * output_size / max(1, img_h),
        ]
        draw_one_box(canvas, box, label=label_idx, thickness=thickness, fill=False)

    return Image.fromarray(canvas)



def stable_label_id(label: str | int | None, num_buckets: int = 4096) -> int:
    """Stable integer id for a LabelMe label; 0 is reserved for padding."""
    if label is None:
        return 0
    if isinstance(label, int):
        return 1 + (int(label) % max(1, num_buckets - 1))
    h = int(hashlib.sha1(str(label).encode("utf-8")).hexdigest(), 16)
    return 1 + (h % max(1, num_buckets - 1))


def points_to_xyxy(points: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def load_labelme_box_tensors(
    json_path: str | Path,
    max_boxes: int = 64,
    label_hash_size: int = 65536,
    jitter: bool = False,
    jitter_prob: float = 0.50,
    return_labels: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Read a LabelMe json into fixed-size coordinate tokens.

    Returns:
        features: [max_boxes, 10] float32, normalized geometry features.
        label_ids: [max_boxes] int64, stable hash ids; 0 means padding/fallback.
        valid_mask: [max_boxes] bool, True for real boxes.
        label_texts: optional [max_boxes] strings, aligned with the box rows.

    The geometry features are:
        x1, y1, x2, y2, cx, cy, w, h, area, log_aspect/4.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_w = int(data.get("imageWidth") or data.get("imageHeight") or 1)
    img_h = int(data.get("imageHeight") or data.get("imageWidth") or 1)
    img_w = max(1, img_w)
    img_h = max(1, img_h)

    rows: List[Tuple[float, float, float, float, str | int | None]] = []
    for shape in data.get("shapes", []):
        points = shape.get("points")
        if not points or len(points) < 2:
            continue
        if jitter:
            x1, y1, x2, y2 = points_to_xyxy(points)
            points = jitter_labelme_box_points([[x1, y1], [x2, y2]], img_w=img_w, img_h=img_h, p=jitter_prob)
        x1, y1, x2, y2 = points_to_xyxy(points)
        x1 = _clamp(x1, 0.0, float(img_w))
        x2 = _clamp(x2, 0.0, float(img_w))
        y1 = _clamp(y1, 0.0, float(img_h))
        y2 = _clamp(y2, 0.0, float(img_h))
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        if x2 - x1 <= 1e-6 or y2 - y1 <= 1e-6:
            continue
        rows.append((x1, y1, x2, y2, shape.get("label")))

    # Stable reading order: top-to-bottom, left-to-right, then label.
    rows.sort(key=lambda r: (((r[1] + r[3]) * 0.5), ((r[0] + r[2]) * 0.5), str(r[4])))
    rows = rows[: max(0, int(max_boxes))]

    features = np.zeros((max_boxes, 10), dtype=np.float32)
    label_ids = np.zeros((max_boxes,), dtype=np.int64)
    valid_mask = np.zeros((max_boxes,), dtype=np.bool_)
    label_texts = [""] * max_boxes

    for idx, (x1, y1, x2, y2, label) in enumerate(rows):
        nx1 = x1 / img_w
        ny1 = y1 / img_h
        nx2 = x2 / img_w
        ny2 = y2 / img_h
        w = max(1e-6, nx2 - nx1)
        h = max(1e-6, ny2 - ny1)
        cx = (nx1 + nx2) * 0.5
        cy = (ny1 + ny2) * 0.5
        area = w * h
        log_aspect = float(np.clip(np.log(w / h) / 4.0, -1.0, 1.0))
        features[idx] = np.asarray([nx1, ny1, nx2, ny2, cx, cy, w, h, area, log_aspect], dtype=np.float32)
        label_ids[idx] = stable_label_id(label, num_buckets=label_hash_size)
        label_texts[idx] = "" if label is None else str(label).strip()
        valid_mask[idx] = True

    if return_labels:
        return features, label_ids, valid_mask, label_texts
    return features, label_ids, valid_mask
