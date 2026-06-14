from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoImageProcessor, CLIPTokenizer

from .augmentations import degrade_image_pil, maybe_corrupt_count
from .box_utils import load_labelme_box_tensors
from .constants import BOX_LABEL_HASH_SIZE, DINO_MODEL_NAME, MAX_APPEARANCE_REFS, MAX_BOXES
from .utils import extract_count_from_name, is_image_file

if PIL.__version__ >= "9.1.0":
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.BILINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }


def load_appearance_image(path: str | os.PathLike[str]) -> Image.Image:
    """Load an object reference image and place it on a white square canvas."""
    img = Image.open(path).convert("RGB")
    arr = torch.from_numpy(np.array(img))
    mask = (arr[:, :, 0] == 0) & (arr[:, :, 1] == 0) & (arr[:, :, 2] == 0)
    if mask.any():
        arr[mask] = torch.tensor([255, 255, 255], dtype=arr.dtype)
        img = Image.fromarray(arr.numpy(), "RGB")

    # w, h = img.size
    # side = max(1, int(max(w, h) / 0.875))
    # square = Image.new("RGB", (side, side), (255, 255, 255))
    # square.paste(img, ((side - w) // 2, (side - h) // 2))
    return img


def build_appearance_instance_metadata(
    counts: List[int],
    max_refs: int,
    *,
    shuffle_extras: bool = False,
) -> tuple[List[int], List[int], List[int]]:
    """Build class-aware instance metadata for appearance references.

    Each object reference image represents one appearance class and ``counts[i]``
    tells how many instances of this class should appear in the target image.
    The model still has a fixed budget of ``max_refs`` instance slots, so this
    helper returns:

    - per-class counts after clipping/truncation;
    - one class id per kept instance slot;
    - one local instance id per kept instance slot.

    When the requested total count is larger than the fixed slot budget, we keep
    at least one instance for as many classes as possible before assigning the
    remaining slots to extra repeated instances. This avoids the old behavior
    where a large count from an early class could silently push later classes out
    or change their positional ids.
    """
    if max_refs <= 0:
        return [], [], []

    clean_counts = [max(1, int(count)) for count in counts]
    if not clean_counts:
        return [], [], []

    num_classes = min(len(clean_counts), max_refs)
    clean_counts = clean_counts[:num_classes]
    total_requested = sum(clean_counts)

    if total_requested <= max_refs:
        kept_counts = clean_counts
    else:
        kept_counts = [1 for _ in range(num_classes)]
        remaining = max_refs - num_classes
        extras: List[int] = []
        if shuffle_extras:
            for class_idx, count in enumerate(clean_counts):
                extras.extend([class_idx] * max(0, count - 1))
            random.shuffle(extras)
        else:
            max_count = max(clean_counts)
            for instance_idx in range(1, max_count):
                for class_idx, count in enumerate(clean_counts):
                    if instance_idx < count:
                        extras.append(class_idx)
        for class_idx in extras[:remaining]:
            kept_counts[class_idx] += 1

    instance_class_ids: List[int] = []
    instance_ids: List[int] = []
    for class_idx, count in enumerate(kept_counts):
        for instance_idx in range(count):
            instance_class_ids.append(class_idx)
            instance_ids.append(instance_idx)

    return kept_counts, instance_class_ids, instance_ids


class MoGenDataset(Dataset):
    """Training dataset for the two MoGen stages."""

    def __init__(
        self,
        data_root: str | os.PathLike[str],
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        mode: str,
        size: int = 1024,
        repeats: int = 1,
        interpolation: str = "bicubic",
        center_crop: bool = True,
        prompt_selection: str = "last",
        dino_model_name: str = DINO_MODEL_NAME,
        max_appearance_refs: int = MAX_APPEARANCE_REFS,
        max_boxes: int = MAX_BOXES,
        box_label_hash_size: int = BOX_LABEL_HASH_SIZE,
        text_drop_prob: float = 0.10,
        control_drop_prob: float = 0.05,
        both_drop_prob: float = 0.05,
        box_jitter_prob: float = 0.50,
        structure_degrade_prob: float = 0.35,
        appearance_degrade_prob: float = 0.45,
        appearance_count_error_prob: float = 0.20,
    ):
        if mode not in {"text", "control"}:
            raise ValueError("mode must be 'text' or 'control'")

        self.data_root = Path(data_root)
        self.mode = mode
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.size = size
        self.center_crop = center_crop
        self.prompt_selection = prompt_selection
        self.max_appearance_refs = max_appearance_refs
        self.max_boxes = max_boxes
        self.box_label_hash_size = box_label_hash_size
        self.text_drop_prob = text_drop_prob
        self.control_drop_prob = control_drop_prob
        self.both_drop_prob = both_drop_prob
        self.box_jitter_prob = box_jitter_prob
        self.structure_degrade_prob = structure_degrade_prob
        self.appearance_degrade_prob = appearance_degrade_prob
        self.appearance_count_error_prob = appearance_count_error_prob

        self.image_root = self.data_root / "image"
        self.text_root = self.data_root / "text"
        self.box_root = self.data_root / "box"
        if not self.image_root.exists():
            raise FileNotFoundError(f"Missing image directory: {self.image_root}")

        self.image_paths = sorted(path for path in self.image_root.iterdir() if path.is_file() and is_image_file(path))
        if not self.image_paths:
            raise FileNotFoundError(f"No images found under {self.image_root}")
        self.structure_candidates = self._build_structure_candidates(self.image_paths)

        self._length = len(self.image_paths) * max(1, int(repeats))
        self.interpolation = PIL_INTERPOLATION[interpolation]

        self.image_transform = transforms.Compose(
            [
                transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.box_transform = transforms.Compose(
            [
                transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.dino_processor = AutoImageProcessor.from_pretrained(dino_model_name) if mode == "control" else None
        if self.dino_processor is not None:
            empty = Image.new("RGB", (224, 224), (255, 255, 255))
            self.empty_dino_tensor = self.dino_processor(images=empty, return_tensors="pt").pixel_values[0] * 0.0
        else:
            self.empty_dino_tensor = torch.zeros(3, 224, 224)

    def __len__(self) -> int:
        return self._length

    def _load_prompt(self, image_path: Path) -> str:
        json_path = self.text_root / f"{image_path.stem}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Missing text json: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            prompts = data
        elif isinstance(data, dict):
            prompts = data.get(image_path.name) or data.get(image_path.stem) or list(data.values())[0]
        else:
            raise ValueError(f"Unsupported prompt json format: {json_path}")

        if isinstance(prompts, str):
            return prompts
        if not prompts:
            return ""
        if self.prompt_selection == "random":
            return random.choice(prompts)
        return prompts[-1]

    @staticmethod
    def _structure_group_key(path: Path) -> str:
        """Return the shared prefix for structure candidates in data/image.

        Filenames such as ``10_1.png``, ``10_2.png`` and ``10_3.png`` share
        the key ``10``.  If a file does not end with ``_<number>``, it falls
        back to its own stem so the image itself remains a valid candidate.
        """
        prefix, sep, suffix = path.stem.rpartition("_")
        if sep and prefix and suffix.isdigit():
            return prefix
        return path.stem

    @classmethod
    def _build_structure_candidates(cls, image_paths: List[Path]) -> Dict[Path, List[Path]]:
        groups: Dict[str, List[Path]] = {}
        for path in image_paths:
            groups.setdefault(cls._structure_group_key(path), []).append(path)
        for candidates in groups.values():
            candidates.sort(key=lambda p: (p.stem, p.suffix.lower()))
        return {path: groups[cls._structure_group_key(path)] for path in image_paths}

    def _load_structure(self, image_path: Path) -> Dict[str, Any]:
        candidates = self.structure_candidates.get(image_path) or [image_path]
        candidates = [path for path in candidates if path.exists() and path.is_file() and is_image_file(path)]
        if candidates:
            structure_path = random.choice(candidates)
            image = Image.open(structure_path).convert("RGB")
            image = degrade_image_pil(image, p=self.structure_degrade_prob)
            # image.save('/home/lyf/MoGen_refactored/results/valid.png')
            tensor = self.dino_processor(images=image, return_tensors="pt").pixel_values[0]
            return {"structure": tensor, "has_structure": 1}
        return {"structure": self.empty_dino_tensor.clone(), "has_structure": 0}

    def _empty_box(self) -> Dict[str, Any]:
        return {
            "box_features": torch.zeros(self.max_boxes, 10, dtype=torch.float32),
            "box_label_ids": torch.zeros(self.max_boxes, dtype=torch.long),
            "box_label_input_ids_2": torch.zeros(
                self.max_boxes, self.tokenizer_2.model_max_length, dtype=torch.long
            ),
            "box_token_mask": torch.zeros(self.max_boxes, dtype=torch.bool),
            "has_box_mask": 0,
        }

    def _tokenize_box_labels(self, label_texts: List[str]) -> torch.Tensor:
        # One label phrase per box. Repeated categories intentionally keep the
        # same semantic label embedding; geometry/order tokens distinguish each
        # instance. Invalid padded rows are masked out before text encoding.
        return self.tokenizer_2(
            label_texts,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer_2.model_max_length,
            return_tensors="pt",
        ).input_ids.long()

    def _load_box(self, image_path: Path) -> Dict[str, Any]:
        box_path = self.box_root / f"{image_path.stem}.json"
        if box_path.exists():
            features, label_ids, valid_mask, label_texts = load_labelme_box_tensors(
                box_path,
                max_boxes=self.max_boxes,
                label_hash_size=self.box_label_hash_size,
                jitter=True,
                jitter_prob=self.box_jitter_prob,
                return_labels=True,
            )
            return {
                "box_features": torch.from_numpy(features).float(),
                "box_label_ids": torch.from_numpy(label_ids).long(),
                "box_label_input_ids_2": self._tokenize_box_labels(label_texts),
                "box_token_mask": torch.from_numpy(valid_mask).bool(),
                "has_box_mask": int(bool(valid_mask.any())),
            }
        return self._empty_box()

    def _load_appearance(self, image_path: Path) -> Dict[str, Any]:
        object_root = Path(str(image_path).replace(f"{os.sep}image{os.sep}", f"{os.sep}object{os.sep}")).with_suffix("")
        empty = {
            "appearance": torch.zeros(self.max_appearance_refs, *self.empty_dino_tensor.shape),
            "appearance_num": 0,
            "appearance_class_num": 0,
            "appearance_class_counts": torch.zeros(self.max_appearance_refs, dtype=torch.long),
            "appearance_instance_class_ids": torch.zeros(self.max_appearance_refs, dtype=torch.long),
            "appearance_instance_ids": torch.zeros(self.max_appearance_refs, dtype=torch.long),
            "has_appearance": 0,
        }
        if not object_root.exists():
            return empty

        object_paths = sorted(path for path in object_root.iterdir() if path.is_file() and is_image_file(path))
        if not object_paths:
            return empty
        if len(object_paths) > self.max_appearance_refs:
            object_paths = sorted(random.sample(object_paths, self.max_appearance_refs))

        class_tensors: List[torch.Tensor] = []
        class_counts: List[int] = []
        for path in object_paths:
            count = extract_count_from_name(path, default=1)
            count = maybe_corrupt_count(count, error_prob=self.appearance_count_error_prob, max_count=self.max_appearance_refs)
            image = load_appearance_image(path)
            image = degrade_image_pil(image, p=self.appearance_degrade_prob)
            # image.save('./results/valid.png')
            tensor = self.dino_processor(images=image, return_tensors="pt").pixel_values[0]
            class_tensors.append(tensor)
            class_counts.append(count)

        if not class_tensors:
            return empty

        kept_counts, instance_class_ids, instance_ids = build_appearance_instance_metadata(
            class_counts,
            self.max_appearance_refs,
            shuffle_extras=True,
        )
        num_classes = len(kept_counts)
        num_refs = len(instance_class_ids)

        padded = torch.zeros(self.max_appearance_refs, *self.empty_dino_tensor.shape, dtype=self.empty_dino_tensor.dtype)
        stacked = torch.stack(class_tensors[:num_classes], dim=0)
        padded[:num_classes] = stacked

        padded_class_counts = torch.zeros(self.max_appearance_refs, dtype=torch.long)
        padded_instance_class_ids = torch.zeros(self.max_appearance_refs, dtype=torch.long)
        padded_instance_ids = torch.zeros(self.max_appearance_refs, dtype=torch.long)
        if num_classes > 0:
            padded_class_counts[:num_classes] = torch.tensor(kept_counts, dtype=torch.long)
        if num_refs > 0:
            padded_instance_class_ids[:num_refs] = torch.tensor(instance_class_ids, dtype=torch.long)
            padded_instance_ids[:num_refs] = torch.tensor(instance_ids, dtype=torch.long)

        return {
            "appearance": padded,
            "appearance_num": num_refs,
            "appearance_class_num": num_classes,
            "appearance_class_counts": padded_class_counts,
            "appearance_instance_class_ids": padded_instance_class_ids,
            "appearance_instance_ids": padded_instance_ids,
            "has_appearance": int(num_refs > 0),
        }

    def _cfg_dropout(self, prompt: str) -> tuple[str, int]:
        if self.mode == "text":
            return ("" if random.random() < self.text_drop_prob else prompt), 0

        r = random.random()
        if r < self.both_drop_prob:
            return "", 1
        r -= self.both_drop_prob
        if r < self.text_drop_prob:
            return "", 0
        r -= self.text_drop_prob
        if r < self.control_drop_prob:
            return prompt, 1
        return prompt, 0

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image_path = self.image_paths[index % len(self.image_paths)]
        raw_image = Image.open(image_path).convert("RGB")
        original_width, original_height = raw_image.size
        original_size = torch.tensor([original_height, original_width])

        image_tensor = self.image_transform(raw_image)
        delta_h = image_tensor.shape[1] - self.size
        delta_w = image_tensor.shape[2] - self.size
        if delta_h < 0 or delta_w < 0:
            raise ValueError(f"Image is smaller than target size after resize: {image_path}")
        if self.center_crop:
            top, left = delta_h // 2, delta_w // 2
        else:
            top = random.randint(0, delta_h) if delta_h > 0 else 0
            left = random.randint(0, delta_w) if delta_w > 0 else 0
        pixel_values = transforms.functional.crop(image_tensor, top=top, left=left, height=self.size, width=self.size)

        prompt = self._load_prompt(image_path)
        # prompt = prompt.rsplit(',', 1)[-1].strip()
        prompt, drop_control = self._cfg_dropout(prompt)

        example: Dict[str, Any] = {
            "pixel_values": pixel_values,
            "original_size": original_size,
            "crop_coords_top_left": torch.tensor([top, left]),
            "target_size": torch.tensor([self.size, self.size]),
            "image_path": str(image_path),
            "drop_control_embeds": drop_control,
        }

        if self.mode == "control":
            example.update(self._load_structure(image_path))
            example.update(self._load_box(image_path))
            example.update(self._load_appearance(image_path))

        example["input_ids"] = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        example["input_ids_2"] = self.tokenizer_2(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer_2.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        return example
