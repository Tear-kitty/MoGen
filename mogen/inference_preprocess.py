from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel

from .box_utils import load_labelme_box_tensors
from .constants import BOX_LABEL_HASH_SIZE, DINO_MODEL_NAME, MAX_APPEARANCE_REFS, MAX_BOXES
from .data import build_appearance_instance_metadata, load_appearance_image
from .utils import extract_count_from_name

BOX_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_dino(device: torch.device | str, dtype: torch.dtype, model_name: str = DINO_MODEL_NAME):
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device=device, dtype=dtype)
    model.eval()
    return processor, model


@torch.inference_mode()
def encode_structure(path: str | os.PathLike[str], processor, dino_model, projector, device, dtype) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    tensor = processor(images=image, return_tensors="pt").pixel_values.to(device=device, dtype=dtype)
    structure_raw = dino_model(tensor).last_hidden_state.to(dtype=dtype)
    structure_query = projector.structure_query.repeat(structure_raw.shape[0], 1, 1).to(device=device, dtype=dtype)
    embeds = projector.structure_cross_attention(structure_query, structure_raw)
    return projector.structure_out(embeds)


@torch.inference_mode()
def encode_box(
    path: str | os.PathLike[str],
    projector,
    device,
    dtype,
    max_boxes: int = MAX_BOXES,
    box_label_hash_size: int = BOX_LABEL_HASH_SIZE,
    tokenizer_2=None,
    text_encoder_2=None,
) -> torch.Tensor:
    features, label_ids, valid_mask, label_texts = load_labelme_box_tensors(
        path,
        max_boxes=max_boxes,
        label_hash_size=box_label_hash_size,
        jitter=False,
        return_labels=True,
    )
    box_features = torch.from_numpy(features).unsqueeze(0).to(device=device, dtype=torch.float32)
    box_label_ids = torch.from_numpy(label_ids).unsqueeze(0).to(device=device, dtype=torch.long)
    box_token_mask = torch.from_numpy(valid_mask).unsqueeze(0).to(device=device, dtype=torch.bool)

    box_label_embeds = None
    if tokenizer_2 is not None and text_encoder_2 is not None and bool(valid_mask.any()):
        label_input_ids = tokenizer_2(
            label_texts,
            padding="max_length",
            truncation=True,
            max_length=tokenizer_2.model_max_length,
            return_tensors="pt",
        ).input_ids.to(device=device)
        embed_dim = getattr(text_encoder_2.config, "projection_dim", None)
        if embed_dim is None:
            embed_dim = getattr(text_encoder_2.config, "hidden_size")
        embed_dim = int(embed_dim)
        box_label_embeds = torch.zeros(1, max_boxes, embed_dim, device=device, dtype=dtype)
        valid = torch.from_numpy(valid_mask).to(device=device, dtype=torch.bool)
        valid_ids = label_input_ids[valid]
        unique_ids, inverse = torch.unique(valid_ids, sorted=False, return_inverse=True, dim=0)
        unique_embeds = text_encoder_2(unique_ids, output_hidden_states=False)[0].to(dtype=dtype)
        box_label_embeds[0, valid] = unique_embeds[inverse]

    embeds = projector.box_encoder(box_features, box_label_ids, box_token_mask, box_label_embeds=box_label_embeds)
    return embeds.to(device=device, dtype=dtype)


@torch.inference_mode()
def encode_appearance(
    appearance_dir: str | os.PathLike[str],
    processor,
    dino_model,
    projector,
    device,
    dtype,
    max_refs: int = MAX_APPEARANCE_REFS,
) -> torch.Tensor:
    root = Path(appearance_dir)
    if not root.exists():
        raise FileNotFoundError(f"appearance_dir does not exist: {root}")

    class_refs = []
    class_counts = []
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            continue
        if len(class_refs) >= max_refs:
            break
        count = extract_count_from_name(path, default=1)
        image = load_appearance_image(path)
        tensor = processor(images=image, return_tensors="pt").pixel_values[0]
        class_refs.append(tensor)
        class_counts.append(count)

    if not class_refs:
        raise FileNotFoundError(f"No appearance reference images found in {root}")

    kept_counts, instance_class_ids, instance_ids = build_appearance_instance_metadata(
        class_counts,
        max_refs,
        shuffle_extras=False,
    )
    num_classes = len(kept_counts)
    num_instances = len(instance_class_ids)

    stacked = torch.stack(class_refs[:num_classes], dim=0).to(device=device, dtype=dtype)
    class_embeds = dino_model(stacked).last_hidden_state.to(dtype=dtype)
    seq_len = class_embeds.shape[1]
    feature_dim = class_embeds.shape[2]

    class_index = torch.tensor(instance_class_ids, device=device, dtype=torch.long)
    instance_index = torch.tensor(instance_ids, device=device, dtype=torch.long).clamp(min=0, max=max_refs - 1)
    embeds = class_embeds[class_index].unsqueeze(0)
    pos = projector.object_pos(instance_index).to(dtype=dtype).unsqueeze(0).unsqueeze(2).expand(1, num_instances, seq_len, -1)
    embeds = torch.cat([embeds, pos], dim=-1)

    token_mask = torch.ones(1, num_instances * seq_len, device=device, dtype=torch.bool)
    embeds = embeds.reshape(1, num_instances * seq_len, embeds.shape[-1])
    embeds = projector.object_out(embeds.to(dtype=dtype))
    return embeds * token_mask[:, :, None].to(embeds.dtype)
