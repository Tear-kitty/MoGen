from __future__ import annotations
import argparse
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline

from mogen.constants import BOX_LABEL_HASH_SIZE, DEFAULT_BASE_MODEL_PATH, DEFAULT_RESULTS_DIR, MAX_BOXES
from mogen.inference_preprocess import encode_appearance, encode_box, encode_structure, load_dino
from mogen.pipeline import MoGenAdapterXL
from mogen.utils import dtype_from_string, safe_stem, str2bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MoGen SDXL inference", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mode", choices=["text", "control"], required=False, default='control')
    parser.add_argument("--checkpoint", type=str, required=False, default='./checkpoints/control-checkpoint-4800/mogen_adapter.bin', help="Path to mogen_adapter.bin")
    parser.add_argument("--base_model_path", type=str, default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--prompt", type=str, required=False, default='five yellow ducklings')
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_RESULTS_DIR)

    parser.add_argument("--structure_image", type=str, default='/home/lyf/MoGen_refactored/data/image/51_2.png')
    parser.add_argument("--box_json", type=str, default=None)
    parser.add_argument("--appearance_dir", type=str, default=None)

    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--scale", type=float, default=0.8, help="Attention injection scale for MoGen condition tokens")
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch_dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--attention_backend", choices=["sdpa", "flash"], default="sdpa", help="Projection attention backend; sdpa is usually enough for inference")
    parser.add_argument("--dino_model_name", type=str, default=None)
    parser.add_argument("--global_text_scale", type=float, default=1.0, help="Scale for enhance_text: text_embeddings + scale * extra")
    parser.add_argument("--use_global_text_in_control", type=str2bool, nargs="?", const=True, default=False, help="Whether control mode passes text embeddings through enhance_text")
    parser.add_argument("--disable_global_text_in_control", action="store_false", dest="use_global_text_in_control", help="Shortcut for --use_global_text_in_control false")
    parser.add_argument("--max_boxes", type=int, default=MAX_BOXES, help="Maximum LabelMe boxes encoded per sample")
    parser.add_argument("--box_label_hash_size", type=int, default=BOX_LABEL_HASH_SIZE, help="Hash buckets for LabelMe box labels")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = dtype_from_string(args.torch_dtype)
    if device.type == "cpu" and dtype != torch.float32:
        dtype = torch.float32

    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.base_model_path,
        torch_dtype=dtype,
        add_watermarker=False,
    )
    pipe.enable_vae_tiling()
    pipe.unet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()
    pipe.text_encoder_2.eval()

    model = MoGenAdapterXL(
        pipe,
        checkpoint_path=args.checkpoint,
        device=device,
        mode=args.mode,
        attention_backend=args.attention_backend,
        dtype=dtype,
        strict_load=True,
        use_global_text_in_control=args.use_global_text_in_control,
        global_text_scale=args.global_text_scale,
        max_boxes=args.max_boxes,
        box_label_hash_size=args.box_label_hash_size,
    )

    structure = box = appearance = None
    if args.mode == "control":
        if args.structure_image is None and args.box_json is None and args.appearance_dir is None:
            raise ValueError("Control mode requires at least one of --structure_image, --box_json, --appearance_dir.")
        dino_processor = dino_model = None
        if args.structure_image or args.appearance_dir:
            dino_name = args.dino_model_name or "facebook/dinov2-with-registers-giant"
            dino_processor, dino_model = load_dino(device, dtype, dino_name)
        if args.structure_image:
            structure = encode_structure(args.structure_image, dino_processor, dino_model, model.projector, device, dtype)
        if args.box_json:
            box = encode_box(
                args.box_json,
                model.projector,
                device,
                dtype,
                max_boxes=args.max_boxes,
                box_label_hash_size=args.box_label_hash_size,
                tokenizer_2=pipe.tokenizer_2,
                text_encoder_2=pipe.text_encoder_2,
            )
        if args.appearance_dir:
            appearance = encode_appearance(args.appearance_dir, dino_processor, dino_model, model.projector, device, dtype)

    images = model.generate(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        structure=structure,
        box=box,
        appearance=appearance,
        scale=args.scale,
        num_samples=args.num_samples,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        height=args.height,
        width=args.width,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_stem(args.prompt)
    for idx, image in enumerate(images):
        path = output_dir / f"{stem}_{args.mode}_{idx:02d}.png"
        image.save(path)
        print(path)


if __name__ == "__main__":
    main()
