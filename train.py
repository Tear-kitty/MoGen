from __future__ import annotations

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

import argparse
import itertools
import logging
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from torch.optim.lr_scheduler import LambdaLR
from diffusers.training_utils import compute_snr
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from mogen.adapter_setup import set_mogen_attention_processors
from mogen.checkpointing import (
    load_attention_state,
    load_matching_from_checkpoint,
    load_mogen_checkpoint,
    save_mogen_checkpoint,
)
from mogen.constants import (
    CONTROL_NUM_TOKENS,
    CONTROL_TARGET_BLOCKS,
    DEFAULT_BASE_ADAPTER_PATH,
    DEFAULT_BASE_MODEL_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TRAIN_DATA_DIR,
    DINO_MODEL_NAME,
    BOX_LABEL_HASH_SIZE,
    MAX_APPEARANCE_REFS,
    MAX_BOXES,
    TEXT_NUM_TOKENS,
    TEXT_TARGET_BLOCKS,
)
from mogen.data import MoGenDataset
from mogen.projection import MoGenProjection
from mogen.utils import dtype_from_string, latest_checkpoint_dir, str2bool

check_min_version("0.25.0")
logger = get_logger(__name__)


class MoGenTrainingModule(nn.Module):
    def __init__(self, unet: UNet2DConditionModel, projector: MoGenProjection, attention_modules: nn.ModuleList, mode: str):
        super().__init__()
        self.unet = unet
        self.projector = projector
        self.attention_modules = attention_modules
        self.mode = mode

    def train(self, mode: bool = True):
        super().train(mode)
        # The SDXL UNet is frozen. Keep it in eval mode while allowing the
        # projector and attention processors to receive gradients.
        self.unet.eval()
        self.projector.train(mode)
        self.attention_modules.train(mode)
        return self

    def forward(self, noisy_latents, timesteps, text_embeds, dino_v2, batch, unet_added_cond_kwargs):
        if self.mode == "text":
            text_embeds, condition_embeds = self.projector.project_text_branch(text_embeds)
        else:
            text_embeds, condition_embeds = self.projector(text_embeds, dino_v2=dino_v2, batch=batch)
        encoder_hidden_states = torch.cat([text_embeds, condition_embeds], dim=1)
        return self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states,
            added_cond_kwargs=unet_added_cond_kwargs,
        ).sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MoGen SDXL adapter", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mode", choices=["text", "control"], required=False, default='control', help="text: text-only stage; control: structure/box/appearance stage")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--base_adapter_path", type=str, default=DEFAULT_BASE_ADAPTER_PATH, help="Original SDXL adapter initialization checkpoint")
    parser.add_argument("--init_text_checkpoint", type=str, default=None, help="Optional text-stage checkpoint used to initialize the control stage")
    parser.add_argument("--train_data_dir", type=str, default=DEFAULT_TRAIN_DATA_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--dino_model_name", type=str, default=DINO_MODEL_NAME)
    parser.add_argument("--global_text_scale", type=float, default=1.0, help="Scale for enhance_text: text_embeddings + scale * extra")
    parser.add_argument("--use_global_text_in_control", type=str2bool, nargs="?", const=True, default=False, help="Whether control mode passes text embeddings through enhance_text")
    parser.add_argument("--disable_global_text_in_control", action="store_false", dest="use_global_text_in_control", help="Shortcut for --use_global_text_in_control false")
    parser.add_argument("--max_boxes", type=int, default=MAX_BOXES, help="Maximum LabelMe boxes encoded per sample")
    parser.add_argument("--box_label_hash_size", type=int, default=BOX_LABEL_HASH_SIZE, help="Hash buckets for LabelMe box labels")

    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--center_crop", action="store_true", default=True)
    parser.add_argument("--random_crop", dest="center_crop", action="store_false")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=14)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=7000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--scale_lr", action="store_true", default=False)
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--min_lr_ratio", type=float, default=0.0, help="Lower bound for cosine LR as a fraction of the initial LR. 0.1 means the LR decays at most 10x.")
    parser.add_argument("--dataloader_num_workers", type=int, default=32)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=0.0)

    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--attention_backend", choices=["flash", "sdpa"], default="flash", help="Projection attention backend; flash is recommended for training")
    parser.add_argument("--prediction_type", type=str, default="epsilon", choices=["epsilon", "v_prediction"])
    parser.add_argument("--snr_gamma", type=float, default=None)
    parser.add_argument("--importance_sampling", action="store_true", default=False)
    parser.add_argument("--scaled_cosine_alpha", type=float, default=0.5)

    parser.add_argument("--text_drop_prob", type=float, default=0.05, help="Text dropout probability for the current stage. The text-only script passes 0.10 explicitly; control stage defaults to 0.05.")
    parser.add_argument("--control_drop_prob", type=float, default=0.05)
    parser.add_argument("--both_drop_prob", type=float, default=0.05)
    parser.add_argument("--box_jitter_prob", type=float, default=0.0)
    parser.add_argument("--structure_degrade_prob", type=float, default=0.0)
    parser.add_argument("--appearance_degrade_prob", type=float, default=0.0)
    parser.add_argument("--appearance_count_error_prob", type=float, default=0.0)
    parser.add_argument("--prompt_selection", choices=["last", "random"], default="last")

    parser.add_argument("--save_steps", type=int, default=300)
    parser.add_argument("--resume_from_checkpoint", type=str, default='latest', help='Path or "latest"')
    parser.add_argument("--resume_mode", choices=["full", "model_only"], default="full", help="full restores model/optimizer/scheduler; model_only loads only MoGen weights and restarts step/epoch/LR schedule.")
    parser.add_argument("--resume_learning_rate", "--resume_lr", dest="resume_learning_rate", type=float, default=None, help="Optional current LR override for full resume. model_only uses --learning_rate instead.")
    parser.add_argument("--report_to", type=str, default="tensorboard", choices=["tensorboard", "wandb", "none"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def importance_sampling_probs(num_train_timesteps: int, alpha: float, device: torch.device) -> torch.Tensor:
    t = torch.arange(num_train_timesteps, dtype=torch.float32, device=device)
    probs = (1.0 / num_train_timesteps) * (1.0 - alpha * torch.cos(math.pi * t / num_train_timesteps))
    return probs / probs.sum()



def _validate_min_lr_ratio(min_lr_ratio: float) -> float:
    if min_lr_ratio < 0.0 or min_lr_ratio > 1.0:
        raise ValueError(f"--min_lr_ratio must be in [0, 1], got {min_lr_ratio}.")
    return float(min_lr_ratio)


def resolve_resume_path(resume_from_checkpoint: Optional[str], output_dir: str, mode: str) -> Optional[Path]:
    """Resolve a resume argument without changing the original path semantics."""
    if not resume_from_checkpoint:
        return None
    if resume_from_checkpoint == "latest":
        latest = latest_checkpoint_dir(output_dir, mode)
        if latest is None:
            raise FileNotFoundError(f"No latest {mode} checkpoint found in {output_dir}")
        return Path(latest)
    return Path(resume_from_checkpoint)


def adapter_file_from_resume_path(resume_path: Path) -> Path:
    """Accept either a checkpoint directory or a direct mogen_adapter.bin path."""
    if resume_path.is_dir():
        adapter_path = resume_path / "mogen_adapter.bin"
        if not adapter_path.exists():
            raise FileNotFoundError(
                f"{resume_path} is a directory, but {adapter_path.name} was not found. "
                "Use --resume_mode full for an Accelerate state directory, or pass a MoGen adapter checkpoint."
            )
        return adapter_path
    if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
    return resume_path


def build_lr_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer, accelerator: Accelerator):
    """Build the LR scheduler.

    Diffusers' cosine scheduler decays all the way to zero.  For adapter
    finetuning it is often useful to stop at a floor such as 0.1x the initial
    LR, so cosine/cosine_with_restarts are implemented here when a non-zero
    floor is requested.  Other schedulers preserve the original Diffusers
    behavior.
    """
    min_lr_ratio = _validate_min_lr_ratio(args.min_lr_ratio)
    num_warmup_steps = args.lr_warmup_steps * accelerator.num_processes
    num_training_steps = args.max_train_steps * accelerator.num_processes

    if args.lr_scheduler == "cosine" and min_lr_ratio > 0.0:
        def lr_lambda(current_step: int) -> float:
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        return LambdaLR(optimizer, lr_lambda, -1)

    if args.lr_scheduler == "cosine_with_restarts" and min_lr_ratio > 0.0:
        # Keep the same default single-cycle behavior as diffusers.get_scheduler
        # unless the caller later extends the CLI with an explicit num_cycles.
        num_cycles = 1

        def lr_lambda(current_step: int) -> float:
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            if progress >= 1.0:
                return min_lr_ratio
            cosine = 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0)))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        return LambdaLR(optimizer, lr_lambda, -1)

    return get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )


def override_full_resume_lr(
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    learning_rate: float,
    logger_: logging.Logger,
) -> None:
    """Set the current LR after loading a full Accelerate training state.

    For LambdaLR-style schedulers, base_lrs are adjusted so the next scheduler
    step continues the same decay curve from the requested current LR instead of
    snapping back to the old checkpoint LR.
    """
    if learning_rate <= 0:
        raise ValueError(f"--resume_learning_rate must be > 0, got {learning_rate}")

    base_optimizer = getattr(optimizer, "optimizer", optimizer)
    base_scheduler = getattr(lr_scheduler, "scheduler", lr_scheduler)

    param_groups = base_optimizer.param_groups
    for group in param_groups:
        group["lr"] = learning_rate
        group["initial_lr"] = learning_rate

    lambdas = getattr(base_scheduler, "lr_lambdas", None)
    if lambdas is not None and hasattr(base_scheduler, "base_lrs"):
        last_epoch = int(getattr(base_scheduler, "last_epoch", 0))
        new_base_lrs = []
        for lr_lambda in lambdas:
            factor = float(lr_lambda(last_epoch))
            if abs(factor) < 1e-12:
                factor = 1e-12
            new_base_lrs.append(float(learning_rate) / factor)
        base_scheduler.base_lrs = new_base_lrs
        base_scheduler._last_lr = [float(learning_rate) for _ in param_groups]
    elif hasattr(base_scheduler, "_last_lr"):
        base_scheduler._last_lr = [float(learning_rate) for _ in param_groups]
        logger_.warning(
            "Overrode optimizer LR, but the scheduler type does not expose lr_lambdas/base_lrs; "
            "the next scheduler.step() may overwrite it."
        )


def load_model_only_resume(
    resume_path: Path,
    projector: MoGenProjection,
    attention_modules: nn.ModuleList,
) -> dict:
    adapter_path = adapter_file_from_resume_path(resume_path)
    return load_mogen_checkpoint(adapter_path, projector, attention_modules, strict=False)


@torch.no_grad()
def add_box_label_embeddings_to_batch(batch: dict, text_encoder_2: CLIPTextModelWithProjection, dtype: torch.dtype) -> None:
    """Encode one frozen SDXL text_encoder_2 pooled embedding per valid box label.

    The dataset only stores token ids so DataLoader workers do not need to own a
    GPU text encoder. Repeated labels are deduplicated inside the batch; repeated
    instances still remain separate box tokens through their geometry/order.
    """
    if "box_label_input_ids_2" not in batch or "box_token_mask" not in batch:
        return

    encoder_device = next(text_encoder_2.parameters()).device
    label_input_ids = batch["box_label_input_ids_2"].to(device=encoder_device)
    box_token_mask = batch["box_token_mask"].to(device=encoder_device, dtype=torch.bool)
    if label_input_ids.ndim != 3:
        raise ValueError(f"Expected box_label_input_ids_2 [B, N, L], got {tuple(label_input_ids.shape)}")

    bsz, max_boxes, seq_len = label_input_ids.shape
    embed_dim = getattr(text_encoder_2.config, "projection_dim", None)
    if embed_dim is None:
        embed_dim = getattr(text_encoder_2.config, "hidden_size")
    embed_dim = int(embed_dim)
    label_embeds = torch.zeros(bsz, max_boxes, embed_dim, device=label_input_ids.device, dtype=dtype)

    flat_mask = box_token_mask.reshape(-1)
    if flat_mask.any():
        flat_ids = label_input_ids.reshape(bsz * max_boxes, seq_len)
        valid_ids = flat_ids[flat_mask]
        unique_ids, inverse = torch.unique(valid_ids, sorted=False, return_inverse=True, dim=0)
        unique_embeds = text_encoder_2(unique_ids, output_hidden_states=False)[0].to(dtype=dtype)
        label_embeds.reshape(bsz * max_boxes, embed_dim)[flat_mask] = unique_embeds[inverse]

    batch["box_label_embeds"] = label_embeds


def setup_logging(args: argparse.Namespace, accelerator: Accelerator) -> None:
    if accelerator.is_main_process:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    log_file = Path(args.output_dir) / args.logging_dir / f"train_{args.mode}.log"
    if accelerator.is_main_process:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logging.getLogger().addHandler(file_handler)

def main() -> None:
    args = parse_args()

    project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=str(Path(args.output_dir) / args.logging_dir))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None if args.report_to == "none" else args.report_to,
        project_config=project_config,
    )
    setup_logging(args, accelerator)

    if args.report_to == "wandb" and not is_wandb_available():
        raise ImportError("Install wandb or set --report_to tensorboard/none.")
    if args.seed is not None:
        set_seed(args.seed)
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    logger.info(f"Loading SDXL from {args.pretrained_model_name_or_path}")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    dino_v2 = None
    if args.mode == "control":
        dino_v2 = AutoModel.from_pretrained(args.dino_model_name)
        dino_v2.requires_grad_(False)
        dino_v2.to(accelerator.device, dtype=weight_dtype)
        dino_v2.eval()

    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    if args.enable_xformers_memory_efficient_attention:
        if not is_xformers_available():
            raise ValueError("xformers is not available. Install it or remove --enable_xformers_memory_efficient_attention.")
        unet.enable_xformers_memory_efficient_attention()

    target_blocks = TEXT_TARGET_BLOCKS if args.mode == "text" else CONTROL_TARGET_BLOCKS
    num_tokens = TEXT_NUM_TOKENS if args.mode == "text" else CONTROL_NUM_TOKENS
    attention_modules = set_mogen_attention_processors(
        unet,
        target_blocks=target_blocks,
        num_tokens=num_tokens,
        train_text=(args.mode == "text"),
        device=accelerator.device,
        dtype=torch.float32,
    )
    if args.base_adapter_path:
        incompatible = load_attention_state(args.base_adapter_path, attention_modules, strict=False)
        if accelerator.is_main_process:
            logger.info(f"Loaded base adapter weights: missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)}")

    projector = MoGenProjection(
        1536,
        512,
        32,
        train_text=(args.mode == "text"),
        max_appearance_refs=MAX_APPEARANCE_REFS,
        attention_backend=args.attention_backend,
        use_global_text_in_control=args.use_global_text_in_control,
        global_text_scale=args.global_text_scale,
        max_boxes=args.max_boxes,
        box_label_hash_size=args.box_label_hash_size,
    ).to(accelerator.device, dtype=torch.float32)

    if args.mode == "control" and args.init_text_checkpoint:
        results = load_matching_from_checkpoint(args.init_text_checkpoint, projector, attention_modules)
        if accelerator.is_main_process:
            logger.info(f"Initialized control stage from text checkpoint: {args.init_text_checkpoint}; loaded={list(results.keys())}")

    model = MoGenTrainingModule(unet, projector, attention_modules, args.mode)
    params_to_optimize = itertools.chain(model.projector.parameters(), model.attention_modules.parameters())

    if args.scale_lr:
        args.learning_rate = args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = MoGenDataset(
        data_root=args.train_data_dir,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        mode=args.mode,
        size=args.resolution,
        repeats=args.repeats,
        center_crop=args.center_crop,
        prompt_selection=args.prompt_selection,
        dino_model_name=args.dino_model_name,
        max_boxes=args.max_boxes,
        box_label_hash_size=args.box_label_hash_size,
        text_drop_prob=args.text_drop_prob,
        control_drop_prob=args.control_drop_prob,
        both_drop_prob=args.both_drop_prob,
        box_jitter_prob=args.box_jitter_prob,
        structure_degrade_prob=args.structure_degrade_prob,
        appearance_degrade_prob=args.appearance_degrade_prob,
        appearance_count_error_prob=args.appearance_count_error_prob,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=args.dataloader_num_workers > 0,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    overrode_max_train_steps = args.max_train_steps is None
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = build_lr_scheduler(args, optimizer, accelerator)
    if args.min_lr_ratio > 0 and args.lr_scheduler not in {"cosine", "cosine_with_restarts"} and accelerator.is_main_process:
        logger.warning("--min_lr_ratio currently affects only cosine/cosine_with_restarts; other schedulers keep diffusers' original behavior.")

    resume_path = resolve_resume_path(args.resume_from_checkpoint, args.output_dir, args.mode)
    if resume_path is not None and args.resume_mode == "model_only":
        results = load_model_only_resume(resume_path, projector, attention_modules)
        if accelerator.is_main_process:
            logger.info(
                f"Loaded only MoGen model weights from {resume_path}. "
                f"Optimizer/scheduler/step are reset; training restarts from global_step=0 with learning_rate={args.learning_rate:.3e}. "
                f"loaded={list(results.keys())}"
            )
        if args.resume_learning_rate is not None and accelerator.is_main_process:
            logger.warning("--resume_learning_rate is ignored when --resume_mode model_only; use --learning_rate instead.")

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, lr_scheduler)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    resume_step = 0
    first_epoch = 0
    resume_batch_skip = 0
    if resume_path is not None and args.resume_mode == "full":
        if not resume_path.is_dir():
            raise ValueError(
                f"--resume_mode full expects an Accelerate checkpoint directory, got {resume_path}. "
                "Use --resume_mode model_only when passing mogen_adapter.bin directly."
            )
        accelerator.load_state(str(resume_path))
        try:
            resume_step = int(resume_path.name.rsplit("-", 1)[-1])
        except ValueError:
            resume_step = 0
        first_epoch = resume_step // num_update_steps_per_epoch
        resume_batch_skip = (resume_step % num_update_steps_per_epoch) * args.gradient_accumulation_steps
        if args.resume_learning_rate is not None:
            override_full_resume_lr(optimizer, lr_scheduler, args.resume_learning_rate, logger)
            logger.info(
                f"Fully resumed from {resume_path} at global_step={resume_step}; "
                f"current LR overridden to {args.resume_learning_rate:.3e}."
            )
        else:
            logger.info(f"Fully resumed from {resume_path} at global_step={resume_step}")
        if resume_batch_skip > 0:
            logger.info(f"Skipping {resume_batch_skip} already-seen dataloader batches in epoch {first_epoch}.")
    elif resume_path is not None and args.resume_mode == "model_only":
        logger.info(f"Model-only resume active: global_step is reset to 0 after loading {resume_path}.")

    if accelerator.is_main_process and args.report_to != "none":
        accelerator.init_trackers("mogen", config=vars(args))

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Mode = {args.mode}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Per-device batch size = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    noise_scheduler.register_to_config(prediction_type=args.prediction_type)
    timestep_probs = None
    if args.importance_sampling:
        timestep_probs = importance_sampling_probs(noise_scheduler.config.num_train_timesteps, args.scaled_cosine_alpha, accelerator.device)
        logger.info("Using relation-focal timestep importance sampling.")

    progress_bar = tqdm(range(resume_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = resume_step
    running_loss = 0.0

    for _epoch in range(first_epoch, args.num_train_epochs):
        model.train()
        for batch_index, batch in enumerate(train_dataloader):
            if _epoch == first_epoch and batch_index < resume_batch_skip:
                continue
            with accelerator.accumulate(model):
                pixel_values = batch["pixel_values"].to(dtype=torch.float32)
                latents = vae.encode(pixel_values).latent_dist.sample().detach()
                latents = latents * vae.config.scaling_factor
                latents = latents.to(dtype=weight_dtype)

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                if timestep_probs is None:
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (bsz,),
                        device=latents.device,
                    )
                else:
                    timesteps = torch.multinomial(timestep_probs, bsz, replacement=True).to(device=latents.device)
                timesteps = timesteps.long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                with torch.no_grad():
                    encoder_output = text_encoder(batch["input_ids"], output_hidden_states=True)
                    text_embeds = encoder_output.hidden_states[-2]
                    encoder_output_2 = text_encoder_2(batch["input_ids_2"], output_hidden_states=True)
                    pooled_text_embeds = encoder_output_2[0]
                    text_embeds_2 = encoder_output_2.hidden_states[-2]
                    text_embeds = torch.cat([text_embeds, text_embeds_2], dim=-1).to(dtype=weight_dtype)
                    pooled_text_embeds = pooled_text_embeds.to(dtype=weight_dtype)
                    if args.mode == "control":
                        add_box_label_embeddings_to_batch(batch, text_encoder_2, weight_dtype)

                add_time_ids = torch.cat(
                    [batch["original_size"], batch["crop_coords_top_left"], batch["target_size"]], dim=1
                ).to(dtype=weight_dtype)
                unet_added_cond_kwargs = {"text_embeds": pooled_text_embeds, "time_ids": add_time_ids}

                # with accelerator.autocast():
                noise_pred = model(noisy_latents, timesteps, text_embeds, dino_v2, batch, unet_added_cond_kwargs)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.snr_gamma is None:
                    loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
                else:
                    snr = compute_snr(noise_scheduler, timesteps)
                    if noise_scheduler.config.prediction_type == "v_prediction":
                        snr = snr + 1
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    per_sample_loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
                    per_sample_loss = per_sample_loss.mean(dim=list(range(1, len(per_sample_loss.shape))))
                    loss = (per_sample_loss * mse_loss_weights).mean()

                accelerator.backward(loss)
                # for idx, (name, param) in enumerate(model.named_parameters()):
                #     if param.requires_grad and param.grad is None:
                #         print("unused:", idx, name, tuple(param.shape))

                if accelerator.sync_gradients and args.max_grad_norm is not None and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                running_loss += loss.detach().float().item()

                if accelerator.is_local_main_process:
                    progress_bar.set_postfix({
                        "loss": f"{loss.detach().float().item():.4f}",
                        "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                    })


                if global_step % args.save_steps == 0:
                    save_dir = Path(args.output_dir) / f"{args.mode}-checkpoint-{global_step}"
                    if accelerator.is_main_process:
                        save_dir.mkdir(parents=True, exist_ok=True)
                    accelerator.wait_for_everyone()
                    accelerator.save_state(str(save_dir))
                    if accelerator.is_main_process:
                        metadata = {
                            "mode": args.mode,
                            "global_step": global_step,
                            "num_tokens": num_tokens,
                            "target_blocks": target_blocks,
                            "base_model": args.pretrained_model_name_or_path,
                            "use_global_text_in_control": args.use_global_text_in_control,
                            "global_text_scale": args.global_text_scale,
                            "max_boxes": args.max_boxes,
                            "box_label_hash_size": args.box_label_hash_size,
                            "learning_rate": args.learning_rate,
                            "lr_scheduler": args.lr_scheduler,
                            "min_lr_ratio": args.min_lr_ratio,
                        }
                        save_mogen_checkpoint(save_dir / "mogen_adapter.bin", model, accelerator, metadata)
                        logger.info(
                            f"Saved {args.mode} checkpoint to {save_dir}; "
                            f"mean_loss={running_loss / max(1, args.save_steps):.6f}; "
                            f"lr={lr_scheduler.get_last_lr()[0]:.3e}"
                        )
                    running_loss = 0.0

                if args.report_to != "none":
                    accelerator.log({"train_loss": loss.detach().float().item(), "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = Path(args.output_dir) / f"{args.mode}-final"
        final_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "mode": args.mode,
            "global_step": global_step,
            "num_tokens": num_tokens,
            "target_blocks": target_blocks,
            "base_model": args.pretrained_model_name_or_path,
            "use_global_text_in_control": args.use_global_text_in_control,
            "global_text_scale": args.global_text_scale,
            "max_boxes": args.max_boxes,
            "box_label_hash_size": args.box_label_hash_size,
            "learning_rate": args.learning_rate,
            "lr_scheduler": args.lr_scheduler,
            "min_lr_ratio": args.min_lr_ratio,
        }
        save_mogen_checkpoint(final_dir / "mogen_adapter.bin", model, accelerator, metadata)
        logger.info(f"Saved final checkpoint to {final_dir}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
