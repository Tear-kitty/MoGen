import argparse
import json
import logging
import math
import random
from pathlib import Path
from typing import Optional
import itertools
import random
import matplotlib.pyplot as plt
import cv2 as cv
from safetensors import safe_open

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1' 

import diffusers
import numpy as np
import PIL
import torch
import torch.nn.functional as F
import torchvision.transforms.functional
import torch.utils.checkpoint
import transformers
import cv2
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (AutoencoderKL, DDPMScheduler, DiffusionPipeline,
                       DPMSolverMultistepScheduler, StableDiffusionPipeline,
                       UNet2DConditionModel, StableDiffusionXLPipeline)
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, create_repo, whoami
# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPFeatureExtractor, CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPImageProcessor
#from templates.embedding import Timesteps

from ip_adapter_XL.ip_adapter import WindowAwareLinearProjection
from ip_adapter_XL.utils import is_torch2_available
if is_torch2_available():
    from ip_adapter_XL.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter_XL.attention_processor import IPAttnProcessor, AttnProcessor
from ip_adapter_XL import revert_model
from ip_adapter_XL import IPAdapterXL

#import alpha_clip
from torchvision import transforms
from safetensors.torch import save_file, load_file
import re

from transformers import AutoImageProcessor, AutoModel

if version.parse(version.parse(
        PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
# ------------------------------------------------------------------------------

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.13.0.dev0")

logger = get_logger(__name__)

IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP', '.tif')
JSON_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP', '.tif', '.json')

def is_image_file(filename):
    # return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
    return filename.endswith(IMG_EXTENSIONS)

def is_json_file(filename):
    # return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
    return filename.endswith(JSON_EXTENSIONS)


def save_progress(text_encoder, placeholder_token_id, accelerator, args,
                  save_path):
    logger.info("Saving embeddings")
    learned_embeds = accelerator.unwrap_model(
        text_encoder).get_input_embeddings().weight[placeholder_token_id]
    learned_embeds_dict = {
        args.placeholder_token: learned_embeds.detach().cpu()
    }
    torch.save(learned_embeds_dict, save_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple example of a training script.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--only_save_embeds",
        action="store_true",
        default=False,
        help="Save only the embeddings for the new concept.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=False,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=
        "Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--train_data_dir", 
        type=str,
        default=None,
        required=True,
        help=
        "The folder that contains the exemplar images (and coarse descriptions) of the specific relation."
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=100,
        help="How many times to repeat the training data.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help=
        "The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=
        ("The resolution for input images, all the images in the train/validation dataset will be resized to this"
         " resolution"),
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Whether to center crop images before resizing to resolution.")
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        required=False,
        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=10000,
        help=
        "Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help=
        "Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-05,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help=
        "Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=
        ('The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
         ' "constant", "constant_with_warmup"]'),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=16, 
        help=
        ("Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
         ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay to use.")
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer")
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help=
        "The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=
        ("[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
         " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=
        ("Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
         " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
         ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=
        ('The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
         ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
         ),
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=
        ("Run validation every X epochs. Validation consists of running the prompt"
         " `args.validation_prompt` multiple times: `args.num_validation_images`"
         " and logging the images."),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=
        ("Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
         " training using `--resume_from_checkpoint`."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=
        ("Whether training should be resumed from a previous checkpoint. Use a path saved by"
         ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
         ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.")

    parser.add_argument(
        "--importance_sampling",
        action='store_true',
        default=False,
        help="Relation-Focal Importance Sampling",
    )
    parser.add_argument(
        "--denoise_loss_weight",
        type=float,
        default=1.0,
        help="Weight of L_denoise",
    )
    parser.add_argument(
        "--steer_loss_weight",
        type=float,
        default=0.0,
        help="Weight of L_steer (for Relation-Steering Contrastive Learning)",
    )
    parser.add_argument(
        "--num_positives",
        type=int,
        default=4,
        help="Number of positive words used for L_steer",
    )
    parser.add_argument(
        "--num_negtives",
        type=int,
        default=4,
        help="Number of negtive words",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default="0.07",
        help="Temperature parameter for L_steer",
    )
    parser.add_argument(
        "--scaled_cosine_alpha",
        type=float,
        default=0.5,
        help="The skewness (alpha) of the Importance Sampling Function",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default='epsilon',
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
             "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_adapter",
        type=str,
        default=True,
        help="whether to use adapter",
    )
    parser.add_argument(
        "--positive_number",
        type=str,
        required=False,
        default='four',
        help="positive samples of placeholder",
    )
    parser.add_argument(
        "--first_step",
        type=int,
        default=0,
        required=None,
        help="checkpoint step",
    )
    parser.add_argument(
        "--train_text",
        type=bool,
        default=False,
        required=True,
    )
    parser.add_argument(
        "--ckpt_path",
        type=bool,
        default=False,
        required=True,
    )


    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.train_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    return args

PALETTE = [
    (255,255,255), (255,0,0), (0,255,0), (0,0,255),
    (255,255,0), (255,0,255), (0,255,255),
    (128,128,128), (255,128,0), (0,128,255),
    (128,0,128), (0,128,0), (128,0,0), (0,0,128),
    (128,128,0), (0,128,128), (192,192,192), (64,64,64),
    (255,64,64), (64,255,64)
]

import hashlib

def _label_to_color(label):
    if isinstance(label, int):
        return PALETTE[label % len(PALETTE)]
    h = int(hashlib.md5(str(label).encode()).hexdigest(), 16)
    return PALETTE[h % len(PALETTE)]

def create_canvas(H=448, W=448):
    return np.zeros((H, W, 3), dtype=np.uint8)

def draw_one_box(canvas, box, label=None, thickness=3, fill=False, clamp=True):
    H, W, _ = canvas.shape
    l,t,r,b = box
    if clamp:
        l = max(0, min(W-1, int(l))); r = max(0, min(W-1, int(r)))
        t = max(0, min(H-1, int(t))); b = max(0, min(H-1, int(b)))
    if r <= l or b <= t:
        return canvas
    color = _label_to_color(label) if label is not None else (255,255,255)
    if fill:
        cv2.rectangle(canvas, (l,t), (r,b), color, -1)
    cv2.rectangle(canvas, (l,t), (r,b), color, thickness)
    return canvas

def to_pil(canvas):
    return Image.fromarray(canvas)

def load_image(path):
    img = Image.open(path).convert("RGBA")           
    bg = Image.new("RGB", img.size, (255, 255, 255))  
    bg.paste(img, mask=img.split()[3])               
    return bg

class ReVersionDataset(Dataset):

    def __init__(
        self,
        data_root,
        tokenizer,
        tokenizer_2,
        size=1024,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.0,  # do not flip horizontally, otherwise might affect the relation
        set="train",
        center_crop=True,
    ):
        self.data_root = data_root

        self.text_path = os.path.join(data_root, 'position')

        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.size = size
        self.center_crop = center_crop
        self.flip_p = flip_p

        # record image paths
        self.image_paths = []
        self.data_root = data_root
        self.image_root = os.path.join(data_root, 'image')
        for file_path in os.listdir(self.image_root):
            # if file_path != 'text.json':
            # for file in os.listdir(os.path.join(self.image_root, file_path)):
            file = os.path.join(self.image_root, file_path)
            if is_image_file(file):
                self.image_paths.append(
                    os.path.join(self.image_root, file_path, file))

        self.image_json_path = os.path.join(data_root, 'text')  
        self.image_box_path = os.path.join(data_root, 'box') 

        self.num_images = int(len(self.image_paths))

        self._length = self.num_images

        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        self.transform_mask = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        ])

        self.clip_image_processor = CLIPImageProcessor()

        if set == "train":
            self._length = self.num_images #* repeats

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

        self.dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-with-registers-giant')

    def __len__(self):
        return self._length
        
    def __getitem__(self, i):
        example = {}
        # exemplar images
        image_path = self.image_paths[i % self.num_images]
        image = Image.open(image_path)
        
        image_name = image_path.split('/')[-1]

        #structure
        structure_ref_path = image_path.replace('/image/', '/structured-image/')
        structure_ref = Image.open(structure_ref_path)
        if not structure_ref.mode == "RGB":
            structure_ref = structure_ref.convert("RGB")
        structure_ref_tensor = self.dino_processor(images=structure_ref, return_tensors="pt").pixel_values[0]
        example["structure"] = structure_ref_tensor

        #apprearance
        new_path = image_path.replace('/image/', '/object/')
        file_stem = os.path.splitext(os.path.basename(new_path))[0]
        appearance_ref_path = os.path.join(os.path.dirname(new_path), file_stem, '')
        if os.path.exists(appearance_ref_path):
            images = [load_image(os.path.join(appearance_ref_path, f))
            for f in os.listdir(appearance_ref_path) if f.lower().endswith(".png")]
            appearance_ref_tensor = [self.dino_processor(images=img, return_tensors="pt").pixel_values[0] for img in images]
            appearance_ref_tensor = torch.stack(appearance_ref_tensor, dim=0)
            b, c, h, w = appearance_ref_tensor.shape
            if b<6:
                appearance_ref_tensor_padded = torch.zeros(6, c, h, w, dtype=appearance_ref_tensor.dtype, device=appearance_ref_tensor.device)
                appearance_ref_tensor_padded[:b] = appearance_ref_tensor
            appearance_ref_num = len(appearance_ref_tensor)
            example["appearance"] = appearance_ref_tensor_padded
            example["appearance_num"] = appearance_ref_num
        else:
            appearance_ref_tensor_padded = torch.zeros(6, 3, 224, 224, dtype=structure_ref_tensor.dtype, device=structure_ref_tensor.device)
            example["appearance"] = appearance_ref_tensor_padded
            example["appearance_num"] = 0

        raw_image = image
        if not image.mode == "RGB":
            raw_image = raw_image.convert("RGB")
        
        # original size
        original_width, original_height = raw_image.size
        original_size = torch.tensor([original_height, original_width])
        
        image_tensor = self.transform(raw_image)
        
        # random crop
        delta_h = image_tensor.shape[1] - self.size
        delta_w = image_tensor.shape[2] - self.size
        assert not all([delta_h, delta_w])
        
        if self.center_crop:
            top = delta_h // 2
            left = delta_w // 2
        else:
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        image = transforms.functional.crop(
            image_tensor, top=top, left=left, height=self.size, width=self.size
        )
        crop_coords_top_left = torch.tensor([top, left]) 

        example["pixel_values"] = image
        example["original_size"] = original_size
        example["crop_coords_top_left"] = crop_coords_top_left
        example["target_size"] = torch.tensor([self.size, self.size])

        json_name = image_name.split('.')[0]+'.json'
        json_path = os.path.join(self.image_json_path, json_name)
        with open(json_path, 'r', encoding='utf-8') as file:
            self.templates = json.load(file)
        # text = random.choice(self.templates[image_name])
        text = self.templates[image_name][0]

        #box
        box_json_name = image_name.split('.')[0]+'_box.json'
        box_json_path = os.path.join(self.image_box_path, box_json_name)
        with open(box_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        label=0
        canvas = create_canvas(H=512, W=512)
        box_mask = []
        for shape in data['shapes']:
            points = shape['points']
            x1, y1 = points[0]
            x2, y2 = points[1]
            x1, y1, x2, y2 = (int(x1*512/original_width)), (int(y1*512/original_width)), (int(x2*512/original_width)), (int(y2*512/original_width))
            if x1>512 or y1>512 or x2>512 or y2>512:
                raise ValueError('xy>512')
            box=[x1, y1, x2, y2]
            canvas = draw_one_box(canvas, box, label, thickness=3, fill=False)
            label+=1
        box_mask = to_pil(canvas)
        # box_mask.save(f"box.png") 
        box_img=self.transform_mask(box_mask)
        example["box_mask"] = box_img

        drop_image_embdes = 0
        possibility = random.random()
        if possibility < 0.05 :
            drop_image_embdes = 1
        elif possibility < 0.1:
            text = ''
        elif possibility < 0.15:
            drop_image_embdes = 1
            text = ''

        example["drop_image_embeds"] = drop_image_embdes
        example["input_ids"] = self.tokenizer( 
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0] 

        example["input_ids_2"] = self.tokenizer_2(
            text,
            max_length=self.tokenizer_2.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids[0]
        
        return example 

def get_full_repo_name(model_id: str,
                       organization: Optional[str] = None,
                       token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"

def importance_sampling_fn(t, max_t, alpha):
    """Importance Sampling Function f(t)"""
    return 1 / max_t * (1 - alpha * math.cos(math.pi * t / max_t))

def main():

    args = parse_args()
    
    def init_logger(filename):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

        # write to file
        handler = logging.FileHandler(filename, mode='w')
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # print to console
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        console.setFormatter(formatter)
        logger.addHandler(console)
        
        return logger

    os.makedirs('~/', exist_ok=True)
    logging_dir = os.path.join('~/', 'log.txt')
    logger = init_logger(logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=
        logging_dir,  # logging_dir=logging_dir, # depends on accelerator vesion
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(
                    Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            repo = Repository(
                args.output_dir, clone_from=repo_name, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"),
                      "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    # Load scheduler, tokenizer and models.
    args.pretrained_model_name_or_path = '~/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b/'
    ip_ckpt = "InstantStyle-main/sdxl_models/ip-adapter_sdxl.bin"
    
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2")
    
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    dino_v2 = AutoModel.from_pretrained('facebook/dinov2-with-registers-giant')
    
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    dino_v2.requires_grad_(False)

    # For mixed precision training we cast the unet and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device) # use fp32
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    dino_v2.to(accelerator.device, dtype=weight_dtype)      
    
    text_embedding_projector = WindowAwareLinearProjection(1536, 512, 32, args.train_text).to('cuda')

    # init adapter modules
    if args.train_text:
        num_tokens = 128
        attn_procs = {}
        unet_sd = unet.state_dict()
        target_blocks = ['down_blocks.2.attentions.1'] #['down_blocks.2.attentions.1','up_blocks.0.attentions.1'] 
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
                for block_name in target_blocks:
                    if block_name in name:
                        selected = True
                        break
                if selected:
                    layer_name = name.split(".processor")[0]
                    weights = {
                        "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                        "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                    }
                    attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=num_tokens, skip=False)
                    attn_procs[name].load_state_dict(weights)
                else:
                    attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=num_tokens, skip=True)
        unet.set_attn_processor(attn_procs)
        adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
        state_dict_ip = torch.load(ip_ckpt, map_location="cpu")
        adapter_modules.load_state_dict(state_dict_ip["ip_adapter"], strict=False)
    else:
        num_tokens = 512
        attn_procs = {}
        unet_sd = unet.state_dict()
        target_blocks = ['blocks'] 
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
                for block_name in target_blocks:
                    if block_name in name:
                        selected = True
                        break
                if selected: 
                    layer_name = name.split(".processor")[0]
                    weights = {
                        "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                        "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                        # "to_k_cond.weight": unet_sd[layer_name + ".to_k.weight"],
                        # "to_v_cond.weight": unet_sd[layer_name + ".to_v.weight"],
                    }
                    attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=num_tokens, skip=False, train_text=args.train_text)
                    attn_procs[name].load_state_dict(weights)
                else:
                    attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=num_tokens, skip=True, train_text=args.train_text)
        unet.set_attn_processor(attn_procs)
        adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
        state_dict_ip = torch.load(ip_ckpt, map_location="cpu")
        adapter_modules.load_state_dict(state_dict_ip["ip_adapter"], strict=False)

    class IPAdapter(torch.nn.Module):
        def __init__(self, unet, text_embedding_projector, adapter_modules):
            super().__init__()
            self.unet = unet
            self.text_embedding_projector = text_embedding_projector
            self.adapter_modules = adapter_modules

        def forward(self, noisy_latents, timesteps, text_embeds, dino, batch, unet_added_cond_kwargs):
            text_embeds, cond_embeds = text_embedding_projector(text_embeds, dino, batch)
            encoder_hidden_states = text_embeds
            encoder_hidden_states = torch.cat([encoder_hidden_states, cond_embeds], dim=1)
            # Predict the noise residual
            noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=unet_added_cond_kwargs).sample
            return noise_pred
    
    ip_adapter = IPAdapter(unet, text_embedding_projector, adapter_modules) 

    if not args.train_text:
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in ip_adapter.text_embedding_projector.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in ip_adapter.adapter_modules.parameters()]))

        ckpt_path = args.ckpt_path
        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        ip_adapter.text_embedding_projector.load_state_dict(state_dict["text_embedding_projector"], strict=False)
        ip_adapter.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=False)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in ip_adapter.text_embedding_projector.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in ip_adapter.adapter_modules.parameters()]))

        # Verify if the weights have 
        assert new_ip_proj_sum != orig_ip_proj_sum, "Weights of text_embedding_projector did not change!"
        assert new_adapter_sum != orig_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"\n Successfully loaded weights from checkpoint")

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    print('learning_rate:', args.learning_rate)
    params_to_opt = itertools.chain(ip_adapter.text_embedding_projector.parameters(), ip_adapter.adapter_modules.parameters()) 
    optimizer = torch.optim.AdamW(
        params_to_opt,    
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # Dataset and DataLoaders creation:
    train_dataset = ReVersionDataset(
        data_root=args.train_data_dir,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        size=args.resolution,
        repeats=args.repeats,
        center_crop=True,
        set="train",
        )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers)
    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil( 
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(  
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps *
        args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps *
        args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    ip_adapter, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        ip_adapter, optimizer, train_dataloader, lr_scheduler)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil((args.max_train_steps - args.first_step) /
                                      num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("textual_inversion", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps  #多卡时，一个batch_size等于单卡batch_size*卡数*梯度累加数

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = args.first_step
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process)  #disable=True时进度条不展示
    progress_bar.set_description("Steps")   #tqdm进度条取名Steps

    # Relation-Focal Importance Sampling
    args.importance_sampling = True
    if args.importance_sampling:
        print("Using Relation-Focal Importance Sampling")
        list_of_candidates = [
            x for x in range(noise_scheduler.config.num_train_timesteps)
        ]
        prob_dist = [
            importance_sampling_fn(x,
                                   noise_scheduler.config.num_train_timesteps,
                                   args.scaled_cosine_alpha)
            for x in list_of_candidates
        ]
        prob_sum = 0
        # normalize the prob_list so that sum of prob is 1
        for i in prob_dist:
            prob_sum += i
        prob_dist = [x / prob_sum for x in prob_dist]

    sum_loss = 0
    for epoch in range(first_epoch, args.num_train_epochs): 
        #text_encoder.train() 
        for step, batch in enumerate(train_dataloader): 
            with accelerator.accumulate(ip_adapter):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=torch.float32)).latent_dist.sample().detach() 
                latents = latents * vae.config.scaling_factor  
                latents = latents.to(accelerator.device, dtype=weight_dtype)
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents) 
                noise_offset = None
                if noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1)).to(accelerator.device, dtype=weight_dtype)
                
                bsz = latents.shape[0]
                # timestep (t) sampling
                timesteps = torch.randint( 
                    0,
                    noise_scheduler.config.num_train_timesteps, (bsz, ),
                    device=latents.device)
                # Relation-Focal Importance Sampling
                if args.importance_sampling:
                    timesteps = np.random.choice(
                        list_of_candidates,
                        size=bsz,
                        replace=True,
                        p=prob_dist)
                    timesteps = torch.tensor(timesteps).cuda()                  
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(
                    latents, noise, timesteps)
                
                with torch.no_grad():
                    encoder_output = text_encoder(batch['input_ids'].to(accelerator.device), output_hidden_states=True)
                    text_embeds = encoder_output.hidden_states[-2]
                    encoder_output_2 = text_encoder_2(batch['input_ids_2'].to(accelerator.device), output_hidden_states=True)
                    pooled_text_embeds = encoder_output_2[0]
                    text_embeds_2 = encoder_output_2.hidden_states[-2]
                    text_embeds = torch.concat([text_embeds, text_embeds_2], dim=-1) # concat
               
                # add cond for positive
                add_time_ids = [
                    batch["original_size"].to(accelerator.device),
                    batch["crop_coords_top_left"].to(accelerator.device),
                    batch["target_size"].to(accelerator.device),
                ]
                add_time_ids = torch.cat(add_time_ids, dim=1).to(accelerator.device, dtype=weight_dtype)
                unet_added_cond_kwargs = {"text_embeds": pooled_text_embeds, "time_ids": add_time_ids}
                
                noise_pred = ip_adapter(noisy_latents, timesteps, text_embeds, dino_v2, batch, unet_added_cond_kwargs)               

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(
                        latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                loss = 0.0

                # L_denoise
                if args.snr_gamma is None:
                    denoise_loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    if noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective requires that we add one to SNR values before we divide by them.
                        snr = snr + 1
                    mse_loss_weights = (
                            torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )

                    denoise_loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
                    denoise_loss = denoise_loss.mean(dim=list(range(1, len(denoise_loss.shape)))) * mse_loss_weights
                    denoise_loss = denoise_loss.mean()
                weighted_denoise_loss = args.denoise_loss_weight * denoise_loss
                loss += weighted_denoise_loss

                # pre = ip_adapter.text_embedding_projector.model_global_out[1].weight.data.detach().clone()
                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # after = ip_adapter.text_embedding_projector.model_global_out[1].weight.data.detach().clone()
                # x = (pre!=after).sum()
                sum_loss += loss.detach().item()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                # print(weighted_denoise_loss)
                progress_bar.update(1)

                global_step += 1
                if global_step%args.save_steps == 0:  
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    if accelerator.is_main_process:
                        text_save = os.path.join(save_path, "checkpoint")
                        unwrapped_text_embedding_projector = accelerator.unwrap_model(text_embedding_projector)
                        state_dict = unwrapped_text_embedding_projector.state_dict()
                        save_file(state_dict, text_save)
                        logger.info(f"Saved text_embeddings state to {text_save}") 

                        accelerator.save_state(save_path)
                        revert_model.revert_model(
                            ckpt= f'{save_path}/model.safetensors',
                            outpdir_adapter_bin= f'{save_path}/text_embedding_projector.bin')
                        logger.info(f"Saved adapter state to {save_path}") 

                mean_step = args.save_steps
                if global_step%mean_step == 0: 
                    if accelerator.is_main_process:
                        logger.info(f"predcition_type: {noise_scheduler.config.prediction_type} lr: {lr_scheduler.get_last_lr()[0]} total_loss: {loss.detach().item()} weighted_denoise_loss: {weighted_denoise_loss.detach().item()} mean_loss: {(sum_loss/mean_step)}")
                        sum_loss = 0

                    #eval
            if global_step >= args.max_train_steps:
                break

        if args.push_to_hub:
            repo.push_to_hub(
                commit_message="End of training",
                blocking=False,
                auto_lfs_prune=True)

    accelerator.end_training()

if __name__ == "__main__":
    main() 
