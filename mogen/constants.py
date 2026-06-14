"""Project-wide defaults.

The path defaults intentionally keep the original server layout. Override them from
CLI when running on another machine.
"""

DEFAULT_BASE_MODEL_PATH = (
    "models--stabilityai--stable-diffusion-xl-base-1.0/"
)

DEFAULT_BASE_ADAPTER_PATH = (
    "sdxl_models/ip-adapter_sdxl.bin"
)

DEFAULT_TRAIN_DATA_DIR = "./data/"
DEFAULT_OUTPUT_DIR = "./checkpoints/"
DEFAULT_RESULTS_DIR = "./results/"

DINO_MODEL_NAME = "facebook/dinov2-with-registers-giant"

TEXT_TARGET_BLOCKS = [
    "down_blocks.2.attentions.1",
]

# Matches every SDXL UNet cross-attention block name that contains "blocks".
CONTROL_TARGET_BLOCKS = ["blocks"]

TEXT_NUM_TOKENS = 64
CONTROL_NUM_TOKENS = 256
MAX_APPEARANCE_REFS = 20
MAX_BOXES = 64
BOX_LABEL_HASH_SIZE = 65536

DEFAULT_NEGATIVE_PROMPT = (
    "text, watermark, lowres, low quality, worst quality, deformed, glitch, "
    "low contrast, noisy, saturation, blurry, bad anatomy"
)
