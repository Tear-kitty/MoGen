<div align="center">

# MoGen Refactored

### A flexible SDXL framework for controllable multi-object image generation

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](#installation)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](#installation)
[![Diffusers](https://img.shields.io/badge/Diffusers-SDXL-8A2BE2?style=flat-square)](#model-zoo--checkpoints)
[![DINOv2](https://img.shields.io/badge/DINOv2-with--registers--giant-0F766E?style=flat-square)](#control-stage)
[![Accelerate](https://img.shields.io/badge/Accelerate-Multi--GPU-111827?style=flat-square)](#training)
[![arXiv](https://img.shields.io/badge/arXiv-2601.05546-B31B1B?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2601.05546)

**MoGen: A Unified Collaborative Framework for Controllable Multi-Object Image Generation**  
[[Paper]](https://arxiv.org/abs/2601.05546) · [[Project / Code]](https://github.com/Tear-kitty/MoGen) · [[Environment Setup]](ENV_SETUP.md)

</div>

---

## Project Structure

```text
MoGen_refactored/
├── acc_configs/
│   ├── single_gpu.yaml
│   ├── gpu2.yaml
│   └── gpu8.yaml
├── mogen/
│   ├── adapter_setup.py
│   ├── attention_processor.py
│   ├── augmentations.py
│   ├── box_utils.py
│   ├── checkpointing.py
│   ├── constants.py
│   ├── data.py
│   ├── inference_preprocess.py
│   ├── pipeline.py
│   ├── projection.py
│   └── utils.py
├── scripts/
│   ├── train_text.sh
│   ├── train_control.sh
│   ├── infer_text.sh
│   └── infer_control.sh
├── train.py
├── train_text.py
├── train_control.py
├── infer.py
├── requirements.txt
├── environment.yml
├── ENV_SETUP.md
└── README.md
```

---

## Installation

A minimal setup is shown below. For a more detailed environment recipe, see [`ENV_SETUP.md`](ENV_SETUP.md).

```bash
conda create -n mogen python=3.10 -y
conda activate mogen

python -m pip install --upgrade pip setuptools wheel ninja packaging
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
MAX_JOBS=8 pip install flash-attn==2.6.3 --no-build-isolation

accelerate test --config_file acc_configs/gpu8.yaml
```

If your CUDA / PyTorch version differs, install the matching PyTorch and FlashAttention builds for your machine.

---

## Dataset

Dataset coming soon

## Training

### Stage 1: Text-only Training

```bash
accelerate launch --config_file acc_configs/gpu8.yaml train_text.py \
  --train_data_dir ${DATA_DIR} \
  --output_dir ${CKPT_DIR} \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-5 \
  --max_train_steps 100000 \
  --save_steps 1000 \
  --mixed_precision fp16 \
  --attention_backend flash \
  --text_drop_prob 0.10 \
  --resume_from_checkpoint None \
  --importance_sampling False \
  --use_global_text_in_control True \
```

The final adapter is saved to:

```text
checkpoints/text-final/mogen_adapter.bin
```

### Stage 2: Control Training

```bash
accelerate launch --config_file acc_configs/gpu8.yaml train_control.py \
  --train_data_dir ${DATA_DIR} \
  --output_dir ${CKPT_DIR} \
  --init_text_checkpoint ${CKPT_DIR}/text-final/mogen_adapter.bin \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-5 \
  --max_train_steps 100000 \
  --save_steps 1000 \
  --mixed_precision fp16 \
  --attention_backend flash \
  --text_drop_prob 0.05 \
  --control_drop_prob 0.05 \
  --both_drop_prob 0.05 \
  --box_jitter_prob 0.50 \
  --structure_degrade_prob 0.35 \
  --appearance_degrade_prob 0.45 \
  --appearance_count_error_prob 0.05 \
  --resume_from_checkpoint None \
  --importance_sampling True \
  --use_global_text_in_control True \
```

The final control adapter is saved to:

```text
checkpoints/control-final/mogen_adapter.bin
```

## Inference

### Text-only Inference

```bash
python infer.py \
  --mode text \
  --checkpoint ${CKPT_DIR}/text-final/mogen_adapter.bin \
  --prompt "..." \
  --output_dir ${RESULT_DIR} \
  --num_samples 4 \
  --guidance_scale 5.0 \
  --num_inference_steps 30 \
  --seed 43 \
  --torch_dtype fp16 \
  --attention_backend sdpa \
  --scale 0.8 \
  --global_text_scale 0.8 \
```
The two inference hyperparameters --scale and --global_text_scale control the strength of the learned MoGen condition and the global text guidance, respectively. Adjusting them can often lead to better visual quality, text alignment, and object consistency. In practice, values around 0.6–1.0 are good starting points.

### Control Inference

```bash
python infer.py \
  --mode control \
  --checkpoint ${CKPT_DIR}/control-final/mogen_adapter.bin \
  --prompt "..." \
  --structure_image ${DATA_DIR}/image/10.png \
  --box_json ${DATA_DIR}/box/10.json \
  --appearance_dir ${DATA_DIR}/object/10/ \
  --output_dir ${RESULT_DIR} \
  --num_samples 4 \
  --guidance_scale 5.0 \
  --num_inference_steps 30 \
  --seed 43 \
  --torch_dtype fp16 \
  --attention_backend sdpa \
  --scale 0.8 \
  --global_text_scale 0.8
```
The three control inputs are optional. During inference, MoGen automatically activates the control signal whose path is provided:

| Argument            | Control signal               |
| ------------------- | ---------------------------- |
| `--structure_image` | structure guidance           |
| `--box_json`        | bounding-box layout guidance |
| `--appearance_dir`  | object appearance guidance   |

---

## Model Zoo & Checkpoints

Suggested local layout:

```text
checkpoints/
├── text-final/
│   └── mogen_adapter.bin
└── control-final/
    └── mogen_adapter.bin
```

---

## Citation

If you use this codebase or build upon MoGen, please cite the paper:

```bibtex
@misc{li2026mogen,
  title  = {MoGen: A Unified Collaborative Framework for Controllable Multi-Object Image Generation},
  author = {Li, Yanfeng and Sun, Yue and Fu, Keren and Im, Sio-Kei and Liu, Xiaoming and Zhai, Guangtao and Liu, Xiaohong and Tan, Tao},
  year   = {2026},
  eprint = {2601.05546},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV}
}
```

---

## Acknowledgements

This implementation is built around SDXL, DINOv2, Hugging Face Diffusers, PyTorch, Accelerate, and FlashAttention. The research direction follows MoGen's unified controllable multi-object generation formulation, with this repository focusing on a clean, extensible, and multi-GPU friendly engineering version.

