#!/usr/bin/env bash
set -euo pipefail

accelerate launch --config_file acc_configs/gpu8.yaml train_control.py \
  --train_data_dir /home/lyf/2D/MoEdit-pami-github/data_test/ \
  --output_dir /home/lyf/2D/MoEdit-pami-github/checkpoints/ \
  --init_text_checkpoint /home/lyf/2D/MoEdit-pami-github/checkpoints/text-final/mogen_adapter.bin \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-5 \
  --max_train_steps 100000 \
  --save_steps 1000 \
  --mixed_precision fp16 \
  --attention_backend flash \
  --box_jitter_prob 0.50 \
  --structure_degrade_prob 0.35 \
  --appearance_degrade_prob 0.45 \
  --appearance_count_error_prob 0.20
