#!/usr/bin/env bash
set -euo pipefail

accelerate launch --config_file acc_configs/gpu8.yaml train_text.py \
  --train_data_dir /home/lyf/2D/MoEdit-pami-github/data_test/ \
  --output_dir /home/lyf/2D/MoEdit-pami-github/checkpoints/ \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-5 \
  --max_train_steps 100000 \
  --save_steps 1000 \
  --mixed_precision fp16 \
  --text_drop_prob 0.10 \
  --attention_backend flash
