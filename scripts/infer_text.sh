#!/usr/bin/env bash
set -euo pipefail

python infer.py \
  --mode text \
  --checkpoint /home/lyf/2D/MoEdit-pami-github/checkpoints/text-final/mogen_adapter.bin \
  --prompt "six reddish-brown mushrooms, in the misty forest" \
  --num_samples 4 \
  --guidance_scale 5 \
  --num_inference_steps 30 \
  --seed 43 \
  --attention_backend sdpa
