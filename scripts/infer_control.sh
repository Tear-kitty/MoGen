#!/usr/bin/env bash
set -euo pipefail

python infer.py \
  --mode control \
  --checkpoint /home/lyf/2D/MoEdit-pami-github/checkpoints/control-final/mogen_adapter.bin \
  --prompt "six reddish-brown mushrooms, in the misty forest" \
  --structure_image /home/lyf/2D/MoEdit-pami-github/data/structure/45.png \
  --box_json /home/lyf/2D/MoEdit-pami-github/data_test/box/62.json \
  --appearance_dir /home/lyf/2D/MoEdit-pami-github/data/object/47/ \
  --num_samples 4 \
  --guidance_scale 5 \
  --num_inference_steps 30 \
  --seed 43 \
  --attention_backend sdpa
