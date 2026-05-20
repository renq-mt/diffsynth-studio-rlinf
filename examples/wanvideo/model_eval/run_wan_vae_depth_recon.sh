#!/bin/bash
set -euo pipefail

REPO=/workspace/qian.ren/robot_vla/diffsynth-studio-rlinf
PYTHON=/opt/venv/openvla-oft/bin/python
VAE_CKPT=/workspace/qian.ren/robot_vla_cuda/gac_wm/diffsynth-studio/models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
DATASET=/cephfs/qian.ren/huggingface_download/datasets/Libero/libero_goal_lerobot_mask_depth
OUTPUT=$REPO/outputs/wan_vae_depth_eval

mkdir -p "$OUTPUT"
cd "$REPO"

CUDA_VISIBLE_DEVICES=0 "$PYTHON" examples/wanvideo/model_eval/eval_wan_vae_depth_recon.py \
  --vae_ckpt "$VAE_CKPT" \
  --dataset "$DATASET" \
  --output_path "$OUTPUT" \
  --device cuda:0 \
  --dtype bfloat16 \
  --split validation \
  --num_samples 100 \
  --num_visualizations 8 \
  --context_frames 5 \
  --prediction_frames 4 \
  --height 256 \
  --width 256 \
  --action_dim 7 \
  --fps 4
