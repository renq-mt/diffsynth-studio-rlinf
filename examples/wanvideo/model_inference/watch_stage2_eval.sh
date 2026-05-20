#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-"/workspace/qian.ren/robot_vla/diffsynth-studio-rlinf-exp+dual-branch-rgb-depth-xattn-safe"}
TRAIN_SESSION=${TRAIN_SESSION:-"exp_dual_xattn_stage2_sh08_ep99"}
CKPT=${CKPT:-"outputs/dual_branch_stage2_from_stage1_sh08_ep99/epoch-299.safetensors"}
OUTPUT_PATH=${OUTPUT_PATH:-"outputs/eval_dual_branch_stream_stage2_sh08_ep299_50eps"}
RGB_CKPT=${RGB_CKPT:-"/workspace/qian.ren/robot_vla/diffsynth-studio-rlinf/outputs/libero_goal_baseline/epoch-99.safetensors"}
VAE_CKPT=${VAE_CKPT:-"/workspace/qian.ren/robot_vla_cuda/gac_wm/diffsynth-studio/models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"}
DATASET=${DATASET:-"/cephfs/qian.ren/huggingface_download/datasets/Libero/libero_goal_lerobot_mask_depth"}
PYTHON=${PYTHON:-"/opt/venv/openvla-oft/bin/python"}
CUDA_DEVICE=${CUDA_DEVICE:-"1"}
NUM_EPISODES=${NUM_EPISODES:-50}

cd "${REPO}"
mkdir -p "${OUTPUT_PATH}"
WATCH_LOG="${OUTPUT_PATH}/watch.log"
EVAL_LOG="${OUTPUT_PATH}/run.log"

echo "watch_started $(date)" >> "${WATCH_LOG}"
while tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null; do
  echo "training_still_running $(date)" >> "${WATCH_LOG}"
  sleep 300
done
echo "training_session_ended $(date)" >> "${WATCH_LOG}"

if [ ! -f "${CKPT}" ]; then
  echo "missing_final_checkpoint ${CKPT}" >> "${WATCH_LOG}"
  exit 1
fi

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON}" examples/wanvideo/model_inference/eval_dual_branch_stream.py \
  --rgb_ckpt "${RGB_CKPT}" \
  --ckpt "${CKPT}" \
  --vae_ckpt "${VAE_CKPT}" \
  --dataset "${DATASET}" \
  --output_path "${OUTPUT_PATH}" \
  --num_episodes "${NUM_EPISODES}" \
  --steps 5 \
  --device cuda:0 \
  > "${EVAL_LOG}" 2>&1

echo "eval_finished $(date)" >> "${WATCH_LOG}"
