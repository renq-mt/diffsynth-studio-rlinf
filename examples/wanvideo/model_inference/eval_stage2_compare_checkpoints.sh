#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-"/workspace/qian.ren/robot_vla/diffsynth-studio-rlinf-exp+dual-branch-rgb-depth-xattn-safe"}
PYTHON=${PYTHON:-"/opt/venv/openvla-oft/bin/python"}
CUDA_DEVICE=${CUDA_DEVICE:-"1"}
RGB_CKPT=${RGB_CKPT:-"/workspace/qian.ren/robot_vla/diffsynth-studio-rlinf/outputs/libero_goal_baseline/epoch-99.safetensors"}
VAE_CKPT=${VAE_CKPT:-"/workspace/qian.ren/robot_vla_cuda/gac_wm/diffsynth-studio/models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"}
DATASET=${DATASET:-"/cephfs/qian.ren/huggingface_download/datasets/Libero/libero_goal_lerobot_mask_depth"}
CKPT_DIR=${CKPT_DIR:-"outputs/dual_branch_stage2_from_stage1_sh08_ep99"}
OUT_ROOT=${OUT_ROOT:-"outputs/eval_dual_branch_stream_stage2_compare_timepoints"}
NUM_EPISODES=${NUM_EPISODES:-50}

cd "${REPO}"
mkdir -p "${OUT_ROOT}"

for EPOCH in 49 99 149; do
  CKPT="${CKPT_DIR}/epoch-${EPOCH}.safetensors"
  OUT="${OUT_ROOT}/epoch-${EPOCH}_50eps"
  LOG="${OUT}/run.log"
  mkdir -p "${OUT}"
  echo "eval_start epoch-${EPOCH} $(date)" | tee -a "${OUT_ROOT}/compare.log"
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON}" examples/wanvideo/model_inference/eval_dual_branch_stream.py \
    --rgb_ckpt "${RGB_CKPT}" \
    --ckpt "${CKPT}" \
    --vae_ckpt "${VAE_CKPT}" \
    --dataset "${DATASET}" \
    --output_path "${OUT}" \
    --num_episodes "${NUM_EPISODES}" \
    --steps 5 \
    --device cuda:0 \
    > "${LOG}" 2>&1
  echo "eval_done epoch-${EPOCH} $(date)" | tee -a "${OUT_ROOT}/compare.log"
done

"${PYTHON}" - <<'PY'
import json
import os

root = "outputs/eval_dual_branch_stream_stage2_compare_timepoints"
rows = []
for epoch in (49, 99, 149):
    path = os.path.join(root, f"epoch-{epoch}_50eps", "summary.json")
    with open(path, "r", encoding="utf-8") as f:
        s = json.load(f)
    rows.append({
        "epoch": epoch,
        "mean_psnr": s.get("mean_psnr"),
        "std_psnr": s.get("std_psnr"),
        "fvd_r3d18": s.get("fvd_r3d18"),
        "mean_depth_mae_m": s.get("mean_depth_mae_m"),
        "mean_depth_rmse_m": s.get("mean_depth_rmse_m"),
        "mean_time_s": s.get("mean_time_s"),
        "summary": path,
    })

out = os.path.join(root, "compare_summary.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)
print(json.dumps(rows, indent=2, ensure_ascii=False))
PY
