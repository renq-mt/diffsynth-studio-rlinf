#!/usr/bin/env bash
# Verification fine-tune: confirm RLinfLerobotDataset + multi-root aggregation
# works end-to-end by loading the dual-branch stage-2 epoch-99 checkpoint and
# running 1 epoch on the rlinf rollout output (aggregated across rank_*/id_*).
#
# After 1 epoch with --val_interval 1, _save_val_reconstruction will produce
# <OUTPUT_PATH>/val_samples/epoch-0000/{gen,gt,compare,depth_pred,depth_gt,depth_compare}.mp4
# which we eyeball to verify the dataset feeds the pipeline correctly.

set -e

FULL_CKPT=${FULL_CKPT:-"/workspace/qian.ren/robot_vla/diffsynth-studio-rlinf-exp+dual-branch-rgb-depth-xattn-safe/outputs/dual_branch_stage2_from_stage1_sh08_ep99/epoch-99.safetensors"}
ROLLOUT_PARENT=${ROLLOUT_PARENT:-"/workspace/qian.ren/robot_vla/rlinf-main/logs/20260514-09:34:40-libero_goal_grpo_collect_openvlaoft/libero_goal_grpo_collect_openvlaoft/rollout_lerobot"}
OUTPUT_PATH=${OUTPUT_PATH:-"outputs/verify_rollout_dataset"}
ACCELERATE_BIN=${ACCELERATE_BIN:-"/opt/venv/openvla-oft/bin/accelerate"}
NUM_EPOCHS=${NUM_EPOCHS:-1}
SAVE_EPOCHS=${SAVE_EPOCHS:-1}
VAL_INTERVAL=${VAL_INTERVAL:-1}
NUM_PROCESSES=${NUM_PROCESSES:-8}
GPU_IDS=${GPU_IDS:-"0,1,2,3,4,5,6,7"}

CUDA_VISIBLE_DEVICES=${GPU_IDS} "${ACCELERATE_BIN}" launch \
  --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml \
  examples/wanvideo/model_training/train_rlinf.py \
  --height 256 \
  --width 256 \
  --num_frames 13 \
  --dataset_repeat 1 \
  --model_paths '[["/workspace/qian.ren/robot_vla_cuda/gac_wm/diffsynth-studio/models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors","/workspace/qian.ren/robot_vla_cuda/gac_wm/diffsynth-studio/models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors","/workspace/qian.ren/robot_vla_cuda/gac_wm/diffsynth-studio/models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors"],"/workspace/qian.ren/robot_vla_cuda/gac_wm/diffsynth-studio/models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth","/workspace/qian.ren/robot_vla_cuda/gac_wm/diffsynth-studio/models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"]' \
  --trainable_models "dit" \
  --extra_inputs "input_image,action" \
  --dataset RLinfLerobotDataset \
  --libero_dataset_name "${ROLLOUT_PARENT}" \
  --libero_context_frames 5 \
  --libero_prediction_frames 8 \
  --libero_image_size 256 256 \
  --libero_action_dim 7 \
  --use_dual_branch \
  --no-freeze_rgb_dit \
  --rgb_pretrain_checkpoint "${FULL_CKPT}" \
  --depth_pretrain_checkpoint "${FULL_CKPT}" \
  --use_gradient_checkpointing \
  --learning_rate 5e-6 \
  --num_epochs "${NUM_EPOCHS}" \
  --save_epochs "${SAVE_EPOCHS}" \
  --val_interval "${VAL_INTERVAL}" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_PATH}" \
  --context_noise_sigma 0.0 \
  --static_video_prob 0.0
