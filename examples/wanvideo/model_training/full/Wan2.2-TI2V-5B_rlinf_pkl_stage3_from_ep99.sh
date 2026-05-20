# Stage-3 joint full-finetune on RLinf pickle rollouts, resumed from
# stage-2 dual-branch epoch-99.
#
# Init state (a single dual-branch ckpt is fed to BOTH flags so each branch is
# repopulated from the same source):
#   pipe.dit                ← stage-2 epoch-99   (RGB branch — the saved ckpt
#                                                 strips this prefix, leaving
#                                                 raw `blocks.*` keys that
#                                                 pipe.dit.load_state_dict
#                                                 picks up via strict=False)
#   pipe.depth_dit          ← stage-2 epoch-99   (pipe.depth_dit.* prefix)
#   cross_branch_attentions ← stage-2 epoch-99   (pipe.cross_branch_attentions.*
#                                                 prefix)
#
# All three modules are trainable (no --freeze_rgb_dit / no --freeze_depth_dit),
# matching the stage-2 joint-finetune intent.

CKPT=${CKPT:-"/workspace/qian.ren/robot_vla/diffsynth-studio-rlinf-exp+dual-branch-rgb-depth-xattn-safe/outputs/dual_branch_stage2_from_stage1_sh08_ep99/epoch-99.safetensors"}
PKL_PATH=${PKL_PATH:-"/workspace/qian.ren/robot_vla/rlinf-main/logs/20260518-05:25:17-libero_goal_grpo_collect_openvlaoft/libero_goal_grpo_collect_openvlaoft/rollout_lerobot"}
OUTPUT_PATH=${OUTPUT_PATH:-"outputs/dual_branch_stage3_rlinf_pkl_from_ep99"}
ACCELERATE_BIN=${ACCELERATE_BIN:-"/opt/venv/openvla-oft/bin/accelerate"}
NUM_EPOCHS=${NUM_EPOCHS:-200}
SAVE_EPOCHS=${SAVE_EPOCHS:-10}
VAL_INTERVAL=${VAL_INTERVAL:-10}
LR=${LR:-5e-6}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "${ACCELERATE_BIN}" launch \
  --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml \
  examples/wanvideo/model_training/train_rlinf.py \
  --height 256 \
  --width 256 \
  --num_frames 13 \
  --dataset_repeat 1 \
  --model_paths '[["/workspace/qian.ren/robot_vla_cuda/gac_wm/diffsynth-studio/models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors","/workspace/qian.ren/robot_vla_cuda/gac_wm/diffsynth-studio/models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors","/workspace/qian.ren/robot_vla_cuda/gac_wm/diffsynth-studio/models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors"],"/workspace/qian.ren/robot_vla_cuda/gac_wm/diffsynth-studio/models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth","/workspace/qian.ren/robot_vla_cuda/gac_wm/diffsynth-studio/models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"]' \
  --trainable_models "dit" \
  --extra_inputs "input_image,action" \
  --dataset RLinfPickleDataset \
  --rlinf_pkl_path "${PKL_PATH}" \
  --libero_context_frames 5 \
  --libero_prediction_frames 8 \
  --libero_image_size 256 256 \
  --libero_action_dim 7 \
  --use_dual_branch \
  --no-freeze_rgb_dit \
  --rgb_pretrain_checkpoint "${CKPT}" \
  --depth_pretrain_checkpoint "${CKPT}" \
  --use_gradient_checkpointing \
  --learning_rate "${LR}" \
  --num_epochs "${NUM_EPOCHS}" \
  --save_epochs "${SAVE_EPOCHS}" \
  --val_interval "${VAL_INTERVAL}" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_PATH}" \
  --context_noise_sigma 0.0 \
  --static_video_prob 0.05
