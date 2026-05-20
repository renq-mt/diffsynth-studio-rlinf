# Stage-2 joint finetune starting from stage-1 epoch-99 dual-branch ckpt.
#
# Init state:
#   pipe.dit                ← libero_goal_baseline/epoch-99   (RGB, was frozen in stage-1; now unfrozen)
#   pipe.depth_dit          ← dual_branch_xattn/epoch-99      (from stage-1 xattn-trained depth)
#   cross_branch_attentions ← dual_branch_xattn/epoch-99      (from stage-1, preserves xattn behavior)
#
# Requires the train_rlinf.py fix that splits the depth_pretrain_checkpoint by
# prefix into depth_dit + cross_branch_attentions state dicts. Without that fix,
# the stage-1 xattn weights are silently discarded.

STAGE1_DEPTH_CKPT=${STAGE1_DEPTH_CKPT:-"/workspace/qian.ren/robot_vla/diffsynth-studio-rlinf-exp+dual-branch-rgb-depth-xattn-safe/outputs/dual_branch_xattn_safe_stage1_20260503_sh08_resume_ep14/epoch-99.safetensors"}
RGB_CKPT=${RGB_CKPT:-"/workspace/qian.ren/robot_vla/diffsynth-studio-rlinf/outputs/libero_goal_baseline/epoch-99.safetensors"}
OUTPUT_PATH=${OUTPUT_PATH:-"outputs/dual_branch_stage2_from_stage1_sh08_ep99"}
ACCELERATE_BIN=${ACCELERATE_BIN:-"/opt/venv/openvla-oft/bin/accelerate"}
NUM_EPOCHS=${NUM_EPOCHS:-300}
SAVE_EPOCHS=${SAVE_EPOCHS:-10}
VAL_INTERVAL=${VAL_INTERVAL:-10}

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
  --dataset LiberoRGBDDataset \
  --libero_dataset_name /cephfs/qian.ren/huggingface_download/datasets/Libero/libero_goal_lerobot_mask_depth \
  --libero_context_frames 5 \
  --libero_prediction_frames 8 \
  --libero_image_size 256 256 \
  --libero_action_dim 7 \
  --use_dual_branch \
  --no-freeze_rgb_dit \
  --rgb_pretrain_checkpoint "${RGB_CKPT}" \
  --depth_pretrain_checkpoint "${STAGE1_DEPTH_CKPT}" \
  --use_gradient_checkpointing \
  --learning_rate 5e-6 \
  --num_epochs "${NUM_EPOCHS}" \
  --save_epochs "${SAVE_EPOCHS}" \
  --val_interval "${VAL_INTERVAL}" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_PATH}" \
  --context_noise_sigma 0.0 \
  --static_video_prob 0.05
