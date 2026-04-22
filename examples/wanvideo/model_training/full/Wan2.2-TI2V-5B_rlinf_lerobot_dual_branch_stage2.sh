# Stage-2 joint finetuning: both RGB dit and depth dit are unfrozen.
# Load stage-1 checkpoint for depth_dit; load rgb-only checkpoint for pipe.dit.
# Requires 8x80GB GPUs with ZeRO-2 (accelerate_config_14B.yaml).
#
# Before running, set:
#   STAGE1_DEPTH_CKPT : depth_dit checkpoint saved at the end of stage-1
#   RGB_CKPT          : rgb-only pretrained checkpoint (same as used in stage-1)

STAGE1_DEPTH_CKPT=${STAGE1_DEPTH_CKPT:-"outputs/dual_branch_stage1/epoch-XXXX.safetensors"}
RGB_CKPT=${RGB_CKPT:-"/workspace/qian.ren/huggingDown/RLinf/RLinf-Wan-LIBERO-Goal/epoch-XXXX.safetensors"}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
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
  --libero_max_depth 5.0 \
  --use_dual_branch \
  --no-freeze_rgb_dit \
  --rgb_pretrain_checkpoint ${RGB_CKPT} \
  --depth_pretrain_checkpoint ${STAGE1_DEPTH_CKPT} \
  --use_gradient_checkpointing \
  --learning_rate 5e-6 \
  --num_epochs 100000 \
  --save_epochs 50 \
  --val_interval 50 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "outputs/dual_branch_stage2" \
  --context_noise_sigma 0.0 \
  --static_video_prob 0.05
