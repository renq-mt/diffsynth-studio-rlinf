# XAttn-only training: RGB dit + depth dit both frozen, only cross_branch_attentions are trained.
# RGB dit   → initialized from RGB pretrained checkpoint, frozen.
# depth dit → initialized from depth-only pretrained checkpoint, frozen.
# cross_branch_attentions → zero-initialized, trained from scratch.
#
# Checkpoint saved: full model state dict (rgb_dit + depth_dit + cross_branch_attentions).
#
# Usage:
#   RGB_CKPT=<path-to-rgb-ckpt.safetensors>
#   DEPTH_CKPT=<path-to-depthonly-ckpt.safetensors>
#   bash examples/wanvideo/model_training/full/Wan2.2-TI2V-5B_rlinf_lerobot_xattn_only.sh

RGB_CKPT=${RGB_CKPT:-"/workspace/qian.ren/robot_vla/diffsynth-studio-rlinf/outputs/libero_goal_baseline/epoch-99.safetensors"}
DEPTH_CKPT=${DEPTH_CKPT:-"/workspace/qian.ren/robot_vla/wt-depth-only-quickval/outputs/depth_only_quickval_single_rlinf_init/epoch-99.safetensors"}

# CUDA_VISIBLE_DEVICES=2,4 /opt/venv/openvla-oft/bin/accelerate launch \
#   --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml \
  python examples/wanvideo/model_training/train_rlinf.py \
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
  --rgb_pretrain_checkpoint ${RGB_CKPT} \
  --depth_pretrain_checkpoint ${DEPTH_CKPT} \
  --freeze_rgb_dit \
  --freeze_depth_dit \
  --output_path "outputs/dual_branch_xattn_only" \
  --learning_rate 1e-4 \
  --num_epochs 100 \
  --save_epochs 5 \
  --val_interval 5 \
  --context_noise_sigma 0.0 \
  --static_video_prob 0.05 \
  --use_gradient_checkpointing
