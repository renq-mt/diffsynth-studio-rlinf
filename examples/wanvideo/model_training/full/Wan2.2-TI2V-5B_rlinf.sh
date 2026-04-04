CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml \
  examples/wanvideo/model_training/train_rlinf.py \
  --height 256 \
  --width 256 \
  --num_frames 13 \
  --dataset_repeat 1 \
  --model_paths '[["/workspace/qian.ren/robot_vla_cuda/gac_wm/diffsynth-studio-mt/models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors","/workspace/qian.ren/robot_vla_cuda/gac_wm/diffsynth-studio-mt/models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors","/workspace/qian.ren/robot_vla_cuda/gac_wm/diffsynth-studio-mt/models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors"],"/workspace/qian.ren/robot_vla_cuda/gac_wm/diffsynth-studio-mt/models/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth","/workspace/qian.ren/robot_vla_cuda/gac_wm/diffsynth-studio-mt/models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"]' \
  --learning_rate 1e-5 \
  --num_epochs 100000 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "outputs/only4test" \
  --trainable_models "dit" \
  --context_noise_sigma 0.0 \
  --static_video_prob 0.05 \
  --extra_inputs "input_image,action" \
  --val_interval 50 \
  --save_epochs 50 \
  --dataset RLinfNpyDataset \
  --dataset_base_path /workspace/qian.ren/huggingDown/RLinf/RLinf-Wan-LIBERO-Spatial/dataset