python \
  examples/wanvideo/model_inference/eval_dual_branch_stream.py \
  --rgb_ckpt /workspace/qian.ren/robot_vla/diffsynth-studio-rlinf/outputs/libero_goal_baseline/epoch-99.safetensors \
  --ckpt /workspace/qian.ren/robot_vla/diffsynth-studio-rlinf-exp+dual-branch-rgb-depth-xattn-safe/outputs/dual_branch_xattn/epoch-4.safetensors \
  --vae_ckpt /workspace/qian.ren/robot_vla_cuda/gac_wm/diffsynth-studio/models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
  --dataset /cephfs/qian.ren/huggingface_download/datasets/Libero/libero_goal_lerobot_mask_depth \
  --output_path outputs/eval_dual_branch_stream_ep49_stg1 \
  --num_episodes 50 --steps 5 --device cuda:0