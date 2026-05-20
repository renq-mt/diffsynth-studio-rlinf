# Ablation: two independent specialist branches, no cross-branch attention.
# pipe.dit    ← RGB-only pretrain  (libero_goal_baseline/epoch-99)
# pipe.depth_dit ← depth-only pretrain (wt-depth-only-quickval/epoch-99)
# cross_branch_attentions ← zero-init (identity; --depth_only_ckpt auto-disables xattn)
# Answers: is the depth quality gap caused by the dual-branch structure itself,
# or by joint xattn training?

CUDA_VISIBLE_DEVICES=2
python \
  examples/wanvideo/model_inference/eval_dual_branch_stream.py \
  --rgb_ckpt /workspace/qian.ren/robot_vla/diffsynth-studio-rlinf/outputs/libero_goal_baseline/epoch-99.safetensors \
  --depth_only_ckpt /workspace/qian.ren/robot_vla/wt-depth-only-quickval/outputs/depth_only_quickval_single_rlinf_init/epoch-99.safetensors \
  --vae_ckpt /workspace/qian.ren/robot_vla_cuda/gac_wm/diffsynth-studio/models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
  --dataset /cephfs/qian.ren/huggingface_download/datasets/Libero/libero_goal_lerobot_mask_depth \
  --output_path outputs/eval_noxattn_depthonly_ep99 \
  --num_episodes 50 --steps 5 --device cuda:0
