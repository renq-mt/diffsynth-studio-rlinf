# Experiment ledger

## dual_branch_xattn_safe_stage1_20260502

- Date: 2026-05-02
- Branch/worktree: `worktree-exp+dual-branch-rgb-depth-xattn-safe`
- Local commit: `251057b9fd4bd878e2809c1c6cfccdcc92bf14b6` plus dirty worktree changes synced with rsync
- Host: `sh03.rlinf.wan` (`aidev-208-3`), 8x A100 80GB
- Remote repo: `/workspace/qian.ren/robot_vla/diffsynth-studio-rlinf-exp+dual-branch-rgb-depth-xattn-safe`
- Launch script: `examples/wanvideo/model_training/full/Wan2.2-TI2V-5B_rlinf_lerobot_dual_branch.sh`
- Tmux session: `exp_dual_xattn_stage1_0502_r2`
- Output path: `outputs/dual_branch_xattn_safe_stage1_20260502`
- Log: `outputs/dual_branch_xattn_safe_stage1_20260502/train.log`
- Dataset: `/cephfs/qian.ren/huggingface_download/datasets/Libero/libero_goal_lerobot_mask_depth`
- Split: `LiberoRGBDDataset` fallback split, train episodes `[0, 450)`, validation last 10%
- Checkpoints:
  - RGB branch init: `/workspace/qian.ren/robot_vla/diffsynth-studio-rlinf/outputs/libero_goal_baseline/epoch-99.safetensors`
  - Depth branch init: `/workspace/qian.ren/robot_vla/diffsynth-studio-rlinf/outputs/libero_goal_baseline/epoch-99.safetensors`
- Key args: `--use_dual_branch --freeze_rgb_dit --learning_rate 1e-6 --num_epochs 100 --save_epochs 5 --val_interval 5 --libero_context_frames 5 --libero_prediction_frames 8 --use_gradient_checkpointing`
- Startup status: reached Epoch 0 training loop; observed `loss_rgb`/`loss_depth` logging and all 8 GPUs at 100% utilization.
- Follow-up: first run reached `epoch-4.safetensors`, then failed after validation at epoch 5 because inference reset `pipe.scheduler.timesteps` to 5 steps. Patched `WanVideoPipeline.training_loss()` to restore 1000 training timesteps before sampling and made the stage-1 script accept `DEPTH_CKPT`/`RGB_CKPT` overrides for restart.
- Restart: `exp_dual_xattn_stage1_0502_r3`, output `outputs/dual_branch_xattn_safe_stage1_20260502_resume_ep4`, launched with `DEPTH_CKPT=outputs/dual_branch_xattn_safe_stage1_20260502/epoch-4.safetensors`; reached Epoch 0 training loop with `loss_rgb`/`loss_depth` logging.
- Follow-up 2: restart reached validation at epoch 4, saved val videos, then rank0 failed with `KeyError: 'input_latents'` after validation. Cause: validation only runs on rank0 and leaves `pipe.scheduler.training=False`; `WanVideoUnit_InputVideoEmbedder` then omits `input_latents` before `training_loss()` can restore scheduler state. Patched `WanTrainingModule.forward_preprocess()` to restore training timesteps before pipeline units run.
