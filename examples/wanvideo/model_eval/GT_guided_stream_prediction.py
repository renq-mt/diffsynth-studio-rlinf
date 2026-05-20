"""
GT-guided stream prediction: teacher-forcing evaluation.
At each chunk boundary, the model receives GT frames as context instead of its own predictions.
This measures per-chunk generation quality decoupled from error accumulation.

Usage:
    python examples/wanvideo/model_eval/GT_guided_stream_prediction.py \
        --model_ckpt <path> \
        --vae_ckpt <path> \
        --data_path <episode_dir> \
        --output_path outputs/eval/gt_guided \
        --total_frames 505 \
        --num_inference_steps 5
"""

import argparse
import os
import gc
import json
import numpy as np
import torch
from PIL import Image

try:
    torch.serialization.add_safe_globals(["set", "OrderedDict", "builtins.set"])
except AttributeError:
    pass

from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig


WINDOW = 9


def _filter_dual_branch_ckpt(model_ckpt):
    """
    Stage-2 dual-branch checkpoints contain pipe.depth_dit.* and
    pipe.cross_branch_attentions.* keys alongside bare RGB DiT keys.
    The model_manager uses key-hash detection, so these extra keys break detection.
    Strip them and cache the RGB-only checkpoint next to the original.
    """
    import safetensors.torch as st
    cached = model_ckpt + ".rgb_only.safetensors"
    if os.path.exists(cached):
        print(f"[ckpt] Using cached RGB-only checkpoint: {cached}")
        return cached
    raw = st.load_file(model_ckpt)
    if not any(k.startswith("pipe.") for k in raw):
        return model_ckpt
    filtered = {k: v for k, v in raw.items() if not k.startswith("pipe.")}
    st.save_file(filtered, cached)
    print(f"[ckpt] Dual-branch checkpoint — RGB-only keys cached to: {cached}")
    return cached


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_ckpt", required=True)
    p.add_argument("--vae_ckpt", required=True)
    # npy format (mutually exclusive with --dataset)
    p.add_argument("--data_path", default=None, help="Episode dir containing rgb.npy and actions.npy")
    # LeRobot format
    p.add_argument("--dataset", default=None, help="LeRobot dataset path (libero_goal_lerobot_mask_depth)")
    p.add_argument("--episode_idx", type=int, default=0, help="Episode index when using --dataset")
    p.add_argument("--output_path", default="outputs/eval/gt_guided")
    p.add_argument("--total_frames", type=int, default=505)
    p.add_argument("--num_inference_steps", type=int, default=5)
    p.add_argument("--cfg_scale", type=float, default=1.0)
    p.add_argument("--sigma_shift", type=float, default=5.0)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def load_episode_lerobot(dataset_path, episode_idx, total_frames, height, width, device):
    from diffsynth.trainers.utils import LiberoRGBDDataset
    ds = LiberoRGBDDataset(
        dataset_name=dataset_path,
        context_frames=5,
        prediction_frames=8,
        image_size=(height, width),
        action_dim=7,
        split="validation",
    )
    assert episode_idx < len(ds.episodes), (
        f"episode_idx={episode_idx} out of range ({len(ds.episodes)} episodes in validation split)"
    )
    ep_start, ep_end, ep_len = ds.episodes[episode_idx]
    T = min(ep_len, total_frames)
    print(f"[data] Episode {episode_idx}: {ep_len} frames, using first {T}")
    gt_frames, actions = [], []
    for i in range(ep_start, ep_start + T):
        item = ds.dataset[i]
        gt_frames.append(ds._load_rgb(item))
        actions.append(ds._load_action(item).numpy())
    actions = np.stack(actions).astype(np.float32)
    actions[0] = 0.0
    actions[0, -1] = -1.0
    action_full = torch.from_numpy(actions).to(dtype=torch.bfloat16, device=device)
    return gt_frames, action_full


def load_episode(data_path, total_frames, device):
    rgb = np.load(os.path.join(data_path, "rgb.npy"))
    if rgb.ndim == 5:
        rgb = rgb[:, 0]
    if rgb.shape[1] == 3:
        rgb = rgb.transpose(0, 2, 3, 1)

    gt_frames = []
    for frame in rgb[:total_frames]:
        if frame.max() <= 1.0:
            frame = (frame * 255).clip(0, 255)
        gt_frames.append(Image.fromarray(frame.astype(np.uint8)))

    actions = np.load(os.path.join(data_path, "actions.npy"))
    if actions.ndim == 3:
        actions = actions[:, 0]
    action_full = torch.from_numpy(actions[:total_frames]).float()
    action_full[0] = 0
    action_full[0, -1] = -1
    action_full = action_full.to(dtype=torch.bfloat16, device=device)

    return gt_frames, action_full


def build_pipeline(model_ckpt, vae_ckpt, device):
    ckpt_path = _filter_dual_branch_ckpt(model_ckpt)
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(path=ckpt_path, offload_device="cpu"),
            ModelConfig(path=vae_ckpt, offload_device="cpu"),
        ],
    )
    pipe.dit.to(device)
    pipe.vae.to(device)
    return pipe


def run_gt_guided_prediction(pipe, gt_frames, action_full, args):
    """
    GT-guided (teacher-forcing) prediction.
    At each chunk, input_image4 is taken from GT frames, not model outputs.
    This isolates per-chunk quality from compounding errors.
    """
    total_frames = args.total_frames
    num_iters = (total_frames - 1) // (WINDOW - 1)

    generated_frames = []
    input_image = gt_frames[0]

    for i in range(num_iters):
        start = i * (WINDOW - 1)
        end = start + WINDOW
        if start == 0:
            idx = np.array([0] * 5 + list(range(1, WINDOW)))
        else:
            idx = np.array([0] + list(range(start - 3, end)))

        action = action_full[idx]

        # GT-guided: always use GT frames as the 4-frame context
        if start == 0:
            input_image4 = [gt_frames[0]] * 4
        else:
            # Context frames must match action idx = range(start-3, start+1)
            # i.e. GT frames at [start-3, start-2, start-1, start]
            ctx_gt = [gt_frames[max(0, start - 3 + k)] for k in range(4)]
            input_image4 = ctx_gt

        out_video = pipe(
            seed=args.seed,
            tiled=False,
            input_image=input_image,
            input_image4=input_image4,
            action=action,
            height=args.height,
            width=args.width,
            num_frames=WINDOW + 4,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg_scale,
            sigma_shift=args.sigma_shift,
            bs_1=True,
        )[0]

        if i == 0:
            generated_frames.extend([out_video[0]] + out_video[-8:])
        else:
            generated_frames.extend(out_video[5:])

    return generated_frames[:total_frames]


def main():
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    assert args.dataset is not None or args.data_path is not None, \
        "Provide either --dataset (LeRobot) or --data_path (npy)"

    print("Loading episode data...")
    if args.dataset is not None:
        gt_frames, action_full = load_episode_lerobot(
            args.dataset, args.episode_idx, args.total_frames,
            args.height, args.width, args.device,
        )
        args.total_frames = len(gt_frames)
    else:
        gt_frames, action_full = load_episode(args.data_path, args.total_frames, args.device)

    print("Building pipeline...")
    pipe = build_pipeline(args.model_ckpt, args.vae_ckpt, args.device)

    print(f"Running GT-guided prediction ({args.total_frames} frames)...")
    pred_frames = run_gt_guided_prediction(pipe, gt_frames, action_full, args)

    pred_path = os.path.join(args.output_path, "pred_gt_guided.mp4")
    save_video(pred_frames, pred_path, fps=30, quality=5)
    print(f"Saved predicted video: {pred_path}")

    gt_path = os.path.join(args.output_path, "gt.mp4")
    save_video(gt_frames, gt_path, fps=30, quality=5)
    print(f"Saved GT video: {gt_path}")

    def frames_to_npy(frames):
        return np.stack([np.array(f) for f in frames])

    np.save(os.path.join(args.output_path, "pred_gt_guided.npy"), frames_to_npy(pred_frames))
    np.save(os.path.join(args.output_path, "gt.npy"), frames_to_npy(gt_frames))

    with open(os.path.join(args.output_path, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print("Done.")

    del pipe
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
