"""
Streaming inference for the dual-branch RGB+depth WanVideo model.

This script only produces evaluation data. Metrics are computed by scripts under
evaluate/.
"""
import argparse
import json
import os
import sys
import time

try:
    import torch
    torch.serialization.add_safe_globals(["set", "OrderedDict", "builtins.set"])
except (AttributeError, ImportError):
    pass

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate long-horizon videos for evaluation.")
    parser.add_argument(
        "--rgb_ckpt",
        type=str,
        default="/workspace/qian.ren/robot_vla/diffsynth-studio-rlinf-dual-branch/outputs/dual_branch_stage2/epoch-49.safetensors.rgb_only.safetensors",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/workspace/qian.ren/robot_vla/diffsynth-studio-rlinf-dual-branch/outputs/dual_branch_stage2/epoch-49.safetensors",
    )
    parser.add_argument("--depth_only_ckpt", type=str, default=None)
    parser.add_argument("--disable_xattn", action="store_true")
    parser.add_argument(
        "--vae_ckpt",
        type=str,
        default="/workspace/qian.ren/robot_vla_cuda/gac_wm/diffsynth-studio/models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2-TI2V-5B/Wan2.2_VAE.pth",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="/cephfs/qian.ren/huggingface_download/datasets/Libero/libero_goal_lerobot_mask_depth",
        help="Path to lerobot dataset root (used when --data_source=lerobot).",
    )
    parser.add_argument(
        "--data_source",
        type=str,
        choices=["lerobot", "pickle"],
        default="lerobot",
        help="Which validation source to evaluate on. 'lerobot' uses LiberoRGBDDataset; "
             "'pickle' uses RLinfPickleDataset against rollout .pkl dumps.",
    )
    parser.add_argument(
        "--rlinf_pkl_path",
        type=str,
        nargs="+",
        default=None,
        help="One or more directories containing rlinf rollout .pkl files. "
             "Required when --data_source=pickle.",
    )
    parser.add_argument("--rlinf_pkl_only_success", action="store_true",
                        help="Pickle source only: keep success episodes.")
    parser.add_argument("--output_path", type=str, default="outputs/eval_dual_branch_stream")
    parser.add_argument("--num_episodes", type=int, default=20, help="Number of val episodes (-1 = all).")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--sigma_shift", type=float, default=5.0)
    parser.add_argument("--window", type=int, default=9)
    parser.add_argument("--context_frames", type=int, default=5)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--save_compare", action="store_true")
    return parser.parse_args()


def load_pipeline(args):
    import copy
    import safetensors.torch as st
    from diffsynth.models.wan_video_dit import CrossBranchAttention
    from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline

    print(f"[infer] Initializing RGB pipeline from: {args.rgb_ckpt}")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=[
            ModelConfig(path=args.rgb_ckpt, offload_device="cpu"),
            ModelConfig(path=args.vae_ckpt, offload_device="cpu"),
        ],
    )
    pipe.dit.to(args.device)
    pipe.vae.to(args.device)

    rgb_sd, depth_sd, xattn_sd = {}, {}, {}
    using_depth_only = args.depth_only_ckpt is not None and os.path.exists(args.depth_only_ckpt)
    dit_keys = set(pipe.dit.state_dict().keys())
    if using_depth_only:
        print(f"[infer] Loading standalone depth branch: {args.depth_only_ckpt}")
        sd = st.load_file(args.depth_only_ckpt)
        for key, value in sd.items():
            if key.startswith("pipe.dit."):
                depth_sd[key[len("pipe.dit."):]] = value
            elif key.startswith("pipe.depth_dit."):
                depth_sd[key[len("pipe.depth_dit."):]] = value
            elif key in dit_keys:
                depth_sd[key] = value
    else:
        print(f"[infer] Loading dual-branch checkpoint: {args.ckpt}")
        sd = st.load_file(args.ckpt)
        for key, value in sd.items():
            if key.startswith("pipe.dit."):
                name = key[len("pipe.dit."):]
                if name in dit_keys:
                    rgb_sd[name] = value
            elif key in dit_keys:
                rgb_sd[key] = value
            elif key.startswith("pipe.depth_dit."):
                depth_sd[key[len("pipe.depth_dit."):]] = value
            elif key.startswith("depth_dit."):
                depth_sd[key[len("depth_dit."):]] = value
            elif key.startswith("pipe.cross_branch_attentions."):
                xattn_sd[key[len("pipe.cross_branch_attentions."):]] = value
            elif key.startswith("cross_branch_attentions."):
                xattn_sd[key[len("cross_branch_attentions."):]] = value

    if rgb_sd and not using_depth_only:
        msg = pipe.dit.load_state_dict(rgb_sd, strict=False)
        print(f"[infer] RGB DiT loaded: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
    elif not using_depth_only:
        print("[infer] No RGB DiT keys found; using --rgb_ckpt initializer.")

    if depth_sd:
        print("[infer] Initializing depth branch and cross-branch attention.")
        pipe.depth_dit = copy.deepcopy(pipe.dit)
        num_blocks = len(pipe.dit.blocks)
        dim = pipe.dit.dim
        num_heads = pipe.dit.blocks[0].num_heads
        pipe.cross_branch_attentions = torch.nn.ModuleList([
            CrossBranchAttention(dim=dim, num_heads=num_heads) for _ in range(num_blocks)
        ]).to(dtype=torch.bfloat16, device=args.device)

        msg = pipe.depth_dit.load_state_dict(depth_sd, strict=False)
        print(f"[infer] depth_dit loaded: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
        if xattn_sd and not using_depth_only:
            msg = pipe.cross_branch_attentions.load_state_dict(xattn_sd, strict=False)
            print(f"[infer] xattn loaded: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")

        if args.disable_xattn or using_depth_only:
            with torch.no_grad():
                for xattn in pipe.cross_branch_attentions:
                    xattn.o_rgb.weight.zero_()
                    xattn.o_rgb.bias.zero_()
                    xattn.o_depth.weight.zero_()
                    xattn.o_depth.bias.zero_()
            print("[infer] xattn output projections zeroed.")

        pipe.depth_dit.to(args.device).eval()
        pipe.cross_branch_attentions.eval()
    else:
        print("[infer] No depth branch found; running RGB-only inference.")

    pipe.dit.eval()
    return pipe


def load_val_dataset(args):
    if args.data_source == "pickle":
        from diffsynth.trainers.utils import RLinfPickleDataset
        if not args.rlinf_pkl_path:
            raise ValueError("--rlinf_pkl_path is required when --data_source=pickle")
        pkl_paths = args.rlinf_pkl_path if len(args.rlinf_pkl_path) > 1 else args.rlinf_pkl_path[0]
        dataset = RLinfPickleDataset(
            dataset_name=pkl_paths,
            context_frames=args.context_frames,
            prediction_frames=args.window - args.context_frames,
            image_size=(args.height, args.width),
            action_dim=7,
            split="validation",
            only_success=args.rlinf_pkl_only_success,
            episode_cache_size=1,  # eval loads episodes one at a time
        )
        print(f"[infer] Pickle validation episodes: {len(dataset.episodes)}")
        return dataset

    from diffsynth.trainers.utils import LiberoRGBDDataset
    dataset = LiberoRGBDDataset(
        dataset_name=args.dataset,
        context_frames=args.context_frames,
        prediction_frames=args.window - args.context_frames,
        image_size=(args.height, args.width),
        action_dim=7,
        split="validation",
    )
    print(f"[infer] Lerobot validation episodes: {len(dataset.episodes)}")
    return dataset


def load_episode_frames(dataset, episode_idx):
    """Use the dataset's sequential loader so we get GT in natural temporal order."""
    rgb_list, actions_np, depth_pil_list = dataset.load_episode_sequential(episode_idx)
    depth_list = [
        torch.from_numpy(np.array(d.convert("L"))).float() / 255.0
        for d in depth_pil_list
    ]
    return rgb_list, actions_np, depth_list, depth_pil_list


def _depth_to_rgb(depth_img):
    gray = np.array(depth_img.convert("L"))
    return Image.fromarray(np.stack([gray, gray, gray], axis=-1))


def depth_frames_to_video(depth_pil_list):
    return [_depth_to_rgb(depth_img) for depth_img in depth_pil_list]


def gt_depth_to_video(gt_depth_list):
    frames = []
    for depth_t in gt_depth_list:
        gray = (depth_t.numpy() * 255).clip(0, 255).astype(np.uint8)
        frames.append(Image.fromarray(np.stack([gray, gray, gray], axis=-1)))
    return frames


def run_stream_inference(pipe, rgb_list, actions, args, depth_pil_list=None):
    window = args.window
    total_frames = len(actions)
    num_iters = max(1, total_frames // (window - 1))

    gen_rgb = []
    gen_depth = []

    input_image = rgb_list[0]
    input_image4 = [input_image] * 4
    depth_input_image = depth_pil_list[0] if depth_pil_list else None
    input_depth4 = [depth_pil_list[0]] * 4 if depth_pil_list else None

    actions_t = torch.from_numpy(actions).to(dtype=torch.bfloat16, device=args.device)
    actions_t[0] = 0
    actions_t[0, -1] = -1

    for i in range(num_iters):
        start = i * (window - 1)
        end = start + window
        if i == 0:
            idx = [0] * args.context_frames + list(range(1, window))
        else:
            idx = [0] + list(range(start - 3, end))
        clamped = np.clip(np.array(idx), 0, len(actions_t) - 1)
        act_win = actions_t[clamped]

        out_video = pipe(
            seed=args.seed,
            tiled=False,
            input_image=input_image,
            input_image4=input_image4,
            depth_input_image=depth_input_image,
            input_depth4=input_depth4,
            action=act_win,
            height=args.height,
            width=args.width,
            num_frames=window + 4,
            num_inference_steps=args.steps,
            cfg_scale=args.cfg_scale,
            sigma_shift=args.sigma_shift,
            bs_1=True,
        )[0]

        raw_depth = getattr(pipe, "_last_depth_frames", None)
        chunk_depth = raw_depth[0] if raw_depth and isinstance(raw_depth[0], list) else raw_depth

        if i == 0:
            gen_rgb.extend([out_video[0]] + out_video[-8:])
            if chunk_depth:
                gen_depth.extend([chunk_depth[0]] + chunk_depth[-8:])
        else:
            gen_rgb.extend(out_video[5:])
            if chunk_depth:
                gen_depth.extend(chunk_depth[5:])

        input_image4 = out_video[-4:]
        input_depth4 = chunk_depth[-4:] if chunk_depth and len(chunk_depth) >= 4 else None

    return gen_rgb[:total_frames], gen_depth[:total_frames] if gen_depth else []


def save_episode_outputs(ep_dir, gen_rgb, gt_rgb, gen_depth, gt_depth, args, save_compare=False):
    from diffsynth import save_video

    os.makedirs(ep_dir, exist_ok=True)
    n = min(len(gen_rgb), len(gt_rgb))
    save_video(gen_rgb[:n], os.path.join(ep_dir, "gen.mp4"), fps=args.fps, quality=args.quality)
    save_video(gt_rgb[:n], os.path.join(ep_dir, "gt.mp4"), fps=args.fps, quality=args.quality)

    if save_compare:
        side = [
            Image.fromarray(np.concatenate([np.array(pred), np.array(gt)], axis=1))
            for pred, gt in zip(gen_rgb[:n], gt_rgb[:n])
        ]
        save_video(side, os.path.join(ep_dir, "compare.mp4"), fps=args.fps, quality=args.quality)

    depth_paths = {}
    if gen_depth:
        nd = min(len(gen_depth), len(gt_depth), n)
        pred_depth_frames = depth_frames_to_video(gen_depth[:nd])
        gt_depth_frames = gt_depth_to_video(gt_depth[:nd])
        if pred_depth_frames:
            save_video(pred_depth_frames, os.path.join(ep_dir, "depth_pred.mp4"), fps=args.fps, quality=args.quality)
            depth_paths["depth_pred_video"] = "depth_pred.mp4"
        if gt_depth_frames:
            save_video(gt_depth_frames, os.path.join(ep_dir, "depth_gt.mp4"), fps=args.fps, quality=args.quality)
            depth_paths["depth_gt_video"] = "depth_gt.mp4"
        if save_compare and pred_depth_frames and gt_depth_frames:
            cmp_n = min(len(pred_depth_frames), len(gt_depth_frames))
            cmp_frames = [
                Image.fromarray(np.concatenate([np.array(pred), np.array(gt)], axis=1))
                for pred, gt in zip(pred_depth_frames[:cmp_n], gt_depth_frames[:cmp_n])
            ]
            save_video(cmp_frames, os.path.join(ep_dir, "depth_compare.mp4"), fps=args.fps, quality=args.quality)
    return n, depth_paths


def main():
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    pipe = load_pipeline(args)
    dataset = load_val_dataset(args)

    num_episodes = len(dataset.episodes) if args.num_episodes == -1 else min(args.num_episodes, len(dataset.episodes))
    print(f"[infer] Generating {num_episodes} episodes -> {args.output_path}")

    manifest = {
        "rgb_ckpt": args.rgb_ckpt,
        "ckpt": args.ckpt,
        "depth_only_ckpt": args.depth_only_ckpt,
        "disable_xattn": bool(args.disable_xattn or args.depth_only_ckpt),
        "dataset": args.dataset,
        "num_episodes": num_episodes,
        "steps": args.steps,
        "cfg_scale": args.cfg_scale,
        "sigma_shift": args.sigma_shift,
        "window": args.window,
        "context_frames": args.context_frames,
        "height": args.height,
        "width": args.width,
        "seed": args.seed,
        "has_depth_branch": getattr(pipe, "depth_dit", None) is not None,
        "episodes": [],
    }

    for ep_i in tqdm(range(num_episodes), desc="episodes"):
        rgb_list, actions, depth_gt, depth_gt_pil = load_episode_frames(dataset, ep_i)
        ep_len = len(actions)
        start_time = time.time()
        gen_rgb, gen_depth = run_stream_inference(pipe, rgb_list, actions, args, depth_pil_list=depth_gt_pil)
        elapsed = time.time() - start_time

        ep_name = f"episode_{ep_i:04d}"
        ep_dir = os.path.join(args.output_path, ep_name)
        frames, depth_paths = save_episode_outputs(
            ep_dir,
            gen_rgb,
            rgb_list,
            gen_depth,
            depth_gt,
            args,
            save_compare=args.save_compare,
        )
        row = {
            "episode": ep_i,
            "episode_dir": ep_name,
            "dataset_frames": ep_len,
            "frames": frames,
            "time_s": elapsed,
            "gen_video": f"{ep_name}/gen.mp4",
            "gt_video": f"{ep_name}/gt.mp4",
        }
        for key, value in depth_paths.items():
            row[key] = f"{ep_name}/{value}"
        manifest["episodes"].append(row)
        with open(os.path.join(ep_dir, "manifest.json"), "w") as f:
            json.dump(row, f, indent=2)

    manifest_path = os.path.join(args.output_path, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[infer] Done. Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
