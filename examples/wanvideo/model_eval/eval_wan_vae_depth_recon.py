"""
Evaluate Wan VAE reconstruction quality on depth sequences.

This script measures how well the Wan video VAE reconstructs depth maps after
an encode/decode round trip. It uses the same depth loading path as
LiberoRGBDDataset, where each depth frame is converted to a 3-channel grayscale
PIL image before entering the model.

Outputs:
  - metrics.json: aggregated metrics
  - samples/*.png: visualization grids (GT / recon / abs error)
  - samples/*.mp4: GT and reconstructed depth videos for a few samples

Example:
    python examples/wanvideo/model_eval/eval_wan_vae_depth_recon.py \
        --vae_ckpt /path/to/Wan2.2_VAE.pth \
        --dataset /cephfs/qian.ren/huggingface_download/datasets/Libero/libero_goal_lerobot_mask_depth \
        --output_path outputs/wan_vae_depth_eval \
        --num_samples 100 \
        --device cuda:0
"""

import argparse
import json
import math
import os
import random
import sys
import time
from typing import List

import numpy as np
import torch
from PIL import Image, ImageDraw

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except ImportError:
    skimage_ssim = None


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from diffsynth import ModelManager, save_video  # noqa: E402
from diffsynth.trainers.utils import LiberoRGBDDataset  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_ckpt", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="outputs/wan_vae_depth_eval")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--num_visualizations", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--context_frames", type=int, default=5)
    parser.add_argument("--prediction_frames", type=int, default=4)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=4)
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pil_list_to_video_tensor(frames: List[Image.Image]) -> torch.Tensor:
    arr = np.stack([np.array(frame.convert("RGB"), dtype=np.float32) for frame in frames], axis=0)
    arr = arr / 127.5 - 1.0
    return torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous()


def video_tensor_to_pil_list(video: torch.Tensor) -> List[Image.Image]:
    video = video.detach().float().clamp(-1, 1)
    video = ((video + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    video = video.permute(1, 2, 3, 0).cpu().numpy()
    return [Image.fromarray(frame, mode="RGB") for frame in video]


def pil_to_gray_npy(frames: List[Image.Image]) -> np.ndarray:
    return np.stack([np.array(frame.convert("L"), dtype=np.uint8) for frame in frames], axis=0)


def compute_metrics(gt_gray: np.ndarray, pred_gray: np.ndarray) -> dict:
    gt = gt_gray.astype(np.float32) / 255.0
    pred = pred_gray.astype(np.float32) / 255.0
    diff = pred - gt
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))
    if mse < 1e-12:
        psnr = 100.0
    else:
        psnr = float(20.0 * math.log10(1.0 / math.sqrt(mse)))

    ssim_scores = []
    if skimage_ssim is not None:
        for gt_frame, pred_frame in zip(gt_gray, pred_gray):
            ssim_scores.append(float(skimage_ssim(gt_frame, pred_frame, data_range=255)))
    ssim = float(np.mean(ssim_scores)) if ssim_scores else None

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "psnr": psnr,
        "ssim": ssim,
    }


def make_visualization(gt_frames: List[Image.Image], pred_frames: List[Image.Image], save_path: str):
    n = len(gt_frames)
    w, h = gt_frames[0].size
    canvas = Image.new("RGB", (n * w, 3 * h + 60), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), "Top: GT depth | Middle: VAE recon | Bottom: abs error x4", fill=(0, 0, 0))

    for i, (gt, pred) in enumerate(zip(gt_frames, pred_frames)):
        gt_rgb = gt.convert("RGB")
        pred_rgb = pred.convert("RGB")
        err = np.abs(
            np.array(gt.convert("L"), dtype=np.int16) - np.array(pred.convert("L"), dtype=np.int16)
        ).astype(np.uint8)
        err = np.clip(err * 4, 0, 255)
        err_rgb = Image.fromarray(err, mode="L").convert("RGB")

        x = i * w
        canvas.paste(gt_rgb, (x, 30))
        canvas.paste(pred_rgb, (x, 30 + h))
        canvas.paste(err_rgb, (x, 30 + 2 * h))

    canvas.save(save_path)


def aggregate_metrics(metrics_list: List[dict]) -> dict:
    result = {}
    metric_keys = ["mse", "mae", "rmse", "psnr", "ssim"]
    for key in metric_keys:
        values = [m[key] for m in metrics_list if m[key] is not None]
        result[key] = float(np.mean(values)) if values else None
    return result


def load_vae(vae_ckpt: str, device: str, dtype: torch.dtype):
    manager = ModelManager(torch_dtype=dtype, device=device)
    manager.load_model(vae_ckpt, device="cpu", torch_dtype=dtype)
    vae = manager.fetch_model("wan_video_vae")
    if vae is None:
        raise RuntimeError(f"Failed to load wan_video_vae from {vae_ckpt}")
    vae.to(device)
    vae.eval()
    return vae


def main():
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "samples"), exist_ok=True)

    seed_everything(args.seed)
    dtype = resolve_dtype(args.dtype)

    num_frames = args.context_frames + args.prediction_frames
    if num_frames % 4 != 1:
        raise ValueError(
            f"num_frames must satisfy num_frames % 4 == 1 for exact Wan VAE round-trip, got {num_frames}."
        )

    dataset = LiberoRGBDDataset(
        dataset_name=args.dataset,
        context_frames=args.context_frames,
        prediction_frames=args.prediction_frames,
        image_size=(args.height, args.width),
        action_dim=args.action_dim,
        split=args.split,
        repeat=args.repeat,
    )
    vae = load_vae(args.vae_ckpt, args.device, dtype)

    sample_count = min(args.num_samples, len(dataset))
    if sample_count <= 0:
        raise ValueError(f"Dataset split '{args.split}' is empty: {args.dataset}")
    vis_budget = min(args.num_visualizations, sample_count)
    sample_indices = np.random.choice(len(dataset), size=sample_count, replace=False)

    per_sample_metrics = []
    saved_visualizations = []
    encode_times = []
    decode_times = []
    total_times = []

    print(f"[eval] dataset windows={len(dataset)}, evaluating samples={sample_count}")
    print(f"[eval] depth format follows LiberoRGBDDataset: 8-bit grayscale replicated to RGB")

    with torch.no_grad():
        for rank, sample_idx in enumerate(sample_indices):
            sample = dataset[int(sample_idx)]
            depth_frames = sample["depth"]

            video = pil_list_to_video_tensor(depth_frames).unsqueeze(0).to(dtype=dtype)

            if args.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize(args.device)
            start_total = time.perf_counter()

            start_encode = time.perf_counter()
            latents = vae.encode(video, device=args.device)
            if args.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize(args.device)
            encode_time = time.perf_counter() - start_encode

            start_decode = time.perf_counter()
            recon = vae.decode(latents, device=args.device).squeeze(0)
            if args.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize(args.device)
            decode_time = time.perf_counter() - start_decode
            total_time = time.perf_counter() - start_total

            encode_times.append(float(encode_time))
            decode_times.append(float(decode_time))
            total_times.append(float(total_time))

            recon_frames = video_tensor_to_pil_list(recon)

            gt_gray = pil_to_gray_npy(depth_frames)
            recon_gray = pil_to_gray_npy(recon_frames)
            metrics = compute_metrics(gt_gray, recon_gray)
            metrics["sample_idx"] = int(sample_idx)
            metrics["encode_time_sec"] = float(encode_time)
            metrics["decode_time_sec"] = float(decode_time)
            metrics["total_time_sec"] = float(total_time)
            metrics["frames_per_sec"] = float(num_frames / total_time) if total_time > 0 else None
            per_sample_metrics.append(metrics)

            if rank < vis_budget:
                stem = f"sample_{rank:03d}_idx_{int(sample_idx):05d}"
                grid_path = os.path.join(args.output_path, "samples", f"{stem}.png")
                gt_video_path = os.path.join(args.output_path, "samples", f"{stem}_gt.mp4")
                recon_video_path = os.path.join(args.output_path, "samples", f"{stem}_recon.mp4")
                make_visualization(depth_frames, recon_frames, grid_path)
                save_video(depth_frames, gt_video_path, fps=args.fps, quality=5)
                save_video(recon_frames, recon_video_path, fps=args.fps, quality=5)
                saved_visualizations.append({
                    "sample_idx": int(sample_idx),
                    "grid_path": grid_path,
                    "gt_video_path": gt_video_path,
                    "recon_video_path": recon_video_path,
                    "psnr": metrics["psnr"],
                    "ssim": metrics["ssim"],
                    "mae": metrics["mae"],
                    "total_time_sec": metrics["total_time_sec"],
                    "frames_per_sec": metrics["frames_per_sec"],
                })

            if (rank + 1) % 10 == 0 or rank + 1 == sample_count:
                running = aggregate_metrics(per_sample_metrics)
                running_fps = float(num_frames / np.mean(total_times)) if total_times else None
                if running["ssim"] is not None:
                    print(
                        f"[eval] {rank + 1}/{sample_count} PSNR={running['psnr']:.3f} "
                        f"SSIM={running['ssim']:.4f} FPS={running_fps:.2f}"
                    )
                else:
                    print(f"[eval] {rank + 1}/{sample_count} PSNR={running['psnr']:.3f} SSIM=unavailable FPS={running_fps:.2f}")

    summary = {
        "vae_ckpt": args.vae_ckpt,
        "dataset": args.dataset,
        "split": args.split,
        "image_size": [args.height, args.width],
        "num_frames": num_frames,
        "num_samples": sample_count,
        "dtype": args.dtype,
        "device": args.device,
        "metrics_mean": aggregate_metrics(per_sample_metrics),
        "efficiency": {
            "encode_time_sec_mean": float(np.mean(encode_times)) if encode_times else None,
            "decode_time_sec_mean": float(np.mean(decode_times)) if decode_times else None,
            "total_time_sec_mean": float(np.mean(total_times)) if total_times else None,
            "frames_per_sec_mean": float(num_frames / np.mean(total_times)) if total_times else None,
            "videos_per_sec_mean": float(1.0 / np.mean(total_times)) if total_times else None,
        },
        "per_sample_metrics": per_sample_metrics,
        "samples": saved_visualizations,
    }

    metrics_path = os.path.join(args.output_path, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[eval] saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
