import argparse
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except ImportError:
    skimage_ssim = None

try:
    from .common import iter_video_pairs, metric_output_path, read_video_frames, save_json, summarize_values
except ImportError:
    from common import iter_video_pairs, metric_output_path, read_video_frames, save_json, summarize_values


def _resize_like(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    if pred.shape[:2] == target.shape[:2]:
        return pred
    return cv2.resize(pred, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_AREA)


def peak_signal_noise_ratio(true: np.ndarray, pred: np.ndarray, data_range: float = 255.0) -> float:
    mse = np.mean((true.astype(np.float64) - pred.astype(np.float64)) ** 2)
    if mse < 1e-12:
        return 100.0
    return float(20.0 * np.log10(data_range / np.sqrt(mse)))


def _ssim_channel(true: np.ndarray, pred: np.ndarray, data_range: float = 255.0) -> float:
    true = true.astype(np.float64)
    pred = pred.astype(np.float64)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    kernel = (11, 11)
    sigma = 1.5

    mu_true = cv2.GaussianBlur(true, kernel, sigma)
    mu_pred = cv2.GaussianBlur(pred, kernel, sigma)
    mu_true_sq = mu_true * mu_true
    mu_pred_sq = mu_pred * mu_pred
    mu_true_pred = mu_true * mu_pred

    sigma_true_sq = cv2.GaussianBlur(true * true, kernel, sigma) - mu_true_sq
    sigma_pred_sq = cv2.GaussianBlur(pred * pred, kernel, sigma) - mu_pred_sq
    sigma_true_pred = cv2.GaussianBlur(true * pred, kernel, sigma) - mu_true_pred

    ssim_map = ((2 * mu_true_pred + c1) * (2 * sigma_true_pred + c2)) / (
        (mu_true_sq + mu_pred_sq + c1) * (sigma_true_sq + sigma_pred_sq + c2)
    )
    return float(ssim_map.mean())


def structural_similarity(true: np.ndarray, pred: np.ndarray, *, data_range: float = 255.0, channel_axis=None) -> float:
    if skimage_ssim is not None:
        return float(skimage_ssim(true, pred, data_range=data_range, channel_axis=channel_axis))
    if channel_axis is None:
        return _ssim_channel(true, pred, data_range=data_range)
    channels = np.moveaxis(true, channel_axis, -1).shape[-1]
    true_last = np.moveaxis(true, channel_axis, -1)
    pred_last = np.moveaxis(pred, channel_axis, -1)
    return float(np.mean([
        _ssim_channel(true_last[..., c], pred_last[..., c], data_range=data_range)
        for c in range(channels)
    ]))


def compute_video_psnr_ssim(
    pred_video: Path,
    true_video: Path,
    *,
    skip_first: int = 1,
    max_frames: Optional[int] = None,
    stride: int = 1,
    grayscale: bool = False,
) -> Dict[str, float]:
    pred_frames = read_video_frames(
        pred_video,
        skip_first=skip_first,
        max_frames=max_frames,
        stride=stride,
        grayscale=grayscale,
    )
    true_frames = read_video_frames(
        true_video,
        skip_first=skip_first,
        max_frames=max_frames,
        stride=stride,
        grayscale=grayscale,
    )
    n = min(len(pred_frames), len(true_frames))
    psnr_values: List[float] = []
    ssim_values: List[float] = []

    for pred, true in zip(pred_frames[:n], true_frames[:n]):
        pred = _resize_like(pred, true)
        psnr_values.append(float(peak_signal_noise_ratio(true, pred, data_range=255)))
        if grayscale:
            ssim_values.append(float(structural_similarity(true, pred, data_range=255)))
        else:
            ssim_values.append(float(structural_similarity(true, pred, channel_axis=2, data_range=255)))

    return {
        "frames": n,
        "psnr": float(np.mean(psnr_values)) if psnr_values else 0.0,
        "ssim": float(np.mean(ssim_values)) if ssim_values else 0.0,
    }


def compute_psnr_ssim_for_folders(
    pred_root: Path,
    true_root: Optional[Path] = None,
    *,
    skip_first: int = 1,
    max_frames: Optional[int] = None,
    stride: int = 1,
    grayscale: bool = False,
    workers: int = 8,
) -> Dict:
    pairs = iter_video_pairs(pred_root, true_root)
    per_sample = []

    def _job(pair: Tuple[str, Path, Path]):
        sample_id, pred, true = pair
        metrics = compute_video_psnr_ssim(
            pred,
            true,
            skip_first=skip_first,
            max_frames=max_frames,
            stride=stride,
            grayscale=grayscale,
        )
        metrics.update({"sample_id": sample_id, "pred_video": str(pred), "true_video": str(true)})
        return metrics

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(_job, pair) for pair in pairs]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="psnr_ssim"):
            per_sample.append(future.result())

    psnr_summary = summarize_values([row["psnr"] for row in per_sample], "psnr")
    ssim_summary = summarize_values([row["ssim"] for row in per_sample], "ssim")
    return {
        "metric": "psnr_ssim",
        "num_samples": len(per_sample),
        **psnr_summary,
        **ssim_summary,
        "per_sample": sorted(per_sample, key=lambda row: row["sample_id"]),
    }


def compute_psnr(pred_folder, true_folder=None):
    result = compute_psnr_ssim_for_folders(Path(pred_folder), Path(true_folder) if true_folder else None)
    return result["mean_psnr"] or 0.0


def compute_psnr_ssim(pred_folder, true_folder=None):
    result = compute_psnr_ssim_for_folders(Path(pred_folder), Path(true_folder) if true_folder else None)
    return result["mean_psnr"] or 0.0, result["mean_ssim"] or 0.0


def parse_args():
    parser = argparse.ArgumentParser(description="Compute PSNR/SSIM for generated videos.")
    parser.add_argument("--pred_root", required=True, help="Inference output folder or predicted video folder.")
    parser.add_argument("--true_root", default=None, help="Optional ground-truth video folder.")
    parser.add_argument("--output_dir", default=None, help="Where to write psnr_ssim.json.")
    parser.add_argument("--skip_first", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--grayscale", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    pred_root = Path(args.pred_root)
    result = compute_psnr_ssim_for_folders(
        pred_root,
        Path(args.true_root) if args.true_root else None,
        skip_first=args.skip_first,
        max_frames=args.max_frames,
        stride=args.stride,
        grayscale=args.grayscale,
        workers=args.workers,
    )
    output_dir = Path(args.output_dir) if args.output_dir else pred_root / "metrics"
    path = save_json(result, metric_output_path(output_dir, "psnr_ssim"))
    print(f"PSNR={result['mean_psnr']} SSIM={result['mean_ssim']} -> {path}")


if __name__ == "__main__":
    main()
