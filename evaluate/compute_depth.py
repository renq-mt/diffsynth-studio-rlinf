import argparse
import concurrent.futures
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

try:
    from .common import iter_depth_pairs, metric_output_path, read_video_frames, save_json, summarize_values
except ImportError:
    from common import iter_depth_pairs, metric_output_path, read_video_frames, save_json, summarize_values


def _resize_like(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    if pred.shape[:2] == target.shape[:2]:
        return pred
    return cv2.resize(pred, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_AREA)


def _to_depth01(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return frame.astype(np.float32) / 255.0


def compute_depth_video_metrics(
    pred_video: Path,
    true_video: Path,
    *,
    skip_first: int = 1,
    max_frames: Optional[int] = None,
    stride: int = 1,
    eps: float = 1e-6,
) -> Dict[str, float]:
    pred_frames = read_video_frames(
        pred_video,
        skip_first=skip_first,
        max_frames=max_frames,
        stride=stride,
        grayscale=True,
        rgb=False,
    )
    true_frames = read_video_frames(
        true_video,
        skip_first=skip_first,
        max_frames=max_frames,
        stride=stride,
        grayscale=True,
        rgb=False,
    )
    n = min(len(pred_frames), len(true_frames))
    mae_values = []
    rmse_values = []
    absrel_values = []

    for pred, true in zip(pred_frames[:n], true_frames[:n]):
        pred = _resize_like(pred, true)
        pred01 = _to_depth01(pred)
        true01 = _to_depth01(true)
        err = pred01 - true01
        mae_values.append(float(np.abs(err).mean()))
        rmse_values.append(float(np.sqrt((err ** 2).mean())))
        absrel_values.append(float((np.abs(err) / np.maximum(true01, eps)).mean()))

    return {
        "frames": n,
        "depth_mae": float(np.mean(mae_values)) if mae_values else 0.0,
        "depth_rmse": float(np.mean(rmse_values)) if rmse_values else 0.0,
        "depth_absrel": float(np.mean(absrel_values)) if absrel_values else 0.0,
    }


def compute_depth_for_folders(
    pred_root: Path,
    true_root: Optional[Path] = None,
    *,
    skip_first: int = 1,
    max_frames: Optional[int] = None,
    stride: int = 1,
    workers: int = 8,
) -> Dict:
    pairs = iter_depth_pairs(pred_root, true_root)

    def _job(pair: Tuple[str, Path, Path]):
        sample_id, pred, true = pair
        metrics = compute_depth_video_metrics(
            pred,
            true,
            skip_first=skip_first,
            max_frames=max_frames,
            stride=stride,
        )
        metrics.update({"sample_id": sample_id, "pred_video": str(pred), "true_video": str(true)})
        return metrics

    per_sample = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(_job, pair) for pair in pairs]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="depth"):
            per_sample.append(future.result())

    return {
        "metric": "depth",
        "num_samples": len(per_sample),
        **summarize_values([row["depth_mae"] for row in per_sample], "depth_mae"),
        **summarize_values([row["depth_rmse"] for row in per_sample], "depth_rmse"),
        **summarize_values([row["depth_absrel"] for row in per_sample], "depth_absrel"),
        "per_sample": sorted(per_sample, key=lambda row: row["sample_id"]),
    }


def compute_depth(pred_folder, true_folder=None):
    result = compute_depth_for_folders(Path(pred_folder), Path(true_folder) if true_folder else None)
    return {
        "mae": result["mean_depth_mae"] or 0.0,
        "rmse": result["mean_depth_rmse"] or 0.0,
        "absrel": result["mean_depth_absrel"] or 0.0,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Compute depth metrics for generated videos.")
    parser.add_argument("--pred_root", required=True, help="Inference output folder or predicted depth folder.")
    parser.add_argument("--true_root", default=None, help="Optional GT depth folder.")
    parser.add_argument("--output_dir", default=None, help="Where to write depth.json.")
    parser.add_argument("--skip_first", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    pred_root = Path(args.pred_root)
    result = compute_depth_for_folders(
        pred_root,
        Path(args.true_root) if args.true_root else None,
        skip_first=args.skip_first,
        max_frames=args.max_frames,
        stride=args.stride,
        workers=args.workers,
    )
    output_dir = Path(args.output_dir) if args.output_dir else pred_root / "metrics"
    path = save_json(result, metric_output_path(output_dir, "depth"))
    print(f"depth_mae={result['mean_depth_mae']} depth_rmse={result['mean_depth_rmse']} -> {path}")


if __name__ == "__main__":
    main()
