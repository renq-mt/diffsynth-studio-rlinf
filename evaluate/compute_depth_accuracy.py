import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    from .common import iter_video_pairs, metric_output_path, save_json, summarize_values
except ImportError:
    from common import iter_video_pairs, metric_output_path, save_json, summarize_values


DEFAULT_DEPTH_MODEL = "/cephfs/qian.ren/huggingface_download/models/depth-anything/Depth-Anything-V2-Large-hf"
DEPTH_ACCURACY_ABSREL_MIN = 0.2228
DEPTH_ACCURACY_ABSREL_MAX = 4.3711


def depth_accuracy_score_from_absrel(
    absrel: float,
    *,
    vmin: float = DEPTH_ACCURACY_ABSREL_MIN,
    vmax: float = DEPTH_ACCURACY_ABSREL_MAX,
) -> float:
    """Normalize AbsRel to the WorldArena S_Depth score where higher is better."""
    if vmax <= vmin:
        return float(absrel)
    normalized_error = (float(absrel) - vmin) / (vmax - vmin)
    normalized_error = max(0.0, min(1.0, normalized_error))
    return 1.0 - normalized_error


def _read_video_tensor(
    path: Path,
    *,
    skip_first: int = 0,
    stride: int = 1,
    max_frames: Optional[int] = None,
) -> torch.Tensor:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    frames: List[np.ndarray] = []
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index >= skip_first and (index - skip_first) % max(1, stride) == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                if max_frames is not None and len(frames) >= max_frames:
                    break
            index += 1
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"No frames read from video: {path}")

    array = np.stack(frames, axis=0).astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()


class DepthEstimator:
    def __init__(self, model_path: Path, device: torch.device):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForDepthEstimation.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        ).to(device)
        self.model.eval()

    @torch.no_grad()
    def infer(self, images: torch.Tensor, *, batch_size: int = 8) -> torch.Tensor:
        depth_chunks = []
        _, _, height, width = images.shape

        for start in range(0, len(images), max(1, batch_size)):
            batch = images[start:start + batch_size]
            images_np = (batch.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
            inputs = self.processor(images=list(images_np), return_tensors="pt").to(self.device)

            target_dtype = next(self.model.parameters()).dtype
            for key, value in inputs.items():
                if torch.is_floating_point(value):
                    inputs[key] = value.to(target_dtype)

            use_amp = self.device.type == "cuda"
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                outputs = self.model(**inputs)
                predicted_depth = outputs.predicted_depth

            depth_map = F.interpolate(
                predicted_depth.unsqueeze(1),
                size=(height, width),
                mode="bicubic",
                align_corners=False,
            )
            depth_chunks.append(depth_map.float().cpu())

        return torch.cat(depth_chunks, dim=0)


def compute_depth_accuracy_video(
    depth_model: DepthEstimator,
    pred_video: Path,
    true_video: Path,
    *,
    device: torch.device,
    skip_first: int = 0,
    stride: int = 1,
    max_frames: Optional[int] = None,
    infer_batch_size: int = 8,
    eps: float = 1e-6,
) -> Dict[str, float]:
    pred_images = _read_video_tensor(
        pred_video,
        skip_first=skip_first,
        stride=stride,
        max_frames=max_frames,
    )
    true_images = _read_video_tensor(
        true_video,
        skip_first=skip_first,
        stride=stride,
        max_frames=max_frames,
    )

    n = min(len(pred_images), len(true_images))
    pred_images = pred_images[:n]
    true_images = true_images[:n]

    pred_depth = depth_model.infer(pred_images, batch_size=infer_batch_size)
    true_depth = depth_model.infer(true_images, batch_size=infer_batch_size)
    if pred_depth.shape != true_depth.shape:
        pred_depth = F.interpolate(
            pred_depth,
            size=(true_depth.shape[2], true_depth.shape[3]),
            mode="bicubic",
            align_corners=False,
        )

    true_median = torch.median(true_depth)
    pred_median = torch.median(pred_depth)
    if pred_median > eps:
        pred_depth = pred_depth * (true_median / pred_median)

    error_map = torch.abs(pred_depth - true_depth) / (true_depth + eps)
    valid_mask = true_depth > 1e-3
    absrel = error_map[valid_mask].mean() if valid_mask.any() else error_map.mean()

    absrel_value = float(absrel.item())
    return {
        "frames": int(n),
        "depth_accuracy_absrel": absrel_value,
        "depth_accuracy_score": depth_accuracy_score_from_absrel(absrel_value),
    }


def compute_depth_accuracy_for_folders(
    pred_root: Path,
    true_root: Optional[Path] = None,
    *,
    model_path: Path = Path(DEFAULT_DEPTH_MODEL),
    device: str = "cuda:0",
    skip_first: int = 0,
    stride: int = 1,
    max_frames: Optional[int] = None,
    infer_batch_size: int = 8,
) -> Dict:
    pairs = iter_video_pairs(pred_root, true_root, pred_name="gen.mp4", true_name="gt.mp4")
    if not pairs:
        raise FileNotFoundError(f"No gen.mp4/gt.mp4 video pairs found under {pred_root}")

    torch_device = torch.device(device if torch.cuda.is_available() or not device.startswith("cuda") else "cpu")
    depth_model = DepthEstimator(model_path, torch_device)

    per_sample = []
    for sample_id, pred, true in tqdm(pairs, desc="depth_accuracy"):
        metrics = compute_depth_accuracy_video(
            depth_model,
            pred,
            true,
            device=torch_device,
            skip_first=skip_first,
            stride=stride,
            max_frames=max_frames,
            infer_batch_size=infer_batch_size,
        )
        metrics.update({"sample_id": sample_id, "pred_video": str(pred), "true_video": str(true)})
        per_sample.append(metrics)

    return {
        "metric": "depth_accuracy",
        "num_samples": len(per_sample),
        "model_path": str(model_path),
        "device": str(torch_device),
        "skip_first": skip_first,
        "stride": stride,
        "max_frames": max_frames,
        "score_formula": (
            "depth_accuracy_score = 1 - clip("
            f"(depth_accuracy_absrel - {DEPTH_ACCURACY_ABSREL_MIN}) / "
            f"({DEPTH_ACCURACY_ABSREL_MAX} - {DEPTH_ACCURACY_ABSREL_MIN}), 0, 1)"
        ),
        "score_bounds": {
            "depth_accuracy_absrel_min": DEPTH_ACCURACY_ABSREL_MIN,
            "depth_accuracy_absrel_max": DEPTH_ACCURACY_ABSREL_MAX,
        },
        **summarize_values([row["depth_accuracy_absrel"] for row in per_sample], "depth_accuracy_absrel"),
        **summarize_values([row["depth_accuracy_score"] for row in per_sample], "depth_accuracy_score"),
        "per_sample": sorted(per_sample, key=lambda row: row["sample_id"]),
    }


def compute_depth_accuracy(pred_folder, true_folder=None, **kwargs):
    result = compute_depth_accuracy_for_folders(
        Path(pred_folder),
        Path(true_folder) if true_folder else None,
        **kwargs,
    )
    return result["mean_depth_accuracy_score"] or 0.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate depth for gen.mp4 and gt.mp4, then compute scale-aligned AbsRel and S_Depth score."
    )
    parser.add_argument("--pred_root", required=True, help="Inference output folder containing episode */gen.mp4 and */gt.mp4.")
    parser.add_argument("--true_root", default=None, help="Optional separate GT video root.")
    parser.add_argument("--output_dir", default=None, help="Where to write depth_accuracy.json.")
    parser.add_argument("--model_path", default=DEFAULT_DEPTH_MODEL, help="Local Depth-Anything HF model path.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip_first", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=None, help="Optional cap for debugging; default uses every frame.")
    parser.add_argument("--infer_batch_size", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    pred_root = Path(args.pred_root)
    result = compute_depth_accuracy_for_folders(
        pred_root,
        Path(args.true_root) if args.true_root else None,
        model_path=Path(args.model_path),
        device=args.device,
        skip_first=args.skip_first,
        stride=args.stride,
        max_frames=args.max_frames,
        infer_batch_size=args.infer_batch_size,
    )
    output_dir = Path(args.output_dir) if args.output_dir else pred_root / "metrics"
    path = save_json(result, metric_output_path(output_dir, "depth_accuracy"))
    print(
        f"depth_accuracy_absrel={result['mean_depth_accuracy_absrel']} "
        f"depth_accuracy_score={result['mean_depth_accuracy_score']} -> {path}"
    )


if __name__ == "__main__":
    main()
