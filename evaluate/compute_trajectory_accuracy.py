import argparse
import importlib.util
import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

try:
    from .common import iter_video_pairs, load_json, metric_output_path, save_json, summarize_values
except ImportError:
    from common import iter_video_pairs, load_json, metric_output_path, save_json, summarize_values


DEFAULT_SAM3_MODEL_PATH = "/cephfs/qian.ren/huggingface_download/models/facebook/sam3"
DEFAULT_WORLDARENA_VIDEO_QUALITY = "/Users/mt-mac/Project/WorldArena/video_quality"
TRAJECTORY_NDTW_NORMALIZATION_MAX = 40.8540


def _sanitize_sample_id(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("_") or "sample"


def _resolve_worldarena_root(path: Optional[str]) -> Path:
    candidates = [
        path,
        os.environ.get("WORLDARENA_VIDEO_QUALITY"),
        DEFAULT_WORLDARENA_VIDEO_QUALITY,
    ]
    for candidate in candidates:
        if not candidate:
            continue
        root = Path(candidate).expanduser()
        if (root / "processing" / "detection_tracking.py").exists() and (
            root / "WorldArena" / "trajectory_accuracy.py"
        ).exists():
            return root
    raise FileNotFoundError(
        "Cannot find WorldArena video_quality root. Pass --worldarena_video_quality "
        "or set WORLDARENA_VIDEO_QUALITY."
    )


def _import_worldarena_modules(video_quality_root: Path):
    sys.path.insert(0, str(video_quality_root / "processing" / "sam3"))
    sys.path.insert(0, str(video_quality_root / "processing"))
    sys.path.insert(0, str(video_quality_root))

    from detection_tracking import nms_boxes, process_video_with_tracking
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    trajectory_path = video_quality_root / "WorldArena" / "trajectory_accuracy.py"
    spec = importlib.util.spec_from_file_location("worldarena_trajectory_accuracy", trajectory_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load trajectory_accuracy.py from {trajectory_path}")
    trajectory_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trajectory_module)
    eval_traj = trajectory_module.eval_traj

    return build_sam3_image_model, Sam3Processor, nms_boxes, process_video_with_tracking, eval_traj


def _check_sam3_model_path(model_path: Path) -> None:
    if not (model_path / "sam3.pt").exists():
        raise FileNotFoundError(
            f"SAM3 model path is missing sam3.pt: {model_path}. "
            "Pass --sam3_model_path or set SAM3_MODEL_PATH."
        )


class GripperDetector:
    """SAM3 gripper detector with optional external BPE tokenizer path."""

    def __init__(self, model_path: Path, build_sam3_image_model, Sam3Processor, nms_boxes):
        checkpoint_path = model_path / "sam3.pt"
        bpe_path = model_path / "bpe_simple_vocab_16e6.txt.gz"
        self.nms_boxes = nms_boxes

        print("Loading SAM3 model...")
        kwargs = {"checkpoint_path": str(checkpoint_path)}
        if bpe_path.exists():
            kwargs["bpe_path"] = str(bpe_path)
        self.model = build_sam3_image_model(**kwargs)
        self.processor = Sam3Processor(self.model)
        print("SAM3 model loaded")

    def detect_frame_single_prompt(self, image, prompt: str = "end effector"):
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        from PIL import Image
        import torch

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        try:
            inference_state = self.processor.set_image(image)
            inference_state = self.processor.set_text_prompt(state=inference_state, prompt=prompt)
            boxes = inference_state["boxes"]
            scores = inference_state["scores"]
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.detach().cpu().numpy()
            if isinstance(scores, torch.Tensor):
                scores = scores.detach().cpu().numpy()
            return boxes, scores
        except Exception as exc:
            print(f"Detection failed: {exc}")
            return [], []

    def detect_frame(self, image, prompts=None):
        prompts = prompts or ["robot arm", "end effector", "gripper"]
        all_boxes = []
        all_scores = []
        for prompt in prompts:
            boxes, scores = self.detect_frame_single_prompt(image, prompt)
            if len(boxes) > 0:
                all_boxes.extend(boxes)
                all_scores.extend(scores)
        if all_boxes:
            return self.nms_boxes(all_boxes, all_scores, iou_threshold=0.3)
        return [], []


def _extract_video_frames(
    video_path: Path,
    output_dir: Path,
    *,
    force: bool = False,
    resize: Tuple[int, int] = (640, 480),
) -> int:
    done_marker = output_dir / ".source"
    expected_source = f"{video_path.resolve()}\n{video_path.stat().st_mtime_ns}\n"
    if not force and done_marker.exists() and done_marker.read_text() == expected_source:
        frame_count = len(list(output_dir.glob("frame_*.jpg")))
        if frame_count > 0:
            return frame_count

    output_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in output_dir.glob("frame_*.jpg"):
        old_frame.unlink()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frame_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if resize is not None:
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(output_dir / f"frame_{frame_count:05d}.jpg"), frame)
            frame_count += 1
    finally:
        cap.release()

    done_marker.write_text(expected_source)
    if frame_count == 0:
        raise ValueError(f"No frames extracted from video: {video_path}")
    return frame_count


def _run_tracking(
    *,
    frames_dir: Path,
    output_dir: Path,
    detector,
    process_video_with_tracking,
    force: bool,
) -> Path:
    ok = process_video_with_tracking(
        str(frames_dir),
        str(output_dir),
        detector=detector,
        data_type="gt",
        saving_fps=12,
        force_reprocess=force,
    )
    if not ok:
        raise RuntimeError(f"SAM3 tracking failed for {frames_dir}")

    traj_path = output_dir / "traj" / "traj.npy"
    if not traj_path.exists():
        raise FileNotFoundError(f"Expected trajectory file was not created: {traj_path}")
    return traj_path


def _trajectory_stats(traj_path: Path) -> Dict[str, int]:
    traj = np.load(traj_path)
    return {
        "frames": int(traj.shape[0]),
        "valid_left": int(np.sum(np.all(traj[:, 0] != -1, axis=1))),
        "valid_right": int(np.sum(np.all(traj[:, 1] != -1, axis=1))),
    }


def compute_trajectory_accuracy_for_folders(
    pred_root: Path,
    true_root: Optional[Path] = None,
    *,
    sam3_model_path: Path,
    worldarena_video_quality: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    work_dir: Optional[Path] = None,
    force_reprocess: bool = False,
    max_samples: Optional[int] = None,
    num_shards: int = 1,
    shard_index: int = 0,
) -> Dict:
    video_quality_root = _resolve_worldarena_root(str(worldarena_video_quality) if worldarena_video_quality else None)

    pairs = iter_video_pairs(pred_root, true_root)
    if num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards")
    if num_shards > 1:
        pairs = pairs[shard_index::num_shards]
    if max_samples is not None:
        pairs = pairs[:max_samples]
    if not pairs:
        raise ValueError(f"No gen/gt video pairs found under {pred_root}")

    output_dir = output_dir or pred_root / "metrics"
    work_dir = work_dir or output_dir / "trajectory_accuracy_work"
    frame_root = work_dir / "frames"
    track_root = work_dir / "tracks"
    frame_root.mkdir(parents=True, exist_ok=True)
    track_root.mkdir(parents=True, exist_ok=True)

    _check_sam3_model_path(sam3_model_path)
    build_sam3_image_model, Sam3Processor, nms_boxes, process_video_with_tracking, eval_traj = _import_worldarena_modules(
        video_quality_root
    )
    detector = GripperDetector(sam3_model_path, build_sam3_image_model, Sam3Processor, nms_boxes)
    per_sample = []

    for sample_id, pred_video, true_video in tqdm(pairs, desc="trajectory_accuracy"):
        safe_id = _sanitize_sample_id(sample_id)
        sample_frame_root = frame_root / safe_id
        sample_track_root = track_root / safe_id

        pred_frames = sample_frame_root / "pred"
        true_frames = sample_frame_root / "gt"
        pred_count = _extract_video_frames(pred_video, pred_frames, force=force_reprocess)
        true_count = _extract_video_frames(true_video, true_frames, force=force_reprocess)

        true_traj = _run_tracking(
            frames_dir=true_frames,
            output_dir=sample_track_root / "gt",
            detector=detector,
            process_video_with_tracking=process_video_with_tracking,
            force=force_reprocess,
        )
        pred_traj = _run_tracking(
            frames_dir=pred_frames,
            output_dir=sample_track_root / "pred",
            detector=detector,
            process_video_with_tracking=process_video_with_tracking,
            force=force_reprocess,
        )

        metrics = eval_traj(str(pred_traj), str(true_traj))
        ndtw = float(metrics["ndtw"])
        row = {
            "sample_id": sample_id,
            "pred_video": str(pred_video),
            "true_video": str(true_video),
            "pred_traj": str(pred_traj),
            "true_traj": str(true_traj),
            "input_pred_frames": pred_count,
            "input_true_frames": true_count,
            "ndtw": ndtw,
            "ndtw_normalized": max(0.0, min(1.0, ndtw / TRAJECTORY_NDTW_NORMALIZATION_MAX)),
        }
        row.update({f"pred_{k}": v for k, v in _trajectory_stats(pred_traj).items()})
        row.update({f"true_{k}": v for k, v in _trajectory_stats(true_traj).items()})
        per_sample.append(row)

    ndtw_summary = summarize_values([row["ndtw"] for row in per_sample], "ndtw")
    ndtw_norm_summary = summarize_values([row["ndtw_normalized"] for row in per_sample], "ndtw_normalized")
    return {
        "metric": "trajectory_accuracy",
        "num_samples": len(per_sample),
        "sam3_model_path": str(sam3_model_path),
        "shard_index": shard_index,
        "num_shards": num_shards,
        "worldarena_video_quality": str(video_quality_root),
        "work_dir": str(work_dir),
        **ndtw_summary,
        **ndtw_norm_summary,
        "per_sample": sorted(per_sample, key=lambda row: row["sample_id"]),
    }


def merge_shard_results(shard_jsons) -> Dict:
    merged_samples = []
    metadata = {}
    for shard_json in shard_jsons:
        data = load_json(shard_json)
        merged_samples.extend(data.get("per_sample", []))
        for key in ("sam3_model_path", "worldarena_video_quality", "work_dir"):
            if key in data and key not in metadata:
                metadata[key] = data[key]

    merged_samples = sorted(merged_samples, key=lambda row: row.get("sample_id", ""))
    ndtw_summary = summarize_values([row["ndtw"] for row in merged_samples], "ndtw")
    ndtw_norm_summary = summarize_values([row["ndtw_normalized"] for row in merged_samples], "ndtw_normalized")
    return {
        "metric": "trajectory_accuracy",
        "num_samples": len(merged_samples),
        **metadata,
        **ndtw_summary,
        **ndtw_norm_summary,
        "per_sample": merged_samples,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Compute SAM3 gripper-trajectory accuracy for generated videos.")
    parser.add_argument("--pred_root", required=True, help="Run folder containing episode dirs with gen.mp4/gt.mp4.")
    parser.add_argument("--true_root", default=None, help="Optional separate ground-truth video root.")
    parser.add_argument("--output_dir", default=None, help="Where to write trajectory_accuracy.json.")
    parser.add_argument("--work_dir", default=None, help="Where to cache extracted frames and SAM3 trajectories.")
    parser.add_argument("--output_json", default=None, help="Exact JSON path to write. Defaults to output_dir/trajectory_accuracy.json.")
    parser.add_argument(
        "--sam3_model_path",
        default=os.environ.get("SAM3_MODEL_PATH", DEFAULT_SAM3_MODEL_PATH),
        help="Folder containing sam3.pt. Uses bpe_simple_vocab_16e6.txt.gz from this folder when present.",
    )
    parser.add_argument(
        "--worldarena_video_quality",
        default=os.environ.get("WORLDARENA_VIDEO_QUALITY", DEFAULT_WORLDARENA_VIDEO_QUALITY),
        help="Path to WorldArena/video_quality.",
    )
    parser.add_argument("--force_reprocess", action="store_true", help="Re-extract frames and rerun SAM3 tracking.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional limit for quick checks.")
    parser.add_argument("--num_shards", type=int, default=1, help="Number of data shards for multi-process/multi-GPU runs.")
    parser.add_argument("--shard_index", type=int, default=0, help="This process shard index in [0, num_shards).")
    parser.add_argument("--merge_shards", nargs="*", default=None, help="Merge existing shard JSONs and exit.")
    return parser.parse_args()


def main():
    args = parse_args()
    pred_root = Path(args.pred_root)
    output_dir = Path(args.output_dir) if args.output_dir else pred_root / "metrics"

    if args.merge_shards is not None:
        result = merge_shard_results([Path(p) for p in args.merge_shards])
        path = save_json(result, Path(args.output_json) if args.output_json else metric_output_path(output_dir, "trajectory_accuracy"))
        print(
            f"merged trajectory_accuracy ndtw={result['mean_ndtw']} "
            f"normalized={result['mean_ndtw_normalized']} -> {path}"
        )
        return

    result = compute_trajectory_accuracy_for_folders(
        pred_root,
        Path(args.true_root) if args.true_root else None,
        sam3_model_path=Path(args.sam3_model_path),
        worldarena_video_quality=Path(args.worldarena_video_quality) if args.worldarena_video_quality else None,
        output_dir=output_dir,
        work_dir=Path(args.work_dir) if args.work_dir else None,
        force_reprocess=args.force_reprocess,
        max_samples=args.max_samples,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    path = save_json(result, Path(args.output_json) if args.output_json else metric_output_path(output_dir, "trajectory_accuracy"))
    print(
        f"trajectory_accuracy ndtw={result['mean_ndtw']} "
        f"normalized={result['mean_ndtw_normalized']} -> {path}"
    )


if __name__ == "__main__":
    main()
