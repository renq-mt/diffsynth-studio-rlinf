import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def ensure_dir(path: os.PathLike) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: os.PathLike) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: Dict, path: os.PathLike) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    return path


def write_csv(rows: Sequence[Dict], path: os.PathLike) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        with open(path, "w") as f:
            f.write("")
        return path
    keys = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return path


def metric_output_path(output_dir: os.PathLike, metric_name: str) -> Path:
    return Path(output_dir) / f"{metric_name}.json"


def video_length(path: os.PathLike) -> int:
    cap = cv2.VideoCapture(str(path))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()


def read_video_frames(
    path: os.PathLike,
    *,
    max_frames: Optional[int] = None,
    stride: int = 1,
    resize: Optional[Tuple[int, int]] = None,
    rgb: bool = True,
    grayscale: bool = False,
    skip_first: int = 0,
) -> List[np.ndarray]:
    path = Path(path)
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
                if resize is not None:
                    frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
                if grayscale:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                elif rgb:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                if max_frames is not None and len(frames) >= max_frames:
                    break
            index += 1
    finally:
        cap.release()
    return frames


def iter_video_pairs(
    pred_root: os.PathLike,
    true_root: Optional[os.PathLike] = None,
    *,
    pred_name: str = "gen.mp4",
    true_name: str = "gt.mp4",
) -> List[Tuple[str, Path, Path]]:
    """Return (sample_id, pred_video, true_video) pairs.

    Supported layouts:
    1. inference output with manifest.json and episode dirs.
    2. episode dirs containing gen.mp4 / gt.mp4.
    3. two flat folders with matching video filenames.
    4. pred folder with per-sample subdirs and true videos in a flat folder.
    """
    pred_root = Path(pred_root)
    true_root_path = Path(true_root) if true_root else None
    manifest_path = pred_root / "manifest.json"
    pairs: List[Tuple[str, Path, Path]] = []

    if manifest_path.exists():
        manifest = load_json(manifest_path)
        pred_manifest_key = "depth_pred_video" if pred_name == "depth_pred.mp4" else "gen_video"
        true_manifest_key = "depth_gt_video" if true_name == "depth_gt.mp4" else "gt_video"
        for row in manifest.get("episodes", []):
            sample_id = str(row.get("episode", row.get("id", len(pairs))))
            pred_path = row.get(pred_manifest_key) or row.get("pred_video")
            true_path = row.get(true_manifest_key) or row.get("true_video")
            if pred_path and true_path:
                pred = Path(pred_path)
                true = Path(true_path)
                if not pred.is_absolute():
                    pred = pred_root / pred
                if not true.is_absolute():
                    true = pred_root / true
                if pred.exists() and true.exists():
                    pairs.append((sample_id, pred, true))
        if pairs:
            return pairs

    if true_root_path is None:
        for ep_dir in sorted(p for p in pred_root.iterdir() if p.is_dir()):
            pred = ep_dir / pred_name
            true = ep_dir / true_name
            if pred.exists() and true.exists():
                pairs.append((ep_dir.name, pred, true))
        return pairs

    pred_files = {p.name: p for p in pred_root.iterdir() if p.suffix.lower() in VIDEO_EXTS}
    for name, pred in sorted(pred_files.items()):
        true = true_root_path / name
        if not true.exists():
            nested = true_root_path / pred.stem / true_name
            true = nested if nested.exists() else true
        if true.exists():
            pairs.append((pred.stem, pred, true))

    if pairs:
        return pairs

    for ep_dir in sorted(p for p in pred_root.iterdir() if p.is_dir()):
        pred = ep_dir / pred_name
        true = true_root_path / f"{ep_dir.name}{Path(true_name).suffix}"
        if not true.exists():
            true = true_root_path / ep_dir.name / true_name
        if pred.exists() and true.exists():
            pairs.append((ep_dir.name, pred, true))
    return pairs


def iter_depth_pairs(pred_root: os.PathLike, true_root: Optional[os.PathLike] = None):
    return iter_video_pairs(
        pred_root,
        true_root,
        pred_name="depth_pred.mp4",
        true_name="depth_gt.mp4",
    )


def summarize_values(values: Sequence[float], prefix: str) -> Dict[str, Optional[float]]:
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return {
            f"mean_{prefix}": None,
            f"std_{prefix}": None,
            f"num_{prefix}": 0,
        }
    return {
        f"mean_{prefix}": float(arr.mean()),
        f"std_{prefix}": float(arr.std()),
        f"num_{prefix}": int(arr.size),
    }
