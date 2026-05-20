import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Optional, Sequence

import cv2
import torch
from tqdm import tqdm

try:
    from .common import ensure_dir, iter_video_pairs, metric_output_path, save_json
except ImportError:
    from common import ensure_dir, iter_video_pairs, metric_output_path, save_json


STYLEGANV_FVD_METRICS = {
    "fvd2048_16f",
    "fvd2048_128f",
    "fvd2048_128f_subsample8f",
}


def resolve_styleganv_repo(styleganv_repo: Optional[Path] = None) -> Path:
    candidates = []
    if styleganv_repo is not None:
        candidates.append(Path(styleganv_repo))
    if os.environ.get("STYLEGANV_REPO"):
        candidates.append(Path(os.environ["STYLEGANV_REPO"]))
    candidates.extend([
        Path("third_party/stylegan-v"),
        Path("/private/tmp/stylegan-v"),
    ])

    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if (candidate / "src" / "metrics" / "frechet_video_distance.py").exists():
            return candidate
    raise FileNotFoundError(
        "StyleGAN-V repo not found. Pass --styleganv_repo /path/to/stylegan-v "
        "or set STYLEGANV_REPO."
    )


def export_styleganv_frame_dataset(
    videos: Sequence[Path],
    out_root: Path,
    *,
    max_frames_per_video: Optional[int] = None,
    stride: int = 1,
    skip_first: int = 0,
    overwrite: bool = True,
) -> Path:
    if not overwrite and out_root.exists() and any(p.is_dir() for p in out_root.iterdir()):
        return out_root
    if overwrite and out_root.exists():
        shutil.rmtree(out_root)
    ensure_dir(out_root)

    for video_i, video_path in enumerate(tqdm(videos, desc=f"extract_{out_root.name}")):
        sample_dir = ensure_dir(out_root / f"{video_i:06d}")
        cap = cv2.VideoCapture(str(video_path))
        frame_i = 0
        saved = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_i >= skip_first and (frame_i - skip_first) % max(1, stride) == 0:
                    cv2.imwrite(str(sample_dir / f"{saved:06d}.png"), frame)
                    saved += 1
                    if max_frames_per_video is not None and saved >= max_frames_per_video:
                        break
                frame_i += 1
        finally:
            cap.release()
        if saved == 0:
            raise RuntimeError(f"No frames exported from {video_path}")
    return out_root


def _import_styleganv(styleganv_repo: Path):
    styleganv_repo = styleganv_repo.resolve()
    sys.path.insert(0, str(styleganv_repo))
    sys.path.insert(0, str(styleganv_repo / "src"))
    try:
        from omegaconf import OmegaConf
        from src import dnnlib
        from metrics import metric_main, metric_utils
    except ImportError as exc:
        raise ImportError(
            "StyleGAN-V dependencies are missing. Install the dependencies for the "
            "stylegan-v checkout used by --styleganv_repo."
        ) from exc
    return OmegaConf, dnnlib, metric_main, metric_utils


def _metric_value(result_dict, requested_metrics: str):
    if not isinstance(result_dict, dict):
        return None
    metric_names = [m for m in requested_metrics.split(",") if m]
    for name in metric_names:
        if name in result_dict:
            return result_dict[name]
    for key, value in result_dict.items():
        if key.startswith("fvd") and isinstance(value, (int, float)):
            return value
    numeric_values = [v for v in result_dict.values() if isinstance(v, (int, float))]
    return numeric_values[0] if numeric_values else None


def compute_fvd_styleganv_i3d(
    pred_root: Path,
    true_root: Optional[Path] = None,
    *,
    styleganv_repo: Optional[Path] = None,
    i3d_detector: Optional[Path] = None,
    work_dir: Optional[Path] = None,
    metrics: str = "fvd2048_16f",
    device: str = "cuda",
    resolution: int = 256,
    max_frames_per_video: Optional[int] = None,
    stride: int = 1,
    skip_first: int = 0,
    use_cache: bool = True,
    reuse_extracted: bool = False,
) -> dict:
    metric_names = [m for m in metrics.split(",") if m]
    invalid = [m for m in metric_names if m not in STYLEGANV_FVD_METRICS]
    if invalid:
        raise ValueError(f"Unsupported StyleGAN-V FVD metrics: {invalid}")
    if resolution not in {128, 256, 512, 1024}:
        raise ValueError("StyleGAN-V FVD resolution must be one of 128, 256, 512, 1024.")

    pred_root = Path(pred_root)
    true_root = Path(true_root) if true_root else None
    pairs = iter_video_pairs(pred_root, true_root)
    if not pairs:
        raise FileNotFoundError(f"No video pairs found under {pred_root}")

    styleganv_repo = resolve_styleganv_repo(styleganv_repo)
    work_dir = Path(work_dir) if work_dir else pred_root / "metrics" / "styleganv_frames"
    fake_path = export_styleganv_frame_dataset(
        [pred for _, pred, _ in pairs],
        work_dir / "fake",
        max_frames_per_video=max_frames_per_video,
        stride=stride,
        skip_first=skip_first,
        overwrite=not reuse_extracted,
    )
    real_path = export_styleganv_frame_dataset(
        [true for _, _, true in pairs],
        work_dir / "real",
        max_frames_per_video=max_frames_per_video,
        stride=stride,
        skip_first=skip_first,
        overwrite=not reuse_extracted,
    )

    OmegaConf, dnnlib, metric_main, metric_utils = _import_styleganv(styleganv_repo)
    if i3d_detector is not None:
        i3d_detector = Path(i3d_detector).expanduser().resolve()
        if not i3d_detector.exists():
            raise FileNotFoundError(f"I3D detector not found: {i3d_detector}")
        original_get_feature_detector = metric_utils.get_feature_detector

        def get_feature_detector(url, *args, **kwargs):
            if "i3d_torchscript" in str(url) or "ge9e5ujwgetktms" in str(url):
                url = str(i3d_detector)
            return original_get_feature_detector(url=url, *args, **kwargs)

        metric_utils.get_feature_detector = get_feature_detector

    cfg = OmegaConf.create({"max_num_frames": max_frames_per_video or 10000})
    real_dataset_kwargs = dnnlib.EasyDict(
        class_name="training.dataset.VideoFramesFolderDataset",
        path=str(real_path),
        cfg=cfg,
        xflip=False,
        resolution=resolution,
        use_labels=False,
    )
    fake_dataset_kwargs = dnnlib.EasyDict(
        class_name="training.dataset.VideoFramesFolderDataset",
        path=str(fake_path),
        cfg=cfg,
        xflip=False,
        resolution=resolution,
        use_labels=False,
    )

    result_dict = {}
    for metric_name in metric_names:
        progress = metric_utils.ProgressMonitor(verbose=True)
        result = metric_main.calc_metric(
            metric=metric_name,
            dataset_kwargs=real_dataset_kwargs,
            gen_dataset_kwargs=fake_dataset_kwargs,
            generator_as_dataset=True,
            num_gpus=1,
            rank=0,
            device=torch.device(device),
            progress=progress,
            cache=use_cache,
            num_runs=1,
        )
        current = dict(result.results) if hasattr(result, "results") else result
        result_dict.update(current)

    return {
        "metric": "fvd",
        "fvd": _metric_value(result_dict, metrics),
        "backend": "stylegan-v/i3d_torchscript",
        "styleganv_repo": str(styleganv_repo),
        "i3d_detector": str(i3d_detector) if i3d_detector else None,
        "styleganv_metrics": metrics,
        "num_samples": len(pairs),
        "fake_dataset": str(fake_path),
        "real_dataset": str(real_path),
        "resolution": resolution,
        "skip_first": skip_first,
        "stride": stride,
        "max_frames_per_video": max_frames_per_video,
        "result": result_dict,
    }


def compute_fvd(pred_folder, true_folder=None, **kwargs):
    return compute_fvd_styleganv_i3d(Path(pred_folder), Path(true_folder) if true_folder else None, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute StyleGAN-V/I3D FVD for generated videos.")
    parser.add_argument("--pred_root", required=True)
    parser.add_argument("--true_root", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--styleganv_repo", default=None)
    parser.add_argument("--i3d_detector", default=None)
    parser.add_argument("--styleganv_work_dir", default=None)
    parser.add_argument("--styleganv_metrics", default="fvd2048_16f")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--skip_first", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_frames_per_video", type=int, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--reuse_extracted", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    pred_root = Path(args.pred_root)
    result = compute_fvd_styleganv_i3d(
        pred_root,
        Path(args.true_root) if args.true_root else None,
        styleganv_repo=Path(args.styleganv_repo) if args.styleganv_repo else None,
        i3d_detector=Path(args.i3d_detector) if args.i3d_detector else None,
        work_dir=Path(args.styleganv_work_dir) if args.styleganv_work_dir else None,
        metrics=args.styleganv_metrics,
        device=args.device,
        resolution=args.resolution,
        skip_first=args.skip_first,
        stride=args.stride,
        max_frames_per_video=args.max_frames_per_video,
        use_cache=not args.no_cache,
        reuse_extracted=args.reuse_extracted,
    )
    output_dir = Path(args.output_dir) if args.output_dir else pred_root / "metrics"
    path = save_json(result, metric_output_path(output_dir, "fvd"))
    print(f"FVD={result['fvd']} -> {path}")


if __name__ == "__main__":
    main()
