import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as TF
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm

try:
    from .common import iter_video_pairs, metric_output_path, save_json, video_length
except ImportError:
    from common import iter_video_pairs, metric_output_path, save_json, video_length


FrameRef = Tuple[Path, int]


class VideoFrameDataset(torch.utils.data.Dataset):
    def __init__(self, frame_refs: Sequence[FrameRef]):
        self.frame_refs = list(frame_refs)
        self.to_tensor = TF.ToTensor()

    def __len__(self):
        return len(self.frame_refs)

    def __getitem__(self, index):
        video_path, frame_index = self.frame_refs[index]
        cap = cv2.VideoCapture(str(video_path))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot read frame {frame_index} from {video_path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return self.to_tensor(frame)
        finally:
            cap.release()


def collect_frame_refs(
    videos: Sequence[Path],
    *,
    skip_first: int = 1,
    stride: int = 1,
    max_frames_per_video: Optional[int] = None,
) -> List[FrameRef]:
    refs: List[FrameRef] = []
    for path in videos:
        n = video_length(path)
        indices = list(range(skip_first, n, max(1, stride)))
        if max_frames_per_video is not None:
            indices = indices[:max_frames_per_video]
        refs.extend((path, idx) for idx in indices)
    return refs


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6) -> float:
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"FID covariance sqrt has imaginary component {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))


@torch.no_grad()
def get_activations(
    frame_refs: Sequence[FrameRef],
    model,
    *,
    batch_size: int = 50,
    dims: int = 2048,
    device: str = "cuda",
    num_workers: int = 4,
) -> np.ndarray:
    if len(frame_refs) < 2:
        raise ValueError("FID needs at least two frames per distribution.")
    dataset = VideoFrameDataset(frame_refs)
    batch_size = min(batch_size, len(dataset))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    pred_arr = np.empty((len(dataset), dims), dtype=np.float64)
    start = 0
    model.eval()
    for batch in tqdm(loader, desc="fid_activations"):
        batch = batch.to(device)
        pred = model(batch)[0]
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        pred_arr[start:start + pred.shape[0]] = pred
        start += pred.shape[0]
    return pred_arr


def activation_statistics(frame_refs: Sequence[FrameRef], model, **kwargs):
    activations = get_activations(frame_refs, model, **kwargs)
    return np.mean(activations, axis=0), np.cov(activations, rowvar=False)


def compute_fid_for_videos(
    pred_root: Path,
    true_root: Optional[Path] = None,
    *,
    device: str = "cuda",
    batch_size: int = 50,
    dims: int = 2048,
    num_workers: int = 4,
    skip_first: int = 1,
    stride: int = 1,
    max_frames_per_video: Optional[int] = None,
) -> dict:
    try:
        from pytorch_fid.inception import InceptionV3
    except ImportError as exc:
        raise ImportError("compute_fid.py requires pytorch-fid: pip install pytorch-fid") from exc

    pairs = iter_video_pairs(pred_root, true_root)
    pred_videos = [pred for _, pred, _ in pairs]
    true_videos = [true for _, _, true in pairs]
    pred_refs = collect_frame_refs(
        pred_videos,
        skip_first=skip_first,
        stride=stride,
        max_frames_per_video=max_frames_per_video,
    )
    true_refs = collect_frame_refs(
        true_videos,
        skip_first=skip_first,
        stride=stride,
        max_frames_per_video=max_frames_per_video,
    )

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    mu_pred, sigma_pred = activation_statistics(
        pred_refs,
        model,
        batch_size=batch_size,
        dims=dims,
        device=device,
        num_workers=num_workers,
    )
    mu_true, sigma_true = activation_statistics(
        true_refs,
        model,
        batch_size=batch_size,
        dims=dims,
        device=device,
        num_workers=num_workers,
    )
    fid = calculate_frechet_distance(mu_pred, sigma_pred, mu_true, sigma_true)
    return {
        "metric": "fid",
        "fid": fid,
        "num_samples": len(pairs),
        "num_pred_frames": len(pred_refs),
        "num_true_frames": len(true_refs),
        "dims": dims,
        "skip_first": skip_first,
        "stride": stride,
        "max_frames_per_video": max_frames_per_video,
        "backend": "pytorch_fid.InceptionV3",
    }


def compute_fid(pred_folder, true_folder=None, **kwargs):
    return compute_fid_for_videos(Path(pred_folder), Path(true_folder) if true_folder else None, **kwargs)["fid"]


def parse_args():
    parser = argparse.ArgumentParser(description="Compute FID for video frames.")
    parser.add_argument("--pred_root", required=True)
    parser.add_argument("--true_root", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--dims", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--skip_first", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_frames_per_video", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    pred_root = Path(args.pred_root)
    result = compute_fid_for_videos(
        pred_root,
        Path(args.true_root) if args.true_root else None,
        device=args.device,
        batch_size=args.batch_size,
        dims=args.dims,
        num_workers=args.num_workers,
        skip_first=args.skip_first,
        stride=args.stride,
        max_frames_per_video=args.max_frames_per_video,
    )
    output_dir = Path(args.output_dir) if args.output_dir else pred_root / "metrics"
    path = save_json(result, metric_output_path(output_dir, "fid"))
    print(f"FID={result['fid']} -> {path}")


if __name__ == "__main__":
    main()
