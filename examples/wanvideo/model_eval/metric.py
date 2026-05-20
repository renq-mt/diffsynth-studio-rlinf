"""
Metric computation: FVD, PSNR, SSIM between GT and predicted videos.

Supports comparing:
  - GT vs stream prediction (free-running)
  - GT vs GT-guided prediction (teacher-forcing)
  - Both at once

FVD uses I3D features (pytorch-fvd or torch_fidelity).
PSNR and SSIM are computed per-frame then averaged.

Usage:
    # Compare a single pair
    python examples/wanvideo/model_eval/metric.py \
        --gt outputs/eval/gt.npy \
        --pred outputs/eval/stream/pred_stream.npy \
        --output outputs/eval/stream/metrics.json

    # Compare both stream and GT-guided at once
    python examples/wanvideo/model_eval/metric.py \
        --gt outputs/eval/gt.npy \
        --pred outputs/eval/stream/pred_stream.npy \
               outputs/eval/gt_guided/pred_gt_guided.npy \
        --names stream gt_guided \
        --output outputs/eval/metrics_all.json

Dependencies:
    pip install scikit-image torch torchvision
    # For FVD:
    pip install pytorch-fid  # provides inception features as fallback
    # Recommended: pip install torch-fidelity
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as ski_psnr
from skimage.metrics import structural_similarity as ski_ssim


# ---------------------------------------------------------------------------
# FVD helpers
# ---------------------------------------------------------------------------

def _load_i3d():
    """Load I3D model for FVD feature extraction."""
    try:
        from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3  # noqa
    except ImportError:
        pass

    # Try pytorch-i3d (common in video generation repos)
    try:
        import sys
        # Attempt to import a bundled or installed i3d
        from torchvision.models.video import r3d_18
        model = r3d_18(pretrained=True)
        model.fc = torch.nn.Identity()
        model.eval()
        return model, "r3d"
    except Exception:
        pass

    raise ImportError(
        "No I3D/R3D model found. Install torchvision>=0.12 for R3D-18 FVD features, "
        "or install torch-fidelity for Inception-based FVD."
    )


def _extract_video_features(model, model_type, frames_npy, device, batch_size=4):
    """
    frames_npy: [T, H, W, 3] uint8
    Returns feature array [N_clips, D].
    Clips the video into non-overlapping 16-frame clips.
    """
    T = frames_npy.shape[0]
    clip_len = 16
    clips = []
    for start in range(0, T - clip_len + 1, clip_len):
        clip = frames_npy[start : start + clip_len]  # [16, H, W, 3]
        clip = clip.astype(np.float32) / 255.0
        clip = torch.from_numpy(clip).permute(3, 0, 1, 2).unsqueeze(0)  # [1, 3, 16, H, W]
        clip = F.interpolate(clip.view(1, 3, clip_len, *clip.shape[-2:]), size=(clip_len, 112, 112), mode="trilinear", align_corners=False)
        clips.append(clip)

    if not clips:
        raise ValueError(f"Video too short for FVD (need >= {clip_len} frames, got {T})")

    features = []
    model = model.to(device)
    with torch.no_grad():
        for i in range(0, len(clips), batch_size):
            batch = torch.cat(clips[i : i + batch_size], dim=0).to(device)
            feat = model(batch)
            if isinstance(feat, (list, tuple)):
                feat = feat[0]
            features.append(feat.cpu().float().numpy())

    return np.concatenate(features, axis=0)  # [N, D]


def _frechet_distance(mu1, sigma1, mu2, sigma2):
    """Numpy implementation of Frechet distance."""
    from scipy.linalg import sqrtm
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def compute_fvd(gt_npy, pred_npy, device="cuda"):
    """
    gt_npy, pred_npy: [T, H, W, 3] uint8
    Returns scalar FVD.
    """
    try:
        model, model_type = _load_i3d()
    except ImportError as e:
        print(f"[FVD] Skipping: {e}")
        return None

    gt_feat = _extract_video_features(model, model_type, gt_npy, device)
    pred_feat = _extract_video_features(model, model_type, pred_npy, device)

    mu_gt, sigma_gt = gt_feat.mean(0), np.cov(gt_feat, rowvar=False)
    mu_pred, sigma_pred = pred_feat.mean(0), np.cov(pred_feat, rowvar=False)

    # Handle single-clip edge case (cov is scalar)
    if sigma_gt.ndim == 0:
        sigma_gt = np.array([[float(sigma_gt)]])
        sigma_pred = np.array([[float(sigma_pred)]])

    return _frechet_distance(mu_gt, sigma_gt, mu_pred, sigma_pred)


# ---------------------------------------------------------------------------
# PSNR / SSIM
# ---------------------------------------------------------------------------

def compute_psnr_ssim(gt_npy, pred_npy):
    """
    gt_npy, pred_npy: [T, H, W] uint8 or [T, H, W, C] uint8
    Returns (mean_psnr, mean_ssim, per_frame_psnr, per_frame_ssim).
    """
    assert gt_npy.shape == pred_npy.shape, (
        f"Shape mismatch: gt={gt_npy.shape}, pred={pred_npy.shape}"
    )
    T = gt_npy.shape[0]
    psnrs, ssims = [], []

    for t in range(T):
        gt_f = gt_npy[t]
        pr_f = pred_npy[t]

        psnr = ski_psnr(gt_f, pr_f, data_range=255)
        if gt_f.ndim == 2:
            ssim = ski_ssim(gt_f, pr_f, data_range=255)
        else:
            ssim = ski_ssim(gt_f, pr_f, data_range=255, channel_axis=-1)

        psnrs.append(float(psnr))
        ssims.append(float(ssim))

    return float(np.mean(psnrs)), float(np.mean(ssims)), psnrs, ssims


# ---------------------------------------------------------------------------
# LPIPS (optional)
# ---------------------------------------------------------------------------

def compute_lpips(gt_npy, pred_npy, device="cuda"):
    """Per-frame LPIPS, averaged. Returns None if lpips not installed."""
    try:
        import lpips
    except ImportError:
        print("[LPIPS] Skipping: pip install lpips")
        return None

    loss_fn = lpips.LPIPS(net="alex").to(device)
    T = gt_npy.shape[0]
    scores = []

    def to_tensor(frame):
        if frame.ndim == 2:
            frame = np.repeat(frame[..., None], 3, axis=2)
        elif frame.ndim == 3 and frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=2)
        t = torch.from_numpy(frame).float() / 127.5 - 1.0  # [-1, 1]
        return t.permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        for t in range(T):
            score = loss_fn(to_tensor(gt_npy[t]), to_tensor(pred_npy[t]))
            scores.append(float(score.item()))

    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_npy_frames(path):
    """
    Load a frame array from .npy and normalize dtype/layout.

    Supported inputs:
      - [T, H, W] grayscale
      - [T, H, W, 1] grayscale
      - [T, H, W, 3] RGB
      - [T, 1, H, W] grayscale
      - [T, 3, H, W] RGB
    """
    arr = np.load(path)
    if arr.ndim != 3 and arr.ndim != 4:
        raise ValueError(f"Expected 3D/4D frame array, got shape {arr.shape} from {path}")

    if arr.ndim == 4:
        if arr.shape[1] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (0, 2, 3, 1))
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.shape[-1] != 3:
            raise ValueError(f"Unsupported channel layout {arr.shape} from {path}")

    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255).clip(0, 255)
        arr = arr.astype(np.uint8)
    return arr


def ensure_three_channels(arr):
    """Convert grayscale frame arrays to RGB by channel replication."""
    if arr.ndim == 3:
        return np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        return np.repeat(arr, 3, axis=-1)
    if arr.ndim == 4 and arr.shape[-1] == 3:
        return arr
    raise ValueError(f"Cannot convert shape {arr.shape} to 3-channel frames")


def evaluate_pair(gt_npy, pred_npy, name, device):
    print(f"\n--- Evaluating: {name} ---")
    print(f"  GT shape:   {gt_npy.shape}")
    print(f"  Pred shape: {pred_npy.shape}")

    # Align lengths
    T = min(len(gt_npy), len(pred_npy))
    gt_npy = gt_npy[:T]
    pred_npy = pred_npy[:T]
    gt_npy_3c = ensure_three_channels(gt_npy)
    pred_npy_3c = ensure_three_channels(pred_npy)

    results = {"name": name, "num_frames": T}

    # PSNR / SSIM
    mean_psnr, mean_ssim, per_psnr, per_ssim = compute_psnr_ssim(gt_npy, pred_npy)
    results["psnr"] = mean_psnr
    results["ssim"] = mean_ssim
    results["per_frame_psnr"] = per_psnr
    results["per_frame_ssim"] = per_ssim
    print(f"  PSNR:  {mean_psnr:.4f} dB")
    print(f"  SSIM:  {mean_ssim:.4f}")

    # LPIPS
    lpips_score = compute_lpips(gt_npy_3c, pred_npy_3c, device)
    results["lpips"] = lpips_score
    if lpips_score is not None:
        print(f"  LPIPS: {lpips_score:.4f}")

    # FVD
    fvd = compute_fvd(gt_npy_3c, pred_npy_3c, device)
    results["fvd"] = fvd
    if fvd is not None:
        print(f"  FVD:   {fvd:.2f}")

    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gt", required=True, help="Path to GT frames .npy; supports grayscale or RGB")
    p.add_argument("--pred", nargs="+", required=True, help="Path(s) to predicted frames .npy")
    p.add_argument("--names", nargs="+", default=None, help="Names for each pred (default: filename)")
    p.add_argument("--output", default="metrics.json", help="Output JSON path")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()

    gt_npy = load_npy_frames(args.gt)

    names = args.names
    if names is None:
        names = [os.path.splitext(os.path.basename(p))[0] for p in args.pred]
    assert len(names) == len(args.pred), "Number of --names must match number of --pred paths"

    all_results = []
    for pred_path, name in zip(args.pred, names):
        pred_npy = load_npy_frames(pred_path)
        result = evaluate_pair(gt_npy, pred_npy, name, args.device)
        all_results.append(result)

    # Summary table
    print("\n=== Summary ===")
    header = f"{'Name':<25} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'FVD':>10}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        lpips_str = f"{r['lpips']:.4f}" if r["lpips"] is not None else "N/A"
        fvd_str = f"{r['fvd']:.2f}" if r["fvd"] is not None else "N/A"
        print(f"{r['name']:<25} {r['psnr']:>8.4f} {r['ssim']:>8.4f} {lpips_str:>8} {fvd_str:>10}")

    # Save JSON (strip per-frame arrays for readability in summary, keep in full output)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved metrics: {args.output}")


if __name__ == "__main__":
    main()
