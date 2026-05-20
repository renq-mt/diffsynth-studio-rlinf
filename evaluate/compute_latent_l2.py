import argparse
import concurrent.futures
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    from .common import metric_output_path, save_json, summarize_values
except ImportError:
    from common import metric_output_path, save_json, summarize_values


def _load_latent(path: Path, key: Optional[str] = None):
    data = torch.load(path, map_location="cpu")
    if key:
        for part in key.split("."):
            data = data[part] if isinstance(data, dict) else getattr(data, part)
    elif isinstance(data, dict) and "obs" in data:
        data = data["obs"]
    return data.float()


def iter_latent_pairs(pred_root: Path, true_root: Path):
    pred_files = sorted([p for p in pred_root.iterdir() if p.suffix in {".pt", ".pth"}])
    pairs = []
    for pred in pred_files:
        true = true_root / pred.name
        if not true.exists():
            true = true_root / pred.stem / "0.pt"
        if true.exists():
            pairs.append((pred.stem, pred, true))
    return pairs


def compute_latent_l2_pair(
    pred_path: Path,
    true_path: Path,
    *,
    pred_key: Optional[str] = None,
    true_key: Optional[str] = None,
    skip_first: int = 1,
) -> float:
    pred = _load_latent(pred_path, pred_key)
    true = _load_latent(true_path, true_key)
    n = min(pred.shape[0], true.shape[0])
    pred = pred[skip_first:n]
    true = true[skip_first:n]
    return float(F.mse_loss(pred, true).item())


def compute_latent_l2_for_folders(
    pred_root: Path,
    true_root: Path,
    *,
    pred_key: Optional[str] = None,
    true_key: Optional[str] = None,
    skip_first: int = 1,
    workers: int = 8,
) -> Dict:
    pairs = iter_latent_pairs(pred_root, true_root)

    def _job(pair: Tuple[str, Path, Path]):
        sample_id, pred, true = pair
        value = compute_latent_l2_pair(
            pred,
            true,
            pred_key=pred_key,
            true_key=true_key,
            skip_first=skip_first,
        )
        return {"sample_id": sample_id, "latent_l2": value, "pred_latent": str(pred), "true_latent": str(true)}

    per_sample = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(_job, pair) for pair in pairs]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="latent_l2"):
            per_sample.append(future.result())

    return {
        "metric": "latent_l2",
        "num_samples": len(per_sample),
        **summarize_values([row["latent_l2"] for row in per_sample], "latent_l2"),
        "per_sample": sorted(per_sample, key=lambda row: row["sample_id"]),
    }


def compute_latent_l2(pred_folder, true_folder):
    result = compute_latent_l2_for_folders(Path(pred_folder), Path(true_folder))
    return result["mean_latent_l2"] or 0.0


def parse_args():
    parser = argparse.ArgumentParser(description="Compute latent-space L2/MSE.")
    parser.add_argument("--pred_root", required=True)
    parser.add_argument("--true_root", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--pred_key", default=None)
    parser.add_argument("--true_key", default=None)
    parser.add_argument("--skip_first", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    pred_root = Path(args.pred_root)
    result = compute_latent_l2_for_folders(
        pred_root,
        Path(args.true_root),
        pred_key=args.pred_key,
        true_key=args.true_key,
        skip_first=args.skip_first,
        workers=args.workers,
    )
    output_dir = Path(args.output_dir) if args.output_dir else pred_root / "metrics"
    path = save_json(result, metric_output_path(output_dir, "latent_l2"))
    print(f"latent_l2={result['mean_latent_l2']} -> {path}")


if __name__ == "__main__":
    main()
