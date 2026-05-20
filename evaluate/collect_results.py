import argparse
from pathlib import Path
from typing import Dict, List

try:
    from .common import load_json, save_json, write_csv
except ImportError:
    from common import load_json, save_json, write_csv


SUMMARY_KEYS = {
    "psnr_ssim": ["mean_psnr", "std_psnr", "mean_ssim", "std_ssim"],
    "latent_l2": ["mean_latent_l2", "std_latent_l2"],
    "depth": ["mean_depth_mae", "std_depth_mae", "mean_depth_rmse", "std_depth_rmse", "mean_depth_absrel"],
    "fid": ["fid"],
    "fvd": ["fvd", "backend", "styleganv_metric"],
}


def flatten_metric(metric_json: Dict) -> Dict:
    metric = metric_json.get("metric", "metric")
    row = {}
    for key in SUMMARY_KEYS.get(metric, []):
        if key in metric_json:
            row[f"{metric}.{key}"] = metric_json[key]
    for key, value in metric_json.items():
        if key.startswith("mean_") or key.startswith("std_") or key in {"fid", "fvd"}:
            row[f"{metric}.{key}"] = value
    row[f"{metric}.num_samples"] = metric_json.get("num_samples")
    return row


def collect_results(metrics_dir: Path) -> Dict:
    metric_files = sorted(p for p in metrics_dir.glob("*.json") if p.name != "summary.json")
    summary: Dict = {"metrics_dir": str(metrics_dir)}
    per_sample_by_id: Dict[str, Dict] = {}

    for path in metric_files:
        data = load_json(path)
        metric = data.get("metric", path.stem)
        summary.update(flatten_metric(data))
        for item in data.get("per_sample", []):
            sample_id = str(item.get("sample_id", "unknown"))
            row = per_sample_by_id.setdefault(sample_id, {"sample_id": sample_id})
            for key, value in item.items():
                if key not in {"sample_id", "pred_video", "true_video", "pred_latent", "true_latent"}:
                    row[f"{metric}.{key}"] = value

    per_sample: List[Dict] = [per_sample_by_id[k] for k in sorted(per_sample_by_id)]
    return {"summary": summary, "per_sample": per_sample}


def parse_args():
    parser = argparse.ArgumentParser(description="Collect metric JSON files into summary tables.")
    parser.add_argument("--metrics_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    metrics_dir = Path(args.metrics_dir)
    output_dir = Path(args.output_dir) if args.output_dir else metrics_dir
    result = collect_results(metrics_dir)
    summary_path = save_json(result["summary"], output_dir / "summary.json")
    per_sample_path = write_csv(result["per_sample"], output_dir / "per_sample.csv")
    print(f"summary -> {summary_path}")
    print(f"per_sample -> {per_sample_path}")


if __name__ == "__main__":
    main()
