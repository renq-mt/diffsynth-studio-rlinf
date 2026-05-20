import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluate.collect_results import collect_results


def load_json(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def find_run_dirs(output_root: Path) -> List[Path]:
    if (output_root / "manifest.json").exists() or (output_root / "metrics").exists():
        return [output_root]
    run_dirs = []
    for path in sorted(output_root.iterdir()):
        if path.is_dir() and ((path / "manifest.json").exists() or (path / "metrics").exists()):
            run_dirs.append(path)
    return run_dirs


def flatten_manifest(run_dir: Path) -> Dict:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = load_json(manifest_path)
    episodes = manifest.get("episodes", [])
    total_frames = sum(int(row.get("frames", 0)) for row in episodes)
    total_time = sum(float(row.get("time_s", 0.0)) for row in episodes)
    row = {
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "num_episodes": manifest.get("num_episodes", len(episodes)),
        "total_frames": total_frames,
        "total_time_s": total_time,
        "mean_time_s": total_time / len(episodes) if episodes else None,
        "fps_generated": total_frames / total_time if total_time > 0 else None,
        "checkpoint": manifest.get("ckpt"),
        "rgb_ckpt": manifest.get("rgb_ckpt"),
        "depth_only_ckpt": manifest.get("depth_only_ckpt"),
        "disable_xattn": manifest.get("disable_xattn"),
        "dataset": manifest.get("dataset"),
        "steps": manifest.get("steps"),
        "cfg_scale": manifest.get("cfg_scale"),
        "sigma_shift": manifest.get("sigma_shift"),
        "window": manifest.get("window"),
        "context_frames": manifest.get("context_frames"),
        "height": manifest.get("height"),
        "width": manifest.get("width"),
        "seed": manifest.get("seed"),
        "has_depth_branch": manifest.get("has_depth_branch"),
    }
    return row


def collect_run_stats(run_dir: Path) -> Dict:
    row = flatten_manifest(run_dir)
    row.setdefault("run", run_dir.name)
    row.setdefault("run_dir", str(run_dir))
    metrics_dir = run_dir / "metrics"
    if metrics_dir.exists():
        summary_path = metrics_dir / "summary.json"
        if not summary_path.exists():
            row.update(collect_results(metrics_dir)["summary"])
        else:
            row.update(load_json(summary_path))
    return row


def write_csv(rows: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Collect inference output and metric summaries into a table.")
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--table", default=None, help="Output CSV path. Defaults to <output_root>/stats.csv.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    run_dirs = find_run_dirs(output_root)
    rows = [collect_run_stats(run_dir) for run_dir in run_dirs]
    table_path = Path(args.table) if args.table else output_root / "stats.csv"
    write_csv(rows, table_path)
    print(f"runs={len(rows)} -> {table_path}")


if __name__ == "__main__":
    main()
