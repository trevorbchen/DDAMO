"""Collect all experiment results into a single CSV.

Walks outputs/ recursively, collecting every directory that contains
metrics.json.  Enriches with config.yaml hyperparameters where available.

Output: outputs/all_results.csv

Usage:
    python scripts/exps/collect_results.py [--output-dir outputs]
"""

import argparse
import json
import yaml
from pathlib import Path

import pandas as pd


def parse_run(run_dir: Path) -> dict | None:
    """Extract metrics + config from a single run directory."""
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None

    with open(metrics_path) as f:
        row = json.load(f)

    row["run_dir"] = str(run_dir)

    # Enrich with config hyperparams
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}

        # Sampler hyperparams (prefixed to avoid collisions)
        sampler_cfg = cfg.get("sampler", {})
        for k, v in sampler_cfg.items():
            if k.startswith("_") or k in ("path", "forward_op"):
                continue
            key = f"sampler_{k}" if not k.startswith("sampler_") else k
            row.setdefault(key, v)

        # Model info
        model_cfg = cfg.get("model", {})
        row.setdefault("model", model_cfg.get("type", "base"))

        # Reward info
        reward_cfg = cfg.get("reward", {})
        row.setdefault("reward", reward_cfg.get("type", ""))

        # Global generation params
        for k in ("softmax_temp", "randomness", "min_add_len", "seed"):
            if k in cfg:
                row.setdefault(k, cfg[k])

    return row


def collect(output_root: str = "outputs") -> pd.DataFrame:
    """Walk output_root recursively and collect all runs."""
    root = Path(output_root)
    rows = []

    for metrics_path in sorted(root.rglob("metrics.json")):
        row = parse_run(metrics_path.parent)
        if row is not None:
            rows.append(row)

    if not rows:
        print(f"No runs found under {root}")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Put key columns first
    key_cols = [
        "model", "sampler", "reward",
        "elapsed_sec", "budget_per_sample", "total_reward_evals",
        "forward_passes", "fp_per_sample",
        "validity", "uniqueness",
        "qed_mean", "qed_top10", "qed_max",
        "num_samples",
    ]
    present = [c for c in key_cols if c in df.columns]
    rest = [c for c in df.columns if c not in present]
    df = df[present + rest]

    out_path = root / "all_results.csv"
    df.to_csv(out_path, index=False)
    print(f"Collected {len(rows)} runs → {out_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    collect(args.output_dir)
