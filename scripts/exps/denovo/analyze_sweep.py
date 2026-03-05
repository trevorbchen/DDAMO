#!/usr/bin/env python3
"""
Compare beam search vs MCTS across budget and wall-time axes.

For each axis (budget / wall_time), groups all runs by method, finds the
best-HP config at each level, and prints a side-by-side table.

Usage:
    python3 compare_sweep.py [--base outputs/sweep/none]
"""

import argparse
import json
import os
import sys
from glob import glob


def load_metrics(base_dir):
    """Return list of metric dicts from all metrics.json under base_dir."""
    rows = []
    for path in glob(os.path.join(base_dir, "**", "metrics.json"), recursive=True):
        with open(path) as f:
            m = json.load(f)
        m["_path"] = path
        rows.append(m)
    return rows


def classify_method(name):
    if name.startswith("beam"):
        return "beam"
    if name.startswith("mcts"):
        return "mcts"
    return "other"


def bucket(value, edges):
    """Return the nearest bucket edge for a value."""
    return min(edges, key=lambda e: abs(e - value))


def best_per_bucket(rows, value_key, bucket_edges, metric="qed_mean"):
    """
    For each method × bucket, find the row with highest `metric`.
    Returns dict: {method: {bucket: best_row}}
    """
    from collections import defaultdict
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        method = classify_method(r.get("name", ""))
        if method == "other":
            continue
        b = bucket(r[value_key], bucket_edges)
        groups[method][b].append(r)

    result = {}
    for method, buckets in groups.items():
        result[method] = {}
        for b, candidates in buckets.items():
            result[method][b] = max(candidates, key=lambda x: x[metric])
    return result


def print_table(best, bucket_edges, value_label, metric="qed_mean"):
    methods = sorted(best.keys())
    header = f"{'':>10}" + "".join(f"  {m:>22}" for m in methods)
    print(header)
    print("-" * len(header))

    for b in sorted(bucket_edges):
        row = f"{value_label}={b:>6}"
        for m in methods:
            r = best.get(m, {}).get(b)
            if r:
                row += f"  {metric}={r[metric]:.4f} ({r['name']})"
            else:
                row += f"  {'—':>22}"
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="outputs/sweep/none")
    parser.add_argument("--metric", default="qed_mean",
                        choices=["qed_mean", "qed_top10", "qed_max"])
    args = parser.parse_args()

    rows = load_metrics(args.base)
    if not rows:
        print(f"No metrics.json found under {args.base}")
        sys.exit(1)

    print(f"Loaded {len(rows)} runs from {args.base}")
    print(f"Metric: {args.metric}\n")

    # ── Forward pass axis (primary compute unit) ──────────────────────────
    fp_edges = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    fp_rows = [r for r in rows if r.get("fp_per_sample", 0) > 0]
    best_fp = best_per_bucket(fp_rows, "fp_per_sample", fp_edges, args.metric)
    print("=" * 70)
    print("  Best HP per METHOD × FORWARD PASSES / sample")
    print("=" * 70)
    print_table(best_fp, fp_edges, "fp/smp", args.metric)

    # ── Rollout budget axis ────────────────────────────────────────────────
    budget_edges = [5, 10, 20, 40, 80, 160, 200, 400]
    budget_rows = [r for r in rows if r.get("budget_per_sample", 0) > 0]
    best_budget = best_per_bucket(budget_rows, "budget_per_sample",
                                  budget_edges, args.metric)
    print()
    print("=" * 70)
    print("  Best HP per METHOD × ROLLOUT BUDGET (reward evals / sample)")
    print("=" * 70)
    print_table(best_budget, budget_edges, "budget", args.metric)

    # ── Wall-time axis ─────────────────────────────────────────────────────
    time_edges = [5, 10, 20, 40, 80, 120, 200, 300]
    best_time = best_per_bucket(rows, "elapsed_sec", time_edges, args.metric)
    print()
    print("=" * 70)
    print("  Best HP per METHOD × WALL TIME (seconds)")
    print("=" * 70)
    print_table(best_time, time_edges, "time  ", args.metric)

    # ── Raw summary table ──────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  All runs (sorted by method, budget)")
    print("=" * 70)
    header = f"{'name':<30}  {'budget':>7}  {'time(s)':>7}  {args.metric:>9}"
    print(header)
    print("-" * len(header))
    for r in sorted(rows, key=lambda x: (classify_method(x.get("name","")),
                                          x.get("budget_per_sample", 0))):
        method = classify_method(r.get("name", ""))
        if method == "other":
            continue
        print(f"{r['name']:<30}  {r.get('budget_per_sample',0):>7.1f}"
              f"  {r['elapsed_sec']:>7.1f}  {r[args.metric]:>9.4f}")


if __name__ == "__main__":
    main()
