#!/usr/bin/env python3
"""Collect Pareto-frontier configs from QED sweep → run Boltz affinity → aggregate.

Usage:
    python3 run_boltz_eval.py [--sweep-dir outputs/sweep/none] [--devices 2] [--dry-run]
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from glob import glob
from pathlib import Path

import pandas as pd

EVALS_DIR = Path(__file__).resolve().parents[3] / "evals"
sys.path.insert(0, str(EVALS_DIR))


# ── Config selection ─────────────────────────────────────────────────────────

def load_all_metrics(sweep_dir: str):
    rows = []
    for p in glob(os.path.join(sweep_dir, "**", "metrics.json"), recursive=True):
        with open(p) as f:
            m = json.load(f)
        m["_dir"] = os.path.dirname(p)
        rows.append(m)
    return rows


def select_pareto_configs(rows, top_k=2):
    """Pick top-k configs per (method × budget_bucket) + baseline."""
    budgets = [20, 50, 100, 200]
    groups = defaultdict(list)

    for r in rows:
        name = r.get("name", "")
        if name.startswith("beam"):
            method = "beam"
        elif name.startswith("mcts"):
            method = "mcts"
        else:
            continue
        b = r.get("budget_per_sample", 0)
        for B in budgets:
            if abs(b - B) <= B * 0.6:
                groups[(method, B)].append(r)
                break

    selected = []
    for key in sorted(groups):
        top = sorted(groups[key], key=lambda x: -x["qed_mean"])[:top_k]
        selected.extend(top)

    # Add baseline
    for r in rows:
        if r.get("name", "") == "uncond_baseline":
            selected.append(r)
            break

    return selected


# ── SMILES collection ────────────────────────────────────────────────────────

def collect_unique_smiles(selected):
    """Return (smiles_to_configs, all_smiles_list)."""
    smiles_to_configs = defaultdict(set)

    for r in selected:
        csv_path = os.path.join(r["_dir"], "samples.csv")
        if not os.path.exists(csv_path):
            print(f"  Warning: {csv_path} missing, skip")
            continue
        df = pd.read_csv(csv_path)
        col = "smiles" if "smiles" in df.columns else "sequence"
        for smi in df[col]:
            smiles_to_configs[smi].add(r["name"])

    return smiles_to_configs


# ── Boltz evaluation ─────────────────────────────────────────────────────────

def run_boltz(smiles_list, out_dir, devices=2):
    from boltz_affinity import run_boltz_affinity

    affinities = run_boltz_affinity(
        smiles_list,
        input_dir=os.path.join(out_dir, "boltz_inputs"),
        out_dir=os.path.join(out_dir, "boltz_outputs"),
        devices=devices,
    )
    return affinities


# ── Aggregation ──────────────────────────────────────────────────────────────

def aggregate_results(selected, smiles_to_configs, smiles_list, affinities, out_dir):
    # Per-molecule results
    mol_df = pd.DataFrame({
        "smiles": smiles_list,
        "boltz_affinity": affinities,
    })
    mol_df.to_csv(os.path.join(out_dir, "all_molecules.csv"), index=False)

    # Build smiles→affinity map
    aff_map = dict(zip(smiles_list, affinities))

    # Per-config summary
    config_results = []
    for r in selected:
        csv_path = os.path.join(r["_dir"], "samples.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        col = "smiles" if "smiles" in df.columns else "sequence"
        affs = [aff_map.get(s) for s in df[col]]
        valid_affs = [a for a in affs if a is not None]

        if valid_affs:
            import numpy as np
            arr = np.array(valid_affs)
            top10_idx = max(1, len(arr) // 10)
            sorted_arr = np.sort(arr)  # lower = better binding
            config_results.append({
                "name": r["name"],
                "method": "beam" if r["name"].startswith("beam") else
                          "mcts" if r["name"].startswith("mcts") else "other",
                "budget": r.get("budget_per_sample", 0),
                "qed_mean": r["qed_mean"],
                "n_unique": len(set(df[col])),
                "n_molecules": len(df),
                "aff_mean": float(arr.mean()),
                "aff_best": float(arr.min()),
                "aff_top10": float(sorted_arr[:top10_idx].mean()),
                "aff_median": float(np.median(arr)),
                "n_valid_aff": len(valid_affs),
            })
        else:
            config_results.append({
                "name": r["name"],
                "method": "beam" if r["name"].startswith("beam") else
                          "mcts" if r["name"].startswith("mcts") else "other",
                "budget": r.get("budget_per_sample", 0),
                "qed_mean": r["qed_mean"],
                "n_unique": len(set(df[col])),
                "n_molecules": len(df),
                "aff_mean": None, "aff_best": None,
                "aff_top10": None, "aff_median": None,
                "n_valid_aff": 0,
            })

    res_df = pd.DataFrame(config_results)
    res_df = res_df.sort_values(["method", "budget"])
    res_df.to_csv(os.path.join(out_dir, "boltz_summary.csv"), index=False)

    # Print comparison table
    print("\n" + "=" * 90)
    print("  Boltz Affinity: Best HP per METHOD × BUDGET (lower = better binding)")
    print("=" * 90)
    print(f"{'Method':<8} {'Budget':>7} {'Aff mean':>9} {'Aff best':>9} "
          f"{'Aff top10':>10} {'QED mean':>9} {'N_uniq':>6}  Config")
    print("-" * 90)
    for _, row in res_df.iterrows():
        aff_m = f"{row['aff_mean']:.3f}" if row['aff_mean'] is not None else "N/A"
        aff_b = f"{row['aff_best']:.3f}" if row['aff_best'] is not None else "N/A"
        aff_t = f"{row['aff_top10']:.3f}" if row['aff_top10'] is not None else "N/A"
        print(f"{row['method']:<8} {row['budget']:>7.1f} {aff_m:>9} {aff_b:>9} "
              f"{aff_t:>10} {row['qed_mean']:>9.4f} {row['n_unique']:>6}  {row['name']}")

    return res_df


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", default="outputs/sweep/none")
    parser.add_argument("--out-dir", default="outputs/boltz_eval")
    parser.add_argument("--devices", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=2,
                        help="Top-k configs per method×budget bucket")
    parser.add_argument("--dry-run", action="store_true",
                        help="Select configs and count molecules, don't run Boltz")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Load and select configs
    print("Loading sweep results...")
    rows = load_all_metrics(args.sweep_dir)
    print(f"  Found {len(rows)} configs")

    selected = select_pareto_configs(rows, top_k=args.top_k)
    print(f"\nSelected {len(selected)} Pareto configs:")
    for r in selected:
        print(f"  {r['name']:<30s} budget={r.get('budget_per_sample',0):>6.1f}  "
              f"qed={r['qed_mean']:.4f}")

    # 2. Collect unique SMILES
    smiles_to_configs = collect_unique_smiles(selected)
    smiles_list = list(smiles_to_configs.keys())
    print(f"\nTotal unique SMILES: {len(smiles_list)}")
    print(f"Estimated Boltz time ({args.devices} GPUs, ~40s/mol): "
          f"{len(smiles_list)*40/args.devices/3600:.1f} hours")

    # Save mapping
    mapping = [{"smiles": s, "configs": ",".join(sorted(cs))}
               for s, cs in smiles_to_configs.items()]
    pd.DataFrame(mapping).to_csv(
        os.path.join(args.out_dir, "smiles_config_map.csv"), index=False)

    if args.dry_run:
        print("\n[DRY RUN] Stopping before Boltz. Files saved to:", args.out_dir)
        return

    # 3. Run Boltz
    print(f"\nRunning Boltz affinity on {len(smiles_list)} molecules "
          f"with {args.devices} GPUs...")
    affinities = run_boltz(smiles_list, args.out_dir, devices=args.devices)

    # 4. Aggregate
    aggregate_results(selected, smiles_to_configs, smiles_list, affinities,
                      args.out_dir)
    print(f"\nResults saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
