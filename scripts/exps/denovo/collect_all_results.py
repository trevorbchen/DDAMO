"""
Collect all formal experiment results into a single CSV.

Reads metrics.json + config.yaml from each run directory.
For runs without metrics.json (budget_curves), computes metrics from samples.csv.

Output: outputs/results/all_results.csv
"""

import json, yaml, csv, sys
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import QED

ROOT = Path(__file__).parent / "outputs" / "results"
OUT = ROOT / "all_results.csv"


def compute_metrics_from_samples(samples_path):
    """Fallback: compute validity/uniqueness/QED from samples.csv."""
    smiles_list = []
    with open(samples_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles_list.append(row["smiles"])

    n = len(smiles_list)
    if n == 0:
        return {}

    valid_mols = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            valid_mols.append(mol)

    validity = len(valid_mols) / n
    unique_smiles = set(Chem.MolToSmiles(m) for m in valid_mols)
    uniqueness = len(unique_smiles) / len(valid_mols) if valid_mols else 0.0

    qed_scores = [QED.qed(m) for m in valid_mols]
    qed_scores.sort(reverse=True)

    return {
        "num_samples": n,
        "validity": validity,
        "uniqueness": uniqueness,
        "qed_mean": sum(qed_scores) / len(qed_scores) if qed_scores else 0.0,
        "qed_top10": sum(qed_scores[:max(1, len(qed_scores)//10)]) / max(1, len(qed_scores)//10) if qed_scores else 0.0,
        "qed_max": qed_scores[0] if qed_scores else 0.0,
    }


def parse_run(run_dir, experiment):
    """Extract all fields from a single run directory."""
    row = {"experiment": experiment, "run_name": run_dir.name}

    # ── metrics ──
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
    else:
        samples_path = run_dir / "samples.csv"
        if samples_path.exists():
            metrics = compute_metrics_from_samples(samples_path)
        else:
            return None

    for k in ["elapsed_sec", "budget_per_sample", "total_reward_evals",
              "forward_passes", "fp_per_sample",
              "validity", "uniqueness",
              "qed_mean", "qed_top10", "qed_max",
              "num_samples", "reward"]:
        row[k] = metrics.get(k, "")

    # ── config (hyperparameters) ──
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        # sampler type
        target = cfg.get("sampler", {}).get("_target_", "")
        if "beam_search" in target:
            row["sampler"] = "beam_search"
        elif "mcts" in target:
            row["sampler"] = "mcts"
        elif "Sampler" in target:
            row["sampler"] = "unconditional"
        else:
            row["sampler"] = target

        # beam search params
        sampler_cfg = cfg.get("sampler", {})
        row["beam_width"] = sampler_cfg.get("beam_width", "")
        row["branching_factor"] = sampler_cfg.get("branching_factor", "")
        row["steps_per_interval"] = sampler_cfg.get("steps_per_interval", "")
        row["elite_buffer_size"] = sampler_cfg.get("elite_buffer_size", "")
        row["diversity_penalty"] = sampler_cfg.get("diversity_penalty", "")
        row["diversity_cutoff"] = sampler_cfg.get("diversity_cutoff", "")

        # mcts params
        row["c_uct"] = sampler_cfg.get("c_uct", "")
        row["rollout_budget_per_sample"] = sampler_cfg.get("rollout_budget_per_sample", "")

        # global params
        row["softmax_temp"] = cfg.get("softmax_temp", "")
        row["randomness"] = cfg.get("randomness", "")
        row["min_add_len"] = cfg.get("min_add_len", "")
        row["seed"] = cfg.get("seed", "")

        # reward
        reward_cfg = cfg.get("reward", {})
        row["reward_type"] = reward_cfg.get("type", "")
        row["reward_alpha"] = reward_cfg.get("alpha", "")

    return row


# ── Collect all runs ──
all_rows = []

# HP sweep
hp_dir = ROOT / "hp_sweep"
if hp_dir.exists():
    for d in sorted(hp_dir.iterdir()):
        if d.is_dir():
            row = parse_run(d, "hp_sweep")
            if row:
                all_rows.append(row)

# Budget curves
bc_dir = ROOT / "budget_curves"
if bc_dir.exists():
    for d in sorted(bc_dir.iterdir()):
        if d.is_dir():
            row = parse_run(d, "budget_curve")
            if row:
                all_rows.append(row)

print(f"Collected {len(all_rows)} runs")

# ── Write CSV ──
fieldnames = [
    # identity
    "experiment", "run_name", "sampler",
    # metrics
    "elapsed_sec", "budget_per_sample", "total_reward_evals",
    "forward_passes", "fp_per_sample",
    "validity", "uniqueness",
    "qed_mean", "qed_top10", "qed_max", "num_samples",
    # beam search HP
    "beam_width", "branching_factor", "steps_per_interval",
    "elite_buffer_size", "diversity_penalty", "diversity_cutoff",
    # mcts HP
    "c_uct", "rollout_budget_per_sample",
    # global HP
    "softmax_temp", "randomness", "min_add_len", "seed",
    # reward
    "reward_type", "reward_alpha",
]

with open(OUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(all_rows)

print(f"Written to {OUT}")
