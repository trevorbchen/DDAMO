"""Design space comparison: controlled axis comparisons for molecule generation.

3 axes:
  - Axis 1: Inference-time sampler (beam_search, mcts, smc, dfkc, daps, uncond)
  - Axis 2: Finetune method (none = base GenMol, ddpp, future methods)
  - Axis 3: Guide reward (qed, flash_affinity, boltz — used during generation)

After generation, an oracle reward scores all molecules for the real evaluation.

Usage:
    python scripts/comparison.py --config configs/comparison/example.yaml
    python scripts/comparison.py --config configs/comparison/example.yaml --dry-run
    python scripts/comparison.py --config configs/comparison/example.yaml --skip-oracle
"""

import argparse
import itertools
import json
import os
import sys
from time import time

import pandas as pd
import yaml

sys.path.insert(0, os.path.realpath("."))
sys.path.insert(0, os.path.join(os.path.realpath("."), "src"))

from genmol.samplers import (
    Sampler, BeamSearchSampler, MCTSSampler, SMCSampler, DFKCSampler, DAPSSampler,
    load_model_from_path,
)
from genmol.rewards import get_reward
from genmol.model_loader import merge_ddpp_checkpoint
from evals.metrics import compute_metrics

SAMPLER_CLASSES = {
    "uncond": Sampler,
    "beam_search": BeamSearchSampler,
    "mcts": MCTSSampler,
    "smc": SMCSampler,
    "dfkc": DFKCSampler,
    "daps": DAPSSampler,
}


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_model(finetune_method, cfg, model_cache):
    """Load GenMol model with optional finetuning (cached). Returns model object."""
    if finetune_method in model_cache:
        return model_cache[finetune_method]

    model_path = os.path.realpath(cfg["model_path"])
    if finetune_method == "ddpp":
        ddpp_ckpt = cfg.get("ddpp_checkpoint")
        if ddpp_ckpt is None:
            raise ValueError("finetune includes 'ddpp' but ddpp_checkpoint is not set")
        model_path = merge_ddpp_checkpoint(model_path, os.path.realpath(ddpp_ckpt))

    model = load_model_from_path(model_path)
    model_cache[finetune_method] = model
    return model


def run_combo(sampler_name, finetune_method, guide_name, cfg, model_cache):
    """Run one (sampler × finetune × guide_reward) combo. Returns (samples, metrics)."""
    model = resolve_model(finetune_method, cfg, model_cache)

    # Guide reward
    if guide_name in ("none",):
        forward_op = None
    else:
        forward_op = get_reward(guide_name, **cfg.get("guide_reward_params", {}))

    # Sampler
    cls = SAMPLER_CLASSES[sampler_name]
    overrides = cfg.get("sampler_overrides", {}).get(sampler_name, {})
    sampler = cls(model=model, forward_op=forward_op, **overrides)

    # Generate
    t0 = time()
    samples = sampler.de_novo_generation(
        cfg.get("num_samples", 100),
        softmax_temp=cfg.get("softmax_temp", 1.0),
        randomness=cfg.get("randomness", 0.3),
        min_add_len=cfg.get("min_add_len", 60),
    )
    elapsed = time() - t0

    # Metrics
    m = compute_metrics(samples)
    metrics = {
        "sampler": sampler_name,
        "finetune": finetune_method,
        "guide_reward": guide_name,
        "wall_sec": round(elapsed, 2),
        "reward_calls": getattr(sampler, "last_reward_evals", 0),
        "forward_passes": getattr(sampler, "last_forward_passes", 0),
        "budget_per_sample": getattr(sampler, "last_budget_per_sample", 0),
        **{k: m[k] for k in ("validity", "uniqueness", "qed_mean", "qed_top10", "qed_max")},
        "num_samples": len(samples),
        **{f"sampler_{k}": v for k, v in overrides.items()},
    }
    return samples, metrics


def run_oracle(oracle_name, samples, oracle_params):
    """Score samples with an oracle. Returns list of scores."""
    if oracle_name == "flash_affinity":
        from evals.flash_affinity import run_flash_affinity
        return run_flash_affinity(samples, **oracle_params)
    elif oracle_name == "boltz":
        from genmol.rewards.boltz import run_boltz_affinity
        return run_boltz_affinity(samples, **oracle_params)
    else:
        raise ValueError(f"Unknown oracle: {oracle_name}")


def oracle_metrics(scores):
    """Compute summary stats from oracle scores."""
    valid = [s for s in scores if s is not None]
    if not valid:
        return {"oracle_mean": None, "oracle_top10": None, "oracle_max": None, "oracle_n_scored": 0}
    valid_sorted = sorted(valid, reverse=True)
    top10_n = max(1, len(valid) // 10)
    return {
        "oracle_mean": round(sum(valid) / len(valid), 4),
        "oracle_top10": round(sum(valid_sorted[:top10_n]) / top10_n, 4),
        "oracle_max": round(max(valid), 4),
        "oracle_n_scored": len(valid),
    }


def main():
    parser = argparse.ArgumentParser(description="Design space comparison orchestrator.")
    parser.add_argument("--config", required=True, help="Comparison config YAML.")
    parser.add_argument("--dry-run", action="store_true", help="Print grid without running.")
    parser.add_argument("--skip-oracle", action="store_true", help="Skip oracle evaluation pass.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    name = cfg["name"]
    output_base = os.path.join(cfg.get("output_dir", "outputs/comparison"), name)

    samplers = cfg.get("samplers", ["uncond"])
    finetunes = cfg.get("finetune", ["none"])
    guides = cfg.get("guide_rewards", ["none"])
    oracle_name = cfg.get("oracle")
    oracle_params = cfg.get("oracle_params", {})

    grid = list(itertools.product(samplers, finetunes, guides))
    print(f"Comparison: {name}")
    print(f"Grid: {len(grid)} combos = {len(samplers)} samplers x {len(finetunes)} finetune x {len(guides)} guides")
    print(f"Oracle: {oracle_name or 'none'}")

    if args.dry_run:
        for s, ft, g in grid:
            overrides = cfg.get("sampler_overrides", {}).get(s, {})
            print(f"  {ft}/{g}/{s}  overrides={overrides}")
        return

    # ── Generation pass ──────────────────────────────────────────────
    model_cache = {}
    all_results = []
    combo_samples = {}

    for s, ft, g in grid:
        tag = f"{ft}/{g}/{s}"
        print(f"\n{'='*60}")
        print(f"Running: {tag}")
        print(f"{'='*60}")

        try:
            samples, metrics = run_combo(s, ft, g, cfg, model_cache)
        except Exception as e:
            print(f"  FAILED: {e}")
            all_results.append({"sampler": s, "finetune": ft, "guide_reward": g, "error": str(e)})
            continue

        combo_dir = os.path.join(output_base, ft, g, s)
        os.makedirs(combo_dir, exist_ok=True)
        pd.DataFrame({"smiles": samples}).to_csv(os.path.join(combo_dir, "samples.csv"), index=False)
        with open(os.path.join(combo_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        combo_samples[(s, ft, g)] = samples
        all_results.append(metrics)

        print(f"  {metrics['wall_sec']}s  validity={metrics['validity']:.3f}  "
              f"qed={metrics['qed_mean']:.4f}  reward_calls={metrics['reward_calls']}")

    # ── Oracle pass ──────────────────────────────────────────────────
    if oracle_name and not args.skip_oracle:
        print(f"\n{'='*60}")
        print(f"Oracle evaluation: {oracle_name}")
        print(f"{'='*60}")

        for i, (s, ft, g) in enumerate(grid):
            if (s, ft, g) not in combo_samples:
                continue
            samples = combo_samples[(s, ft, g)]
            valid_smiles = [smi for smi in samples if smi]
            if not valid_smiles:
                continue

            tag = f"{ft}/{g}/{s}"
            print(f"  Scoring {tag} ({len(valid_smiles)} molecules)...")

            try:
                scores = run_oracle(oracle_name, valid_smiles, oracle_params)
                om = oracle_metrics(scores)
                all_results[i].update(om)

                combo_dir = os.path.join(output_base, ft, g, s)
                pd.DataFrame({"smiles": valid_smiles, "oracle_score": scores}).to_csv(
                    os.path.join(combo_dir, "oracle_scores.csv"), index=False)

                print(f"    mean={om['oracle_mean']}  top10={om['oracle_top10']}  max={om['oracle_max']}")
            except Exception as e:
                print(f"    FAILED: {e}")

    # ── Summary ──────────────────────────────────────────────────────
    summary = pd.DataFrame(all_results)
    summary_path = os.path.join(output_base, "summary.csv")
    os.makedirs(output_base, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    print(f"\n{'='*60}")
    print(f"Summary: {len(all_results)} combos -> {summary_path}")
    print(f"{'='*60}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
