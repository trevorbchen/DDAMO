"""Design space comparison with wall-clock budget control and oracle timeline logging.

Runs each (sampler x finetune x guide) combo under a fixed wall-clock time budget.
Logs oracle evaluations one-by-one so the timeline can be subsampled post-hoc to
simulate a more expensive oracle (e.g. 10x, 100x cost multiplier).

Timeline log format (oracle_timeline.jsonl per combo):
    {"call": 1,  "smiles": "...", "score": -0.4, "best_so_far": -0.4, "top10_mean": -0.4, "wall_sec": 12.3}
    {"call": 2,  "smiles": "...", "score": -0.1, "best_so_far": -0.4, "top10_mean": -0.25, "wall_sec": 24.1}
    ...

Post-hoc cost simulation:
    Subsample timeline at stride N to simulate oracle that costs N times more.
    Read off best_so_far at indices [0, N, 2N, ...] to get the budget curve.

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
import time
from collections import deque

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
    "uncond":      Sampler,
    "beam_search": BeamSearchSampler,
    "mcts":        MCTSSampler,
    "smc":         SMCSampler,
    "dfkc":        DFKCSampler,
    "daps":        DAPSSampler,
}


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ── Model loading (cached) ────────────────────────────────────────────────────

def resolve_model(finetune_method, cfg, model_cache):
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


# ── Generation (wall-clock controlled) ───────────────────────────────────────

def run_generation(sampler_name, finetune_method, guide_name, cfg, model_cache,
                   wall_budget_sec=None):
    """Generate molecules for up to wall_budget_sec seconds (or cfg.num_samples if no budget).

    Returns (samples, metrics).
    """
    model = resolve_model(finetune_method, cfg, model_cache)

    forward_op = None
    if guide_name not in ("none", None):
        forward_op = get_reward(guide_name, **cfg.get("guide_reward_params", {}))

    cls = SAMPLER_CLASSES[sampler_name]
    overrides = cfg.get("sampler_overrides", {}).get(sampler_name, {})
    sampler = cls(model=model, forward_op=forward_op, **overrides)

    gen_kwargs = dict(
        softmax_temp=cfg.get("softmax_temp", 1.0),
        randomness=cfg.get("randomness", 0.3),
        min_add_len=cfg.get("min_add_len", 60),
    )

    samples = []
    t_start = time.time()

    if wall_budget_sec is None:
        # fixed num_samples mode (original behaviour)
        batch = sampler.de_novo_generation(cfg.get("num_samples", 100), **gen_kwargs)
        samples.extend([s for s in batch if s])
    else:
        # wall-clock mode: keep generating batches until time runs out
        batch_size = cfg.get("batch_size", 16)
        while (time.time() - t_start) < wall_budget_sec:
            batch = sampler.de_novo_generation(batch_size, **gen_kwargs)
            samples.extend([s for s in batch if s])

    elapsed = time.time() - t_start

    m = compute_metrics(samples)
    metrics = {
        "sampler":          sampler_name,
        "finetune":         finetune_method,
        "guide_reward":     guide_name,
        "wall_sec":         round(elapsed, 2),
        "wall_budget_sec":  wall_budget_sec,
        "n_generated":      len(samples),
        "reward_calls":     getattr(sampler, "last_reward_evals", 0),
        "forward_passes":   getattr(sampler, "last_forward_passes", 0),
        "budget_per_sample": getattr(sampler, "last_budget_per_sample", 0),
        **{k: m[k] for k in ("validity", "uniqueness", "qed_mean", "qed_top10", "qed_max")},
        **{f"sampler_{k}": v for k, v in overrides.items()},
    }
    return samples, metrics


# ── Oracle (one-by-one with timeline logging) ─────────────────────────────────

def make_oracle_fn(oracle_name, oracle_params):
    """Return a callable: smiles -> float | None."""
    if oracle_name == "flash_affinity":
        from genmol.rewards.flash_affinity import FlashAffinityForwardOp
        model = FlashAffinityForwardOp(**oracle_params)
        def _score(smi):
            t = model([smi])
            v = t[0].item()
            return v if v != 0.0 else None
        return _score

    elif oracle_name == "boltz":
        from genmol.rewards.boltz import BoltzAffinityReward
        model = BoltzAffinityReward(**oracle_params)
        def _score(smi):
            t = model([smi])
            v = t[0].item()
            return v if v != 0.0 else None
        return _score

    else:
        raise ValueError(f"Unknown oracle: {oracle_name}")


def run_oracle_with_timeline(oracle_fn, samples, timeline_path, t_experiment_start):
    """Score samples one-by-one, logging a timeline entry after each call.

    Timeline entry (JSONL):
        call          -- cumulative oracle call index (1-based)
        smiles        -- molecule evaluated
        score         -- oracle score (null if failed)
        best_so_far   -- best score seen so far (lower = better for affinity)
        top10_scores  -- running list of top-10% scores
        top10_mean    -- mean of top-10% so far
        wall_sec      -- seconds since experiment start

    The timeline can be subsampled at stride N post-hoc to simulate an oracle
    that costs N times more per call.
    """
    timeline = []
    all_scores = []          # all valid scores so far
    call_idx = 0

    # load existing timeline if resuming
    if os.path.exists(timeline_path):
        with open(timeline_path) as f:
            for line in f:
                entry = json.loads(line)
                timeline.append(entry)
                if entry["score"] is not None:
                    all_scores.append(entry["score"])
        call_idx = len(timeline)
        print(f"  Resuming from {call_idx} existing oracle calls")

    fout = open(timeline_path, "a")

    for smi in samples:
        if not smi:
            continue

        call_idx += 1

        # -- evaluate oracle -------------------------------------------
        t_before = time.time() - t_experiment_start
        score = oracle_fn(smi)
        t_after = time.time() - t_experiment_start

        # -- update running stats --------------------------------------
        if score is not None:
            all_scores.append(score)

        best_so_far = min(all_scores) if all_scores else None   # lower = better binding
        top10_n = max(1, len(all_scores) // 10)
        top10_mean = (
            sum(sorted(all_scores)[:top10_n]) / top10_n
            if all_scores else None
        )

        entry = {
            "call":         call_idx,
            "smiles":       smi,
            "score":        score,
            "best_so_far":  best_so_far,
            "top10_mean":   top10_mean,
            "wall_sec_before": round(t_before, 3),
            "wall_sec_after":  round(t_after, 3),
        }
        timeline.append(entry)
        fout.write(json.dumps(entry) + "\n")
        fout.flush()

        if call_idx % 10 == 0:
            print(f"    call {call_idx}  score={score}  best={best_so_far:.4f}"
                  if best_so_far is not None else f"    call {call_idx}  score=None")

    fout.close()

    # summary stats from final timeline
    valid_scores = [e["score"] for e in timeline if e["score"] is not None]
    if not valid_scores:
        return {"oracle_mean": None, "oracle_top10": None, "oracle_max": None,
                "oracle_best": None, "oracle_n_scored": 0}

    sorted_scores = sorted(valid_scores)
    top10_n = max(1, len(sorted_scores) // 10)
    return {
        "oracle_mean":     round(sum(valid_scores) / len(valid_scores), 4),
        "oracle_top10":    round(sum(sorted_scores[:top10_n]) / top10_n, 4),
        "oracle_max":      round(max(valid_scores), 4),
        "oracle_best":     round(min(valid_scores), 4),   # best binder (lower IC50)
        "oracle_n_scored": len(valid_scores),
    }


# ── Budget curve simulation (post-hoc) ───────────────────────────────────────

def simulate_budget_curve(timeline_path, cost_multipliers=(1, 2, 5, 10, 50, 100)):
    """Read a timeline JSONL and simulate best_so_far under various cost multipliers.

    For multiplier N: pretend each oracle call costs N units.
    Read off best_so_far at calls [1, 1+N, 1+2N, ...].

    Returns list of dicts: {cost_multiplier, effective_calls, best_so_far, top10_mean}
    """
    timeline = []
    with open(timeline_path) as f:
        for line in f:
            timeline.append(json.loads(line))

    results = []
    for N in cost_multipliers:
        sampled = [timeline[i] for i in range(0, len(timeline), N)]
        for entry in sampled:
            results.append({
                "cost_multiplier": N,
                "real_calls":      entry["call"],
                "effective_calls": entry["call"] // N,
                "best_so_far":     entry["best_so_far"],
                "top10_mean":      entry["top10_mean"],
                "wall_sec":        entry.get("wall_sec_after"),
            })
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-oracle", action="store_true")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip generation, only re-run oracle on existing samples")
    parser.add_argument("--simulate-budget", action="store_true",
                        help="Post-hoc budget curve simulation from existing timelines")
    args = parser.parse_args()

    cfg = load_config(args.config)
    name = cfg["name"]
    output_base = os.path.join(cfg.get("output_dir", "outputs/comparison"), name)

    samplers   = cfg.get("samplers",      ["uncond"])
    finetunes  = cfg.get("finetune",      ["none"])
    guides     = cfg.get("guide_rewards", ["none"])
    oracle_name   = cfg.get("oracle")
    oracle_params = cfg.get("oracle_params", {})
    wall_budget   = cfg.get("wall_budget_sec", None)   # None = use num_samples

    grid = list(itertools.product(samplers, finetunes, guides))
    print(f"Comparison: {name}")
    print(f"Grid: {len(grid)} combos")
    print(f"Wall budget: {wall_budget}s" if wall_budget else "Fixed num_samples mode")
    print(f"Oracle: {oracle_name or 'none'}")

    if args.dry_run:
        for s, ft, g in grid:
            print(f"  {ft}/{g}/{s}")
        return

    # ── Budget curve simulation only ─────────────────────────────────
    if args.simulate_budget:
        multipliers = cfg.get("cost_multipliers", [1, 2, 5, 10, 50, 100])
        all_curves = []
        for s, ft, g in grid:
            combo_dir = os.path.join(output_base, ft, g, s)
            tl_path = os.path.join(combo_dir, "oracle_timeline.jsonl")
            if not os.path.exists(tl_path):
                print(f"No timeline found: {tl_path}")
                continue
            rows = simulate_budget_curve(tl_path, multipliers)
            for r in rows:
                r.update({"sampler": s, "finetune": ft, "guide": g})
            all_curves.extend(rows)
        if all_curves:
            curve_path = os.path.join(output_base, "budget_curves.csv")
            pd.DataFrame(all_curves).to_csv(curve_path, index=False)
            print(f"Budget curves saved → {curve_path}")
        return

    t_experiment_start = time.time()
    model_cache = {}
    all_results = []
    combo_samples = {}

    # ── Generation pass ───────────────────────────────────────────────
    if not args.skip_generation:
        for s, ft, g in grid:
            tag = f"{ft}/{g}/{s}"
            print(f"\n{'='*60}\nRunning: {tag}\n{'='*60}")
            try:
                samples, metrics = run_generation(
                    s, ft, g, cfg, model_cache, wall_budget_sec=wall_budget
                )
            except Exception as e:
                print(f"  FAILED: {e}")
                all_results.append({"sampler": s, "finetune": ft, "guide_reward": g,
                                    "error": str(e)})
                continue

            combo_dir = os.path.join(output_base, ft, g, s)
            os.makedirs(combo_dir, exist_ok=True)
            pd.DataFrame({"smiles": samples}).to_csv(
                os.path.join(combo_dir, "samples.csv"), index=False)
            with open(os.path.join(combo_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)

            combo_samples[(s, ft, g)] = samples
            all_results.append(metrics)
            print(f"  {metrics['wall_sec']}s  n={metrics['n_generated']}  "
                  f"validity={metrics['validity']:.3f}  qed={metrics['qed_mean']:.4f}")
    else:
        # load existing samples
        for s, ft, g in grid:
            combo_dir = os.path.join(output_base, ft, g, s)
            csv_path = os.path.join(combo_dir, "samples.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                combo_samples[(s, ft, g)] = df["smiles"].dropna().tolist()
                metrics_path = os.path.join(combo_dir, "metrics.json")
                if os.path.exists(metrics_path):
                    with open(metrics_path) as f:
                        all_results.append(json.load(f))

    # ── Oracle pass (per-molecule timeline) ───────────────────────────
    if oracle_name and not args.skip_oracle:
        oracle_fn = make_oracle_fn(oracle_name, oracle_params)
        print(f"\n{'='*60}\nOracle: {oracle_name}\n{'='*60}")

        for i, (s, ft, g) in enumerate(grid):
            if (s, ft, g) not in combo_samples:
                continue
            samples = [smi for smi in combo_samples[(s, ft, g)] if smi]
            if not samples:
                continue

            tag = f"{ft}/{g}/{s}"
            print(f"\n  Scoring {tag} ({len(samples)} molecules)...")

            combo_dir = os.path.join(output_base, ft, g, s)
            os.makedirs(combo_dir, exist_ok=True)
            tl_path = os.path.join(combo_dir, "oracle_timeline.jsonl")

            try:
                om = run_oracle_with_timeline(
                    oracle_fn, samples, tl_path, t_experiment_start
                )
                # patch into all_results
                for r in all_results:
                    if (r.get("sampler") == s and r.get("finetune") == ft
                            and r.get("guide_reward") == g):
                        r.update(om)
                        break

                combo_dir2 = os.path.join(combo_dir)
                with open(os.path.join(combo_dir2, "metrics.json"), "w") as f:
                    # find the matching result
                    row = next((r for r in all_results
                                if r.get("sampler") == s), {})
                    json.dump(row, f, indent=2)

                print(f"    best={om['oracle_best']}  top10={om['oracle_top10']}  "
                      f"mean={om['oracle_mean']}  n={om['oracle_n_scored']}")
            except Exception as e:
                print(f"    FAILED: {e}")

        # ── Budget curve simulation ───────────────────────────────────
        multipliers = cfg.get("cost_multipliers", [1, 2, 5, 10, 50, 100])
        all_curves = []
        for s, ft, g in grid:
            combo_dir = os.path.join(output_base, ft, g, s)
            tl_path = os.path.join(combo_dir, "oracle_timeline.jsonl")
            if not os.path.exists(tl_path):
                continue
            rows = simulate_budget_curve(tl_path, multipliers)
            for r in rows:
                r.update({"sampler": s, "finetune": ft, "guide": g})
            all_curves.extend(rows)
        if all_curves:
            curve_path = os.path.join(output_base, "budget_curves.csv")
            pd.DataFrame(all_curves).to_csv(curve_path, index=False)
            print(f"\nBudget curves → {curve_path}")

    # ── Summary ───────────────────────────────────────────────────────
    summary = pd.DataFrame(all_results)
    os.makedirs(output_base, exist_ok=True)
    summary_path = os.path.join(output_base, "summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary → {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
