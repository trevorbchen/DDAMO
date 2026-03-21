"""Design space comparison: rank methods under controlled compute budgets.

Supports two config formats:
  1. Grid mode:  samplers × finetune × guide_rewards (Cartesian product)
  2. Methods mode: explicit list of {name, sampler, finetune, guide} dicts

After generation, an oracle reward scores all molecules one-by-one, logging a
timeline entry after each call so the timeline can be subsampled post-hoc to
simulate a more expensive oracle (e.g. 10x, 100x cost multiplier).

Timeline log format (oracle_timeline.jsonl per method):
    {"call": 1, "smiles": "...", "score": -0.4, "best_so_far": -0.4,
     "top10_mean": -0.4, "wall_sec_before": 12.3, "wall_sec_after": 12.9}
    ...

Post-hoc cost simulation:
    Subsample timeline at stride N to simulate oracle that costs N times more.
    Read off best_so_far at indices [0, N, 2N, ...] to get the budget curve.

Usage:
    python scripts/comparison.py --config configs/comparison/fa_1hr.yaml
    python scripts/comparison.py --config configs/comparison/fa_1hr.yaml --dry-run
    python scripts/comparison.py --config configs/comparison/fa_1hr.yaml --skip-oracle
    python scripts/comparison.py --config configs/comparison/fa_1hr.yaml --simulate-budget
    python scripts/comparison.py --config configs/comparison/fa_1hr.yaml --skip-generation
"""

import argparse
import itertools
import json
import os
import sys
from datetime import datetime
from time import time

import pandas as pd
import yaml

sys.path.insert(0, os.path.realpath("."))
sys.path.insert(0, os.path.join(os.path.realpath("."), "src"))
sys.path.insert(0, os.path.join(os.path.realpath("."), "FlashAffinity", "src"))

from genmol.samplers import (
    Sampler, BeamSearchSampler, MCTSSampler, SMCSampler, DFKCSampler, DAPSSampler,
    load_model_from_path, decode_smiles,
)
from genmol.rewards import get_reward, KLPenalizedReward, ThresholdReward
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


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_methods(cfg):
    """Build list of methods from config.

    Supports either explicit 'methods' list or grid mode (samplers x finetune x guide_rewards).
    Returns list of dicts with keys: name, sampler, finetune, guide.
    """
    if "methods" in cfg:
        return cfg["methods"]

    # Grid mode (backward compat)
    samplers = cfg.get("samplers", ["uncond"])
    finetunes = cfg.get("finetune", ["none"])
    guides = cfg.get("guide_rewards", ["none"])
    methods = []
    for s, ft, g in itertools.product(samplers, finetunes, guides):
        methods.append({"name": f"{ft}/{g}/{s}", "sampler": s, "finetune": ft, "guide": g})
    return methods


def get_budget_sec(cfg):
    """Read budget from config, supporting both naming conventions."""
    return cfg.get("wall_clock_budget_sec") or cfg.get("wall_budget_sec")


# ── Generation methods ─────────────────────────────────────────────────────────

def run_inference_method(method, cfg, model_cache):
    """Run an inference-time method (sampler with optional guide reward).

    Returns (samples: list[str], metrics: dict).
    """
    sampler_name = method["sampler"]
    guide_name = method.get("guide", "none")
    timeout_sec = get_budget_sec(cfg)

    # Load model (cached)
    if "base" not in model_cache:
        model_cache["base"] = load_model_from_path(os.path.realpath(cfg["model_path"]))
    model = model_cache["base"]

    # Guide reward
    if guide_name in ("none", None, ""):
        forward_op = None
    else:
        forward_op = get_reward(guide_name, **cfg.get("guide_reward_params", {}))

    # KL penalty wrapper
    kl_cfg = cfg.get("kl_penalty", {})
    if kl_cfg.get("enabled", False) and forward_op is not None:
        lam = kl_cfg.get("lam", 0.01)
        forward_op = KLPenalizedReward(forward_op, model, lam=lam)

    # Sampler
    cls = SAMPLER_CLASSES[sampler_name]
    overrides = cfg.get("sampler_overrides", {}).get(sampler_name, {})
    sampler = cls(model=model, forward_op=forward_op, **overrides)

    # Generate — all methods just generate molecules; oracle scoring happens separately
    start_ts = datetime.now().isoformat()
    t0 = time()

    samples = sampler.de_novo_generation(
        cfg.get("num_samples", 100),
        softmax_temp=cfg.get("softmax_temp", 1.0),
        randomness=cfg.get("randomness", 0.3),
        min_add_len=cfg.get("min_add_len", 60),
        timeout_sec=timeout_sec,
    )

    elapsed = time() - t0
    end_ts = datetime.now().isoformat()

    m = compute_metrics(samples)
    metrics = {
        "method": method["name"],
        "sampler": sampler_name,
        "finetune": "none",
        "guide_reward": guide_name,
        "start_time": start_ts,
        "end_time": end_ts,
        "wall_sec": round(elapsed, 2),
        "reward_calls": getattr(sampler, "last_reward_evals", 0),
        "forward_passes": getattr(sampler, "last_forward_passes", 0),
        **{k: m[k] for k in ("validity", "uniqueness", "qed_mean", "qed_top10", "qed_max")},
        "num_samples": len(samples),
        **{f"sampler_{k}": v for k, v in overrides.items()},
    }
    return samples, metrics


def run_ddpp_method(method, cfg, model_cache):
    """Run DDPP: train for budget, then generate unconditionally.

    Returns (samples: list[str], metrics: dict).
    """
    from genmol.finetune import DDPPLBTrainer

    timeout_sec = get_budget_sec(cfg)
    ddpp_cfg = cfg.get("ddpp_overrides", {})

    # Load pretrained model
    if "base" not in model_cache:
        model_cache["base"] = load_model_from_path(os.path.realpath(cfg["model_path"]))
    model = model_cache["base"]

    # Guide reward for DDPP training data collection
    guide_name = method.get("guide", "none")
    if guide_name in ("none", None, ""):
        oracle_name = cfg.get("oracle", "flash_affinity")
        reward_fn = get_reward(oracle_name, **cfg.get("oracle_params", {}))
    else:
        reward_fn = get_reward(guide_name, **cfg.get("guide_reward_params", {}))

    model_path = os.path.realpath(cfg["model_path"])
    trainer = DDPPLBTrainer(
        model_path=model_path,
        reward_fn=reward_fn,
        lr=ddpp_cfg.get("lr", 1e-4),
        batch_size=ddpp_cfg.get("batch_size", 16),
        lr_logz=ddpp_cfg.get("lr_logz", 1e-3),
        warmup_logz_steps=ddpp_cfg.get("warmup_logz_steps", 500),
        replay_buffer_size=ddpp_cfg.get("replay_buffer_size", 10000),
        refill_interval=ddpp_cfg.get("refill_interval", 250),
        ema_decay=ddpp_cfg.get("ema_decay", 0.9999),
    )

    # Budget split: 85% train, 10% generate, 5% score
    t0 = time()
    train_budget = timeout_sec * 0.85 if timeout_sec else None
    max_steps = ddpp_cfg.get("num_steps", 10000)
    trainer.train(max_steps, timeout_sec=train_budget)
    train_elapsed = time() - t0
    print(f"  [ddpp] Training done: {train_elapsed:.0f}s, {trainer.global_step} steps", flush=True)

    num_gen = cfg.get("ddpp_num_gen", 1000)
    samples = trainer.generate(num_gen)
    samples = [s for s in samples if s]
    gen_elapsed = time() - t0 - train_elapsed
    print(f"  [ddpp] Generated {len(samples)} valid molecules in {gen_elapsed:.0f}s", flush=True)

    # Score with FA within remaining budget
    score_budget = timeout_sec - (time() - t0) if timeout_sec else None
    reward_calls = 0
    if samples and score_budget and score_budget > 0:
        from evals.flash_affinity import run_flash_affinity
        oracle_params = cfg.get("oracle_params", {})
        print(f"  [ddpp] Scoring {len(samples)} mols ({score_budget:.0f}s remaining)...", flush=True)
        scores = run_flash_affinity(samples, **oracle_params)
        reward_calls = len(samples)
    else:
        scores = [None] * len(samples)

    total_elapsed = time() - t0

    metrics = {
        "method": method["name"],
        "sampler": "uncond",
        "finetune": "ddpp",
        "guide_reward": guide_name,
        "wall_sec": round(total_elapsed, 2),
        "train_sec": round(train_elapsed, 2),
        "train_steps": trainer.global_step,
        "reward_calls": reward_calls,
        "forward_passes": 0,
        **compute_metrics(samples),
        "num_samples": len(samples),
    }
    return samples, metrics


# ── MCTS + surrogate active learning loop ──────────────────────────────────────

def run_mcts_surrogate_method(method, cfg, model_cache, oracle_fn,
                              timeline_path, t_experiment_start,
                              higher_is_better=True):
    """Active learning: MCTS guided by surrogate + online surrogate/model update.

    Each epoch:
      1. Generate candidates with MCTS (surrogate as forward_op once fitted)
      2. Score all candidates with surrogate, keep top 20%
      3. Oracle-score top 20% one-by-one (logged to timeline)
      4. Retrain surrogate on all accumulated oracle labels
      5. Run DDPP gradient steps to fine-tune the generative model

    Returns (oracle_scored_smiles, metrics) with metrics["oracle_logged"]=True
    so the standard oracle pass skips this method.
    """
    from genmol.surrogate import FingerprintSurrogate
    from genmol.finetune import DDPPLBTrainer
    import random as _random

    timeout_sec = get_budget_sec(cfg)
    t0 = time()

    # Load base model
    if "base" not in model_cache:
        model_cache["base"] = load_model_from_path(os.path.realpath(cfg["model_path"]))
    model = model_cache["base"]

    overrides = cfg.get("sampler_overrides", {}).get("mcts", {})
    active_cfg = cfg.get("mcts_surrogate", {})
    candidates_per_epoch = active_cfg.get("candidates_per_epoch", 200)
    top_frac = active_cfg.get("top_frac", 0.2)           # fraction sent to oracle
    ddpp_steps_per_epoch = active_cfg.get("ddpp_steps_per_epoch", 50)

    surrogate = FingerprintSurrogate()

    # DDPP trainer uses surrogate as reward — since surrogate is updated in-place,
    # the DDPP loss always reflects the latest surrogate knowledge.
    ddpp_cfg = cfg.get("ddpp_overrides", {})
    trainer = DDPPLBTrainer(
        model_path=os.path.realpath(cfg["model_path"]),
        reward_fn=surrogate,
        lr=ddpp_cfg.get("lr", 1e-4),
        batch_size=ddpp_cfg.get("batch_size", 16),
        lr_logz=ddpp_cfg.get("lr_logz", 1e-3),
        warmup_logz_steps=ddpp_cfg.get("warmup_logz_steps", 0),
        replay_buffer_size=ddpp_cfg.get("replay_buffer_size", 10000),
        refill_interval=0,  # we control refills manually each epoch
        initial_buffer_from_pretrained=0,  # skip initial fill; surrogate not fitted yet
        ema_decay=ddpp_cfg.get("ema_decay", 0.9999),
    )

    # Timeline state
    all_oracle_smiles_scores = []  # (smiles, score)
    call_idx = 0
    all_scores_seen = []
    os.makedirs(os.path.dirname(timeline_path) or ".", exist_ok=True)
    fout = open(timeline_path, "a")
    epoch = 0

    while (time() - t0) < (timeout_sec or float("inf")):
        epoch += 1
        remaining = (timeout_sec - (time() - t0)) if timeout_sec else None
        if remaining is not None and remaining < 10:
            break

        # 1. Generate candidates with MCTS
        # Use the fine-tuned model after first epoch, base model before
        gen_model = trainer.finetuned if epoch > 1 else model
        mcts = MCTSSampler(
            model=gen_model,
            forward_op=surrogate if surrogate.is_fitted else None,
            **overrides,
        )
        candidates = mcts.de_novo_generation(
            candidates_per_epoch,
            softmax_temp=cfg.get("softmax_temp", 1.0),
            randomness=cfg.get("randomness", 0.3),
            min_add_len=cfg.get("min_add_len", 60),
        )
        candidates = [s for s in candidates if s]
        if not candidates:
            continue

        # 2. Score with surrogate, take top fraction
        top_n = max(1, int(len(candidates) * top_frac))
        if surrogate.is_fitted:
            surr_scores = surrogate.predict(candidates)
            ranked = sorted(zip(surr_scores, candidates),
                            reverse=higher_is_better, key=lambda x: x[0])
            top_candidates = [smi for _, smi in ranked[:top_n]]
        else:
            # No surrogate yet: random sample
            top_candidates = _random.sample(candidates, min(top_n, len(candidates)))

        # 3. Oracle-score top candidates (log to timeline)
        epoch_oracle = []
        for smi in top_candidates:
            if timeout_sec and (time() - t0) >= timeout_sec:
                break
            call_idx += 1
            t_before = time() - t_experiment_start
            score = oracle_fn(smi)
            t_after = time() - t_experiment_start

            if score is not None:
                all_scores_seen.append(score)
                epoch_oracle.append((smi, score))
                all_oracle_smiles_scores.append((smi, score))

            if all_scores_seen:
                sorted_ts = sorted(all_scores_seen, reverse=higher_is_better)
                best_so_far = sorted_ts[0]
                top10_n = max(1, len(all_scores_seen) // 10)
                top10_mean = sum(sorted_ts[:top10_n]) / top10_n
            else:
                best_so_far = top10_mean = None

            entry = {
                "call":            call_idx,
                "smiles":          smi,
                "score":           score,
                "best_so_far":     best_so_far,
                "top10_mean":      top10_mean,
                "wall_sec_before": round(t_before, 3),
                "wall_sec_after":  round(t_after, 3),
            }
            fout.write(json.dumps(entry) + "\n")
            fout.flush()

        # 4. Retrain surrogate on all accumulated oracle data
        if all_oracle_smiles_scores:
            smiles_fit = [s for s, _ in all_oracle_smiles_scores]
            scores_fit = [sc for _, sc in all_oracle_smiles_scores]
            surrogate.fit(smiles_fit, scores_fit)

        # 5. DDPP gradient steps (fine-tune generative model via updated surrogate)
        if surrogate.is_fitted and epoch_oracle:
            trainer._fill_buffer(trainer.finetuned, ddpp_cfg.get("batch_size", 16),
                                  label=f"epoch-{epoch}-onpolicy")
            for _ in range(ddpp_steps_per_epoch):
                if timeout_sec and (time() - t0) >= timeout_sec:
                    break
                trainer.train_step()

        best = all_scores_seen[0] if all_scores_seen else None
        if all_scores_seen:
            best = sorted(all_scores_seen, reverse=higher_is_better)[0]
        print(f"  [mcts_surrogate] epoch {epoch}: {len(epoch_oracle)} oracle calls, "
              f"best={best}, surrogate_pts={len(all_oracle_smiles_scores)}, "
              f"{time()-t0:.0f}s/{timeout_sec}s", flush=True)

    fout.close()

    all_oracle_smiles = [s for s, _ in all_oracle_smiles_scores]
    valid_scores = [sc for _, sc in all_oracle_smiles_scores if sc is not None]
    if valid_scores:
        sorted_vs = sorted(valid_scores, reverse=higher_is_better)
        top10_n = max(1, len(valid_scores) // 10)
        om = {
            "oracle_mean":     round(sum(valid_scores) / len(valid_scores), 4),
            "oracle_top10":    round(sum(sorted_vs[:top10_n]) / top10_n, 4),
            "oracle_best":     round(sorted_vs[0], 4),
            "oracle_n_scored": len(valid_scores),
        }
    else:
        om = {"oracle_mean": None, "oracle_top10": None,
              "oracle_best": None, "oracle_n_scored": 0}

    metrics = {
        "method":        method["name"],
        "sampler":       "mcts",
        "finetune":      "mcts_surrogate",
        "guide_reward":  "surrogate",
        "wall_sec":      round(time() - t0, 2),
        "reward_calls":  call_idx,
        "num_samples":   len(all_oracle_smiles),
        "surrogate_pts": len(all_oracle_smiles_scores),
        "oracle_logged": True,
        **om,
    }
    return all_oracle_smiles, metrics


# ── DDPP with threshold reward ──────────────────────────────────────────────────

def run_threshold_ddpp_method(method, cfg, model_cache):
    """DDPP fine-tuning with max(r(x)-t, 0) reward.

    Wraps the oracle reward with ThresholdReward before passing to DDPPLBTrainer.
    t = 80th percentile of all rewards seen, updated every 10 batch calls.
    Only top-20% molecules get positive reward, focusing training on the best candidates.

    Otherwise identical to run_ddpp_method.
    """
    from genmol.rewards import ThresholdReward

    timeout_sec = get_budget_sec(cfg)
    ddpp_cfg = cfg.get("ddpp_overrides", {})
    threshold_cfg = cfg.get("threshold_reward", {})

    if "base" not in model_cache:
        model_cache["base"] = load_model_from_path(os.path.realpath(cfg["model_path"]))

    guide_name = method.get("guide", "none")
    if guide_name in ("none", None, ""):
        oracle_name = cfg.get("oracle", "flash_affinity")
        base_reward_fn = get_reward(oracle_name, **cfg.get("oracle_params", {}))
    else:
        base_reward_fn = get_reward(guide_name, **cfg.get("guide_reward_params", {}))

    # Wrap with threshold
    reward_fn = ThresholdReward(
        base_reward_fn,
        update_every=threshold_cfg.get("update_every", 10),
        percentile=threshold_cfg.get("percentile", 80.0),
        min_samples=threshold_cfg.get("min_samples", 20),
    )

    model_path = os.path.realpath(cfg["model_path"])
    from genmol.finetune import DDPPLBTrainer
    trainer = DDPPLBTrainer(
        model_path=model_path,
        reward_fn=reward_fn,
        lr=ddpp_cfg.get("lr", 1e-4),
        batch_size=ddpp_cfg.get("batch_size", 16),
        lr_logz=ddpp_cfg.get("lr_logz", 1e-3),
        warmup_logz_steps=ddpp_cfg.get("warmup_logz_steps", 500),
        replay_buffer_size=ddpp_cfg.get("replay_buffer_size", 10000),
        refill_interval=ddpp_cfg.get("refill_interval", 250),
        ema_decay=ddpp_cfg.get("ema_decay", 0.9999),
    )

    t0 = time()
    train_budget = timeout_sec * 0.85 if timeout_sec else None
    max_steps = ddpp_cfg.get("num_steps", 10000)
    trainer.train(max_steps, timeout_sec=train_budget)
    train_elapsed = time() - t0
    print(f"  [ddpp_threshold] Training done: {train_elapsed:.0f}s, "
          f"{trainer.global_step} steps, threshold={reward_fn.threshold:.4f}", flush=True)

    num_gen = cfg.get("ddpp_num_gen", 1000)
    samples = trainer.generate(num_gen)
    samples = [s for s in samples if s]

    metrics = {
        "method":        method["name"],
        "sampler":       "uncond",
        "finetune":      "ddpp_threshold",
        "guide_reward":  guide_name,
        "wall_sec":      round(time() - t0, 2),
        "train_sec":     round(train_elapsed, 2),
        "train_steps":   trainer.global_step,
        "reward_threshold": round(reward_fn.threshold, 4),
        "reward_calls":  0,
        "forward_passes": 0,
        **compute_metrics(samples),
        "num_samples":   len(samples),
    }
    return samples, metrics


# ── Oracle (per-call timeline logging) ─────────────────────────────────────────

def make_oracle_fn(oracle_name, oracle_params):
    """Return a callable: smiles -> float | None (scores one molecule at a time)."""
    if oracle_name == "flash_affinity":
        try:
            from genmol.rewards.flash_affinity import FlashAffinityForwardOp
            model = FlashAffinityForwardOp(**oracle_params)
            def _score_fa(smi):
                t = model([smi])
                v = t[0].item()
                return v if v != 0.0 else None
            return _score_fa
        except Exception:
            pass
        # Fallback: wrap batch scorer
        from evals.flash_affinity import run_flash_affinity
        def _score_fa_batch(smi):
            results = run_flash_affinity([smi], **oracle_params)
            return results[0] if results else None
        return _score_fa_batch

    elif oracle_name == "boltz":
        from genmol.rewards.boltz import BoltzAffinityReward
        model = BoltzAffinityReward(**oracle_params)
        def _score_boltz(smi):
            t = model([smi])
            v = t[0].item()
            return v if v != 0.0 else None
        return _score_boltz

    else:
        raise ValueError(f"Unknown oracle: {oracle_name}. Supported: flash_affinity, boltz")


def run_oracle_with_timeline(oracle_fn, samples, timeline_path, t_experiment_start,
                             higher_is_better=True):
    """Score samples one-by-one, logging a JSONL timeline entry after each call.

    Timeline entry fields:
        call             -- cumulative oracle call index (1-based)
        smiles           -- molecule evaluated
        score            -- oracle score (null if failed)
        best_so_far      -- best score seen so far
        top10_mean       -- mean of running top-10% scores
        wall_sec_before  -- seconds since experiment start (before oracle call)
        wall_sec_after   -- seconds since experiment start (after oracle call)

    The timeline can be subsampled at stride N post-hoc to simulate an oracle
    that costs N times more per call.
    """
    timeline = []
    all_scores = []
    call_idx = 0

    # Resume from existing timeline if present
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

        t_before = time() - t_experiment_start
        score = oracle_fn(smi)
        t_after = time() - t_experiment_start

        if score is not None:
            all_scores.append(score)

        if all_scores:
            sorted_scores = sorted(all_scores, reverse=higher_is_better)
            best_so_far = sorted_scores[0]
            top10_n = max(1, len(all_scores) // 10)
            top10_mean = sum(sorted_scores[:top10_n]) / top10_n
        else:
            best_so_far = None
            top10_mean = None

        entry = {
            "call":            call_idx,
            "smiles":          smi,
            "score":           score,
            "best_so_far":     best_so_far,
            "top10_mean":      top10_mean,
            "wall_sec_before": round(t_before, 3),
            "wall_sec_after":  round(t_after, 3),
        }
        timeline.append(entry)
        fout.write(json.dumps(entry) + "\n")
        fout.flush()

        if call_idx % 10 == 0:
            if best_so_far is not None:
                print(f"    call {call_idx}  score={score}  best={best_so_far:.4f}")
            else:
                print(f"    call {call_idx}  score=None")

    fout.close()

    valid_scores = [e["score"] for e in timeline if e["score"] is not None]
    if not valid_scores:
        return {"oracle_mean": None, "oracle_top10": None, "oracle_best": None,
                "oracle_n_scored": 0}

    sorted_scores = sorted(valid_scores, reverse=higher_is_better)
    top10_n = max(1, len(sorted_scores) // 10)
    return {
        "oracle_mean":     round(sum(valid_scores) / len(valid_scores), 4),
        "oracle_top10":    round(sum(sorted_scores[:top10_n]) / top10_n, 4),
        "oracle_best":     round(sorted_scores[0], 4),
        "oracle_n_scored": len(valid_scores),
    }


# ── Budget curve simulation (post-hoc) ─────────────────────────────────────────

def simulate_budget_curve(timeline_path, cost_multipliers=(1, 2, 5, 10, 50, 100)):
    """Read a timeline JSONL and simulate best_so_far under various cost multipliers.

    For multiplier N: pretend each oracle call costs N units.
    Read off best_so_far at calls [0, N, 2N, ...].

    Returns list of dicts: {cost_multiplier, real_calls, effective_calls,
                             best_so_far, top10_mean, wall_sec}
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
                "cost_multiplier":  N,
                "real_calls":       entry["call"],
                "effective_calls":  entry["call"] // N,
                "best_so_far":      entry["best_so_far"],
                "top10_mean":       entry["top10_mean"],
                "wall_sec":         entry.get("wall_sec_after"),
            })
    return results


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Design space comparison orchestrator.")
    parser.add_argument("--config", required=True, help="Comparison config YAML.")
    parser.add_argument("--dry-run", action="store_true", help="Print methods without running.")
    parser.add_argument("--skip-oracle", action="store_true", help="Skip oracle evaluation pass.")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip generation; re-run oracle on existing samples.csv files.")
    parser.add_argument("--simulate-budget", action="store_true",
                        help="Post-hoc budget curve simulation from existing timelines only.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    name = cfg["name"]
    output_base = os.path.join(cfg.get("output_dir", "outputs/comparison"), name)
    methods = build_methods(cfg)

    budget_sec = get_budget_sec(cfg)
    oracle_name = cfg.get("oracle")
    oracle_params = cfg.get("oracle_params", {})
    higher_is_better = cfg.get("oracle_higher_is_better", True)

    print(f"Comparison: {name}")
    print(f"Methods: {len(methods)}")
    print(f"Budget: {budget_sec}s" if budget_sec else "Budget: unlimited")
    print(f"Oracle: {oracle_name or 'none'}")
    kl_cfg = cfg.get("kl_penalty", {})
    if kl_cfg.get("enabled"):
        print(f"KL penalty: lambda={kl_cfg.get('lam', 0.01)}")

    if args.dry_run:
        for m in methods:
            overrides = cfg.get("sampler_overrides", {}).get(m["sampler"], {})
            print(f"  {m['name']}  sampler={m['sampler']}  finetune={m.get('finetune','none')}  "
                  f"guide={m.get('guide', 'none')}  overrides={overrides}")
        return

    # ── Budget curve simulation only ─────────────────────────────────
    if args.simulate_budget:
        multipliers = cfg.get("cost_multipliers", [1, 2, 5, 10, 50, 100])
        all_curves = []
        for m in methods:
            method_dir = os.path.join(output_base, m["name"])
            tl_path = os.path.join(method_dir, "oracle_timeline.jsonl")
            if not os.path.exists(tl_path):
                print(f"No timeline found: {tl_path}")
                continue
            rows = simulate_budget_curve(tl_path, multipliers)
            for r in rows:
                r["method"] = m["name"]
            all_curves.extend(rows)
        if all_curves:
            curve_path = os.path.join(output_base, "budget_curves.csv")
            os.makedirs(output_base, exist_ok=True)
            pd.DataFrame(all_curves).to_csv(curve_path, index=False)
            print(f"Budget curves saved -> {curve_path}")
        return

    t_experiment_start = time()
    model_cache = {}
    all_results = []
    method_samples = {}

    # ── Generation pass ───────────────────────────────────────────────
    if not args.skip_generation:
        for m in methods:
            print(f"\n{'='*60}")
            print(f"Running: {m['name']}")
            print(f"{'='*60}")

            try:
                finetune = m.get("finetune", "none")
                if finetune == "mcts_surrogate":
                    if not oracle_name:
                        raise ValueError("mcts_surrogate requires an oracle in config")
                    oracle_fn_al = make_oracle_fn(oracle_name, oracle_params)
                    method_dir_al = os.path.join(output_base, m["name"])
                    os.makedirs(method_dir_al, exist_ok=True)
                    tl_path_al = os.path.join(method_dir_al, "oracle_timeline.jsonl")
                    samples, metrics = run_mcts_surrogate_method(
                        m, cfg, model_cache, oracle_fn_al,
                        tl_path_al, t_experiment_start,
                        higher_is_better=higher_is_better,
                    )
                elif finetune == "ddpp_threshold":
                    samples, metrics = run_threshold_ddpp_method(m, cfg, model_cache)
                elif finetune == "ddpp":
                    samples, metrics = run_ddpp_method(m, cfg, model_cache)
                else:
                    samples, metrics = run_inference_method(m, cfg, model_cache)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  FAILED: {e}")
                all_results.append({"method": m["name"], "error": str(e)})
                continue

            method_dir = os.path.join(output_base, m["name"])
            os.makedirs(method_dir, exist_ok=True)
            pd.DataFrame({"smiles": samples}).to_csv(
                os.path.join(method_dir, "samples.csv"), index=False)
            with open(os.path.join(method_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)

            method_samples[m["name"]] = samples
            all_results.append(metrics)
            print(f"  {metrics['wall_sec']}s  validity={metrics.get('validity', 0):.3f}  "
                  f"qed={metrics.get('qed_mean', 0):.4f}  n={metrics['num_samples']}")
    else:
        # Load existing samples from disk
        for m in methods:
            method_dir = os.path.join(output_base, m["name"])
            csv_path = os.path.join(method_dir, "samples.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                method_samples[m["name"]] = df["smiles"].dropna().tolist()
                metrics_path = os.path.join(method_dir, "metrics.json")
                if os.path.exists(metrics_path):
                    with open(metrics_path) as f:
                        all_results.append(json.load(f))

    # ── Oracle pass (per-molecule timeline) ───────────────────────────
    if oracle_name and not args.skip_oracle:
        oracle_fn = make_oracle_fn(oracle_name, oracle_params)
        print(f"\n{'='*60}")
        print(f"Oracle: {oracle_name}  higher_is_better={higher_is_better}")
        print(f"{'='*60}")

        for m in methods:
            if m["name"] not in method_samples:
                continue
            # Skip methods that logged their own oracle timeline (e.g. mcts_surrogate)
            method_result = next((r for r in all_results if r.get("method") == m["name"]), {})
            if method_result.get("oracle_logged"):
                continue
            samples = [s for s in method_samples[m["name"]] if s]
            if not samples:
                continue

            print(f"\n  Scoring {m['name']} ({len(samples)} molecules)...")
            method_dir = os.path.join(output_base, m["name"])
            os.makedirs(method_dir, exist_ok=True)
            tl_path = os.path.join(method_dir, "oracle_timeline.jsonl")

            try:
                om = run_oracle_with_timeline(
                    oracle_fn, samples, tl_path, t_experiment_start,
                    higher_is_better=higher_is_better,
                )
                # Patch oracle metrics into all_results
                for r in all_results:
                    if r.get("method") == m["name"]:
                        r.update(om)
                        break
                # Re-save metrics.json with oracle stats included
                metrics_path = os.path.join(method_dir, "metrics.json")
                row = next((r for r in all_results if r.get("method") == m["name"]), {})
                with open(metrics_path, "w") as f:
                    json.dump(row, f, indent=2)
                print(f"    best={om['oracle_best']}  top10={om['oracle_top10']}  "
                      f"mean={om['oracle_mean']}  n={om['oracle_n_scored']}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"    FAILED: {e}")

        # Auto-run budget curve simulation after oracle pass
        multipliers = cfg.get("cost_multipliers", [1, 2, 5, 10, 50, 100])
        all_curves = []
        for m in methods:
            method_dir = os.path.join(output_base, m["name"])
            tl_path = os.path.join(method_dir, "oracle_timeline.jsonl")
            if not os.path.exists(tl_path):
                continue
            rows = simulate_budget_curve(tl_path, multipliers)
            for r in rows:
                r["method"] = m["name"]
            all_curves.extend(rows)
        if all_curves:
            curve_path = os.path.join(output_base, "budget_curves.csv")
            pd.DataFrame(all_curves).to_csv(curve_path, index=False)
            print(f"\nBudget curves -> {curve_path}")

    # ── Summary ───────────────────────────────────────────────────────
    summary = pd.DataFrame(all_results)
    os.makedirs(output_base, exist_ok=True)
    summary_path = os.path.join(output_base, "summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"\n{'='*60}")
    print(f"Summary: {len(all_results)} methods -> {summary_path}")
    print(f"{'='*60}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
