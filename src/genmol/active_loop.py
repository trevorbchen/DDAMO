"""
Adaptive molecular optimisation with Thompson Sampling and tail-aware fine-tuning.

Algorithm per epoch:
  1. Generate M candidates from the fine-tuned backbone (cheap).
  2. Encode candidates → backbone embeddings.
  3. Thompson Sampling: s_i ~ N(mu_i, sigma_i²) → select top-K.
  4. Evaluate top-K with FA oracle (expensive).
  5. Fine-tune backbone on above-tau buffer molecules (CVaR or best-so-far).
  6. Reinit + retrain ensemble on all accumulated FA data.
  7. Log diagnostics: f*, mean sigma, Spearman rho.

Selection modes:
  thompson         — generate M, Thompson-sample top-K (default)
  beam             — beam search guided by ensemble mean μ (deterministic)
  thompson_beam    — beam search guided by Thompson sample μ+σε (stochastic)
  thompson_mcts    — MCTS guided by Thompson sample (complete Thompson)
  bandit_ensemble  — UCB bandit: per-candidate σ from ensemble, select by UCB=μ+z·σ, eliminate when UCB < best_fa
  bandit_global    — UCB bandit: global σ from residuals, select by UCB=μ+z·σ_global, eliminate when UCB < best_fa
  bandit_ts        — Thompson sampling bandit: draw s_i~N(μ_i,σ_i²), evaluate all where s_i > best_fa (TS-optimistic set, descending), eliminate when UCB < best_fa

Note: tau_mode="cvar" (default) sets τ = 80th percentile of all accumulated scores (top 20% CVaR).
      tau_mode="best_so_far" sets τ = current best FA score. These are distinct reward shaping strategies.
"""

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch

from genmol.ensemble import EnsembleScorer, encode_smiles
from genmol.finetune.ddpp import DDPPLBTrainer

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ActiveLoopConfig:
    # Budget — use wall_budget_sec for time-based runs (overrides n_epochs if set)
    n_epochs: int = 20
    wall_budget_sec: float = None   # if set, run until this many seconds elapsed
    M: int = 1000           # candidates generated per epoch
    K: int = 25             # FA calls per epoch (oracle budget)

    # Oracle
    reward_shift: float = 3.0   # shifts negative FA scores to positive (for DDPP log)

    # Clipped-reward variant
    tau_mode: Literal["cvar", "best_so_far", "none", "anneal"] = "cvar"
    tau_quantile: float = 0.80  # used when tau_mode="cvar" (top 20%)

    # Ensemble + selection method
    use_ensemble: bool = True
    selection: Literal[
        "thompson", "beam", "thompson_beam", "thompson_mcts",
        "bandit_ensemble", "bandit_global", "bandit_ts", "bandit_ts_upper"
    ] = "thompson"
    beam_width: int = 8
    mcts_rollout_budget: int = None   # per-sample rollout budget; None = MCTS default
    n_ensemble: int = 10
    ensemble_hidden: int = 256
    ensemble_epochs: int = 100
    ensemble_lr: float = 1e-3
    ensemble_batch: int = 256

    # Bandit params (bandit_ensemble / bandit_global)
    bandit_z: float = 1.96   # z-score for UCB score and elimination threshold

    # Reproducibility
    seed: int = 0

    # DDPP fine-tuning
    ddpp_steps_per_epoch: int = 50

    # Generation
    softmax_temp: float = 1.0
    randomness: float = 0.3
    min_add_len: int = 60

    # Logging
    output_dir: str = "outputs/active_loop"
    log_every: int = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _topk_indices(scores: torch.Tensor, k: int) -> list[int]:
    k = min(k, len(scores))
    return scores.topk(k).indices.tolist()


def _compute_tau(fa_scores: list[float], mode: str, quantile: float, f_star: float,
                 epoch: int = 0, wall_frac: float = 0.0) -> float:
    if mode == "none":
        return float(min(fa_scores)) - 1.0  # all molecules used for training
    if mode == "anneal":
        # Linear anneal: 0% → 99% quantile over the run (wall_frac = elapsed/budget)
        q = min(0.99, wall_frac * 0.99)
        n_keep = max(1, int(len(fa_scores) * (1.0 - q)))
        tau = float(sorted(fa_scores, reverse=True)[n_keep - 1])
        return tau
    if mode == "best_so_far":
        return f_star
    # CVaR: quantile of all accumulated scores
    return float(np.quantile(fa_scores, quantile))


def _make_ensemble_forward_op(ensemble, gen_model, mean_sigma_ref, thompson: bool = False):
    """
    Returns a forward_op callable for BeamSearchSampler / MCTSSampler.
    If thompson=False: returns ensemble mean μ (deterministic).
    If thompson=True:  returns Thompson sample μ + σ·ε (stochastic).
    """
    def _op(smiles_list):
        valid_mask = [s is not None and isinstance(s, str) and len(s) > 0
                      for s in smiles_list]
        valid_smiles = [s for s, ok in zip(smiles_list, valid_mask) if ok]
        out = torch.full((len(smiles_list),), -1e9, device=gen_model.device)
        if not valid_smiles:
            return out
        gen_model.set_eval()
        emb = gen_model.get_embeddings(valid_smiles)
        mu, sigma = ensemble.predict_unnorm(emb)
        gen_model.set_train()
        mean_sigma_ref[0] = sigma.mean().item()
        valid_idx = [i for i, ok in enumerate(valid_mask) if ok]
        scores = (mu + sigma * torch.randn_like(mu)) if thompson else mu
        out[valid_idx] = scores
        return out
    return _op


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_active_loop(
    ddpp_trainer: DDPPLBTrainer,
    fa_oracle: Callable[[list[str]], list[float]],
    cfg: ActiveLoopConfig = None,
) -> dict:
    """
    Run the active learning loop.

    Args:
        ddpp_trainer:  Initialised DDPPLBTrainer (provides generation + fine-tuning).
        fa_oracle:     callable(list[str]) → list[float], returns FA scores (higher better).
        cfg:           ActiveLoopConfig; uses defaults if None.

    Returns:
        dict with 'f_star', 'best_smiles', 'history' (per-epoch logs), 'all_pairs'.
    """
    if cfg is None:
        cfg = ActiveLoopConfig()

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "active_loop_log.jsonl"

    # Use abstract GenerativeModel interface — works with any trainer/backbone
    gen_model = ddpp_trainer.model
    device = ddpp_trainer.device

    # Ensemble: lazy init after first epoch (need input_dim from backbone)
    hidden_size = gen_model.embedding_dim
    ensemble = EnsembleScorer(
        input_dim=hidden_size,
        n_models=cfg.n_ensemble,
        hidden_dim=cfg.ensemble_hidden,
        device=str(device),
    )

    # Accumulated buffer: (smiles, fa_score) pairs — raw FA scores, not shifted
    all_pairs: list[tuple[str, float]] = []
    f_star = float("-inf")
    best_smiles = None
    history = []

    # sigma_global for bandit_global: std(fa - surrogate_mu) on accumulated pairs
    # Initialised to a conservative prior; updated each epoch after ensemble training.
    sigma_global = 0.2

    # Held-out set for rho: 20% of pairs, updated each epoch
    rng = random.Random(cfg.seed)

    wall_start = time.time()
    if cfg.wall_budget_sec:
        log.info(f"Starting active loop: wall_budget={cfg.wall_budget_sec/3600:.1f}hr, "
                 f"M={cfg.M}, K={cfg.K}, selection={cfg.selection}")
    else:
        log.info(f"Starting active loop: {cfg.n_epochs} epochs, "
                 f"M={cfg.M}, K={cfg.K}, selection={cfg.selection}")

    # Per-call oracle timeline (same format as search methods)
    timeline_path = output_dir / "oracle_timeline.jsonl"
    oracle_call_counter = 0
    best_so_far = float("-inf")

    # Reward-hacking monitors (QED + SA) — not counted as oracle cost
    try:
        from rdkit import Chem
        from rdkit.Chem import QED as _QED
        def _compute_qed(smi):
            if not isinstance(smi, str) or len(smi) == 0:
                return None
            mol = Chem.MolFromSmiles(smi)
            return round(_QED.qed(mol), 4) if mol else None
    except ImportError:
        _compute_qed = lambda smi: None

    try:
        from rdkit.Chem import RDConfig
        import sys, os
        sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
        import sascorer as _sa
        from rdkit import Chem as _Chem2
        def _compute_sa(smi):
            if not isinstance(smi, str) or len(smi) == 0:
                return None
            mol = _Chem2.MolFromSmiles(smi)
            return round(_sa.calculateScore(mol), 4) if mol else None
    except Exception:
        _compute_sa = lambda smi: None

    def logged_fa_oracle(smiles_list, phase="oracle_eval"):
        """Wrapper that logs every FA call to oracle_timeline.jsonl."""
        nonlocal oracle_call_counter, best_so_far
        scores = fa_oracle(smiles_list)
        with open(timeline_path, "a") as tf:
            for smi, sc in zip(smiles_list, scores):
                if sc is None or np.isnan(sc):
                    continue
                oracle_call_counter += 1
                best_so_far = max(best_so_far, sc)
                entry = {
                    "call": oracle_call_counter,
                    "smiles": smi,
                    "score": sc,
                    "best_so_far": best_so_far,
                    "epoch": epoch,
                    "phase": phase,
                    "elapsed_sec": round(time.time() - wall_start, 3),
                    "qed": _compute_qed(smi),
                    "sa": _compute_sa(smi),
                }
                tf.write(json.dumps(entry) + "\n")
        return scores

    epoch = 0
    cumulative_diffusion_passes = 0  # total molecules generated (diffusion forward passes)
    cumulative_ddpp_steps = 0        # total DDPP gradient steps
    while True:
        if cfg.wall_budget_sec and (time.time() - wall_start) >= cfg.wall_budget_sec:
            log.info(f"Wall budget reached after {epoch} epochs.")
            break
        if not cfg.wall_budget_sec and epoch >= cfg.n_epochs:
            break
        t0 = time.time()
        t_gen, t_surrogate, t_oracle, t_ddpp, t_ensemble = 0.0, 0.0, 0.0, 0.0, 0.0
        n_attempted_epoch, n_valid_epoch = 0, 0

        # ── Steps 1-3: Candidate generation and selection ─────────────────
        ensemble_ready = cfg.use_ensemble and len(all_pairs) >= 4
        mean_sigma = float("nan")
        _bandit_new_pairs = None   # set by bandit modes; skips bulk oracle in step 4

        if cfg.selection in ("beam", "thompson_beam") and ensemble_ready:
            # ── Beam search guided by ensemble surrogate ──────────────────
            # beam: deterministic (mu); thompson_beam: stochastic (mu + sigma*eps)
            _t = time.time()
            from genmol.samplers import BeamSearchSampler
            thompson = (cfg.selection == "thompson_beam")
            mean_sigma_ref = [float("nan")]
            _fwd = _make_ensemble_forward_op(ensemble, gen_model,
                mean_sigma_ref, thompson=thompson,
            )
            beam_sampler = BeamSearchSampler(
                ddpp_trainer.model_path,
                forward_op=_fwd,
                beam_width=cfg.beam_width,
                device=str(device),
            )
            selected = beam_sampler.de_novo_generation(
                num_samples=cfg.K,
                softmax_temp=cfg.softmax_temp,
                randomness=cfg.randomness,
                min_add_len=cfg.min_add_len,
            )
            mean_sigma = mean_sigma_ref[0]
            t_gen += time.time() - _t  # beam gen includes surrogate scoring
            n_attempted_epoch = cfg.K  # beam returns only valid
            n_valid_epoch = len(selected)
            log.info(
                f"[epoch {epoch}] {cfg.selection} (ensemble surrogate) → "
                f"{len(selected)} candidates, sigma={mean_sigma:.4f}"
            )
            embeddings = None

        elif cfg.selection == "thompson_mcts" and ensemble_ready:
            # ── MCTS guided by Thompson sample ────────────────────────────
            _t = time.time()
            from genmol.samplers.mcts import MCTSSampler
            mean_sigma_ref = [float("nan")]
            _fwd = _make_ensemble_forward_op(ensemble, gen_model,
                mean_sigma_ref, thompson=True,
            )
            mcts_sampler = MCTSSampler(
                ddpp_trainer.model_path,
                forward_op=_fwd,
                device=str(device),
                rollout_budget_per_sample=cfg.mcts_rollout_budget,
            )
            selected = mcts_sampler.de_novo_generation(
                num_samples=cfg.K,
                softmax_temp=cfg.softmax_temp,
                randomness=cfg.randomness,
                min_add_len=cfg.min_add_len,
            )
            mean_sigma = mean_sigma_ref[0]
            t_gen += time.time() - _t
            n_attempted_epoch = cfg.K  # MCTS returns only valid
            n_valid_epoch = len(selected)
            log.info(
                f"[epoch {epoch}] thompson_mcts → "
                f"{len(selected)} candidates, sigma={mean_sigma:.4f}"
            )
            embeddings = None

        elif cfg.selection in ("bandit_ensemble", "bandit_global", "bandit_ts", "bandit_ts_upper") and ensemble_ready:
            # ── Multi-fidelity bandit ─────────────────────────────────────
            # Generate M candidates
            _t = time.time()
            log.info(f"[epoch {epoch}] Bandit: generating {cfg.M} candidates...")
            gen_chunk = 64
            candidates = []
            n_attempted_epoch, n_valid_epoch = 0, 0
            for _i in range(0, cfg.M, gen_chunk):
                n = min(gen_chunk, cfg.M - _i)
                candidates.extend(ddpp_trainer.generate(
                    n,
                    softmax_temp=cfg.softmax_temp,
                    randomness=cfg.randomness,
                    min_add_len=cfg.min_add_len,
                ))
                n_attempted_epoch += getattr(ddpp_trainer, '_last_n_attempted', n)
                n_valid_epoch += getattr(ddpp_trainer, '_last_n_valid', len(candidates))
            if not candidates:
                log.warning(f"[epoch {epoch}] No candidates generated, skipping.")
                epoch += 1
                continue

            t_gen += time.time() - _t
            # Encode all candidates
            _t = time.time()
            gen_model.set_eval()
            with torch.no_grad():
                embeddings = gen_model.get_embeddings(candidates)
            mu, sigma = ensemble.predict_unnorm(embeddings)
            gen_model.set_train()
            t_surrogate += time.time() - _t
            mean_sigma = sigma.mean().item()

            if cfg.selection in ("bandit_ensemble", "bandit_ts", "bandit_ts_upper"):
                sigma_arr = sigma           # per-candidate uncertainty
            else:
                sigma_arr = torch.full_like(mu, sigma_global)  # global uncertainty

            ucb = mu + cfg.bandit_z * sigma_arr   # [M] — used for elimination in all bandit modes

            # For bandit_ts/bandit_ts_upper: draw Thompson samples once per epoch
            # bandit_ts: standard N(mu, sigma²) draws
            # bandit_ts_upper: half-normal — only upward perturbations (mu + |eps|*sigma)
            if cfg.selection == "bandit_ts":
                ts_samples = mu + sigma_arr * torch.randn_like(mu)
            elif cfg.selection == "bandit_ts_upper":
                ts_samples = mu + sigma_arr * torch.abs(torch.randn_like(mu))
            else:
                ts_samples = None

            active = list(range(len(candidates)))
            best_fa_running = f_star

            bandit_pairs: list[tuple[str, float]] = []
            calls_left = cfg.K

            _t = time.time()
            while calls_left > 0 and active:
                # Selection: rank by UCB or TS samples, pick top candidate
                if cfg.selection in ("bandit_ts", "bandit_ts_upper"):
                    best_i = max(active, key=lambda i: ts_samples[i].item())
                else:
                    best_i = max(active, key=lambda i: ucb[i].item())
                smi = candidates[best_i]
                score = logged_fa_oracle([smi], phase="bandit_oracle")[0]
                active.remove(best_i)
                calls_left -= 1

                if score is not None and not np.isnan(score):
                    bandit_pairs.append((smi, score))
                    best_fa_running = max(best_fa_running, score)

            sigma_desc = (f"{sigma_global:.4f}" if cfg.selection == "bandit_global"
                          else "per-cand (TS-upper)" if cfg.selection == "bandit_ts_upper"
                          else "per-cand (TS)" if cfg.selection == "bandit_ts"
                          else "per-cand (UCB)")
            log.info(
                f"[epoch {epoch}] {cfg.selection}: {cfg.K - calls_left} oracle calls, "
                f"{len(bandit_pairs)} valid, sigma={sigma_desc}"
            )
            t_oracle += time.time() - _t
            _bandit_new_pairs = bandit_pairs
            selected = []

        else:
            # ── Generate M random candidates, then select ─────────────────
            _t = time.time()
            log.info(f"[epoch {epoch}] Generating {cfg.M} candidates...")
            gen_chunk = 64
            candidates = []
            raw_all_epoch = []  # all raw SMILES (None for invalid) for diversity logging
            n_attempted_epoch, n_valid_epoch = 0, 0
            for _i in range(0, cfg.M, gen_chunk):
                n = min(gen_chunk, cfg.M - _i)
                _g = ddpp_trainer.generate(
                    n,
                    softmax_temp=cfg.softmax_temp,
                    randomness=cfg.randomness,
                    min_add_len=cfg.min_add_len,
                )
                candidates.extend(_g)
                raw_all_epoch.extend(getattr(ddpp_trainer, '_last_raw_smiles', _g))
                n_attempted_epoch += getattr(ddpp_trainer, '_last_n_attempted', n)
                n_valid_epoch += getattr(ddpp_trainer, '_last_n_valid', len(candidates))
            if not candidates:
                log.warning(f"[epoch {epoch}] No valid candidates generated, skipping.")
                epoch += 1
                continue
            t_gen += time.time() - _t
            log.info(f"[epoch {epoch}] Got {len(candidates)} valid candidates.")

            # Encode for ensemble (if enabled)
            _t = time.time()
            if cfg.use_ensemble:
                gen_model.set_eval()
                with torch.no_grad():
                    embeddings = gen_model.get_embeddings(candidates)
                gen_model.set_train()
            else:
                embeddings = None

            # Select K candidates
            if ensemble_ready and cfg.selection == "thompson":
                sampled_scores = ensemble.thompson_sample(embeddings)
                selected_idx = _topk_indices(sampled_scores, cfg.K)
                mean_sigma = ensemble.predict(embeddings)[1].mean().item()
            else:
                selected_idx = rng.sample(range(len(candidates)), min(cfg.K, len(candidates)))
                if cfg.use_ensemble:
                    log.info(f"[epoch {epoch}] Cold start: random selection.")

            t_surrogate += time.time() - _t
            selected = [candidates[i] for i in selected_idx]

        # ── Step 4: FA oracle evaluation ──────────────────────────────────
        if _bandit_new_pairs is not None:
            # Bandit modes: already evaluated inline above
            new_pairs = _bandit_new_pairs
        else:
            _t = time.time()
            log.info(f"[epoch {epoch}] Evaluating {len(selected)} candidates with FA oracle...")
            fa_scores = logged_fa_oracle(selected, phase="batch_oracle")
            new_pairs = [
                (smi, score)
                for smi, score in zip(selected, fa_scores)
                if score is not None and not np.isnan(score)
            ]
            t_oracle += time.time() - _t
        all_pairs.extend(new_pairs)

        # Update f* and best SMILES
        for smi, score in new_pairs:
            if score > f_star:
                f_star = score
                best_smiles = smi

        # ── Step 5: Fine-tune backbone with clipped reward ────────────────
        _t = time.time()
        tau = float("nan")
        if all_pairs:
            raw_scores = [s for _, s in all_pairs]
            wall_frac = (time.time() - wall_start) / cfg.wall_budget_sec if cfg.wall_budget_sec else epoch / max(cfg.n_epochs, 1)
            tau = _compute_tau(raw_scores, cfg.tau_mode, cfg.tau_quantile, f_star,
                               epoch=epoch, wall_frac=wall_frac)

            ddpp_trainer.reward_clip_threshold = tau + cfg.reward_shift
            shifted_new = [(smi, score + cfg.reward_shift) for smi, score in new_pairs]
            ddpp_trainer.add_scored_molecules(
                [s for s, _ in shifted_new],
                [r for _, r in shifted_new],
            )
            ddpp_trainer.train(max_steps=cfg.ddpp_steps_per_epoch)
            cumulative_ddpp_steps += cfg.ddpp_steps_per_epoch
            log.info(
                f"[epoch {epoch}] DDPP steps done (step={ddpp_trainer.step}, tau={tau:.3f})"
            )

        t_ddpp += time.time() - _t
        cumulative_diffusion_passes += n_attempted_epoch

        # ── Step 6: Retrain ensemble ──────────────────────────────────────
        _t = time.time()
        rho = float("nan")
        if cfg.use_ensemble and len(all_pairs) >= cfg.n_ensemble:
            all_smiles = [s for s, _ in all_pairs]
            all_fa = torch.tensor([sc for _, sc in all_pairs], dtype=torch.float, device=device)

            gen_model.set_eval()
            with torch.no_grad():
                batch = 256
                emb_list = []
                for i in range(0, len(all_smiles), batch):
                    emb_list.append(
                        gen_model.get_embeddings(all_smiles[i : i + batch])
                    )
                all_emb = torch.cat(emb_list, dim=0)
            gen_model.set_train()

            ensemble.reinit_and_train(
                all_emb, all_fa,
                n_epochs=cfg.ensemble_epochs,
                lr=cfg.ensemble_lr,
                batch_size=cfg.ensemble_batch,
            )

            # Diagnostics: Spearman rho on held-out 20%
            n_held = max(4, int(0.2 * len(all_pairs)))
            held_idx = rng.sample(range(len(all_pairs)), n_held)
            held_emb = all_emb[held_idx]
            held_fa = all_fa[held_idx]
            rho = ensemble.spearman_rho(held_emb, held_fa)

            # Update sigma_global for bandit_global: std(fa - surrogate_mu) on all data
            if cfg.selection == "bandit_global":
                mu_all, _ = ensemble.predict_unnorm(all_emb)
                sigma_global = (all_fa - mu_all).std().clamp(min=1e-4).item()
                log.info(f"[epoch {epoch}] sigma_global updated: {sigma_global:.4f}")

        t_ensemble += time.time() - _t

        # ── Step 7: Log epoch ─────────────────────────────────────────────
        elapsed = time.time() - t0
        epoch_log = {
            "epoch": epoch,
            "f_star": f_star,
            "best_smiles": best_smiles,
            "n_oracle_calls": len(all_pairs),
            "mean_sigma": mean_sigma,
            "spearman_rho": rho,
            "tau": tau if all_pairs else None,
            "elapsed_sec": elapsed,
            "epoch_fa_scores": [sc for _, sc in new_pairs],
            "epoch_fa_smiles": [smi for smi, _ in new_pairs],
            "epoch_all_smiles": [
                {"smiles": _s, "valid": _s is not None,
                 "selected": (_s is not None and _s in set(ss for ss, _ in new_pairs))}
                for _s in (raw_all_epoch if 'raw_all_epoch' in dir() else [])
            ],
            "t_gen": round(t_gen, 3),
            "t_surrogate": round(t_surrogate, 3),
            "t_oracle": round(t_oracle, 3),
            "t_ddpp": round(t_ddpp, 3),
            "t_ensemble": round(t_ensemble, 3),
            "n_attempted": n_attempted_epoch,
            "n_valid": n_valid_epoch,
            "cumulative_diffusion_passes": cumulative_diffusion_passes,
            "cumulative_ddpp_steps": cumulative_ddpp_steps,
            "wall_sec": round(time.time() - wall_start, 3),
        }
        history.append(epoch_log)

        if epoch % cfg.log_every == 0:
            log.info(
                f"[epoch {epoch}] f*={f_star:.4f} | rho={rho:.3f} | "
                f"sigma={mean_sigma:.4f} | oracle_calls={len(all_pairs)} | t={elapsed:.1f}s"
            )

        with open(log_path, "a") as f:
            f.write(json.dumps(epoch_log) + "\n")

        epoch += 1

    # Save final results
    results = {
        "f_star": f_star,
        "best_smiles": best_smiles,
        "history": history,
        "all_pairs": all_pairs,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"Active loop done. Best: {best_smiles} (FA={f_star:.4f})")
    return results
