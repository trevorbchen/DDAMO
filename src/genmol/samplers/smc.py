"""SMC-based samplers for GenMol: Discrete FKC and vanilla SMC.

Implements:
  - DFKCSampler: Discrete Feynman-Kac Correctors (arxiv 2601.10403).
    SMC with modified denoising (annealing or reward-tilted) + importance
    weight correction + ESS-based resampling.
  - SMCSampler: Vanilla particle filter. Standard denoising with periodic
    reward-based resampling.
"""

import math
import os
import random
import warnings
from time import time

import torch
import torch.nn.functional as F

from .base import Sampler, decode_smiles

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def compute_ess(log_weights):
    """Effective sample size from log-space weights. Returns float."""
    # Normalize in log space for stability
    log_w = log_weights - log_weights.max()
    w = torch.exp(log_w)
    w = w / w.sum()
    return 1.0 / (w ** 2).sum().item()


def systematic_resample(log_weights):
    """Systematic resampling. Returns index tensor of size K."""
    K = log_weights.shape[0]
    log_w = log_weights - log_weights.max()
    w = torch.exp(log_w)
    w = w / w.sum()

    cumsum = torch.cumsum(w, dim=0)
    u = (torch.rand(1, device=w.device) + torch.arange(K, device=w.device, dtype=w.dtype)) / K
    indices = torch.searchsorted(cumsum, u).clamp(max=K - 1)
    return indices


def multinomial_resample(log_weights):
    """Multinomial resampling. Returns index tensor of size K."""
    log_w = log_weights - log_weights.max()
    w = torch.exp(log_w)
    w = w / w.sum()
    return torch.multinomial(w, w.shape[0], replacement=True)


# ---------------------------------------------------------------------------
# DFKCSampler
# ---------------------------------------------------------------------------

class DFKCSampler(Sampler):
    """Discrete Feynman-Kac Corrector for MDLM (Hasan et al., arXiv 2601.10403).

    Modes:
      annealing - sample from p_t^β via scaled logits (Eq 15) with
                  Feynman-Kac weight correction (Corollary 3.2, Eq 14).
                  No reward function needed.
      reward    - tilt distribution toward high reward via Δβ_t·r(x)
                  weight increments. Oracle calls gated by score_start
                  and score_interval to avoid wasting budget on early
                  garbage SMILES. Uses accumulated d_beta (telescoping).
    """

    def __init__(
        self,
        path=None,
        forward_op=None,
        num_particles=8,
        mode="reward",
        beta=2.0,
        beta_schedule="linear",
        score_start=0.5,
        score_interval=5,
        ess_threshold=0.5,
        resample_strategy="systematic",
        rollout=True,
        scoring_mode=None,
        fresh_weight=False,
        score_interval_late=None,
        late_phase=0.8,
        elite_ratio=0.0,
        elite_buffer_size=50,
        elite_mask_ratio=0.5,
        seed=None,
        verbose=False,
        model=None,
        **kwargs,
    ):
        super().__init__(path=path, forward_op=forward_op, model=model, **kwargs)
        self.forward_op = forward_op
        self.num_particles = max(int(num_particles), 2)
        self.mode = mode
        self.beta = float(beta)
        self.beta_schedule = beta_schedule
        self.score_start = float(score_start)
        self.score_interval = max(int(score_interval), 1)
        self.score_interval_late = int(score_interval_late) if score_interval_late is not None else None
        self.late_phase = float(late_phase)
        self.rollout = bool(rollout)
        # scoring_mode overrides rollout: "completion" (default), "tweedie", "current"
        if scoring_mode is None:
            scoring_mode = "completion" if self.rollout else "current"
        self.scoring_mode = scoring_mode
        self.fresh_weight = bool(fresh_weight)
        self.elite_ratio = float(elite_ratio)
        self.elite_buffer_size = int(elite_buffer_size)
        self.elite_mask_ratio = float(elite_mask_ratio)
        self.ess_threshold = float(ess_threshold)
        self.resample_strategy = resample_strategy
        self.verbose = bool(verbose)
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        if self.mode == "reward" and self.beta_schedule == "constant":
            warnings.warn(
                "DFKCSampler: mode='reward' with beta_schedule='constant' gives "
                "d_beta=0 at every step, making reward weights uniform. "
                "Use 'linear', 'quadratic', or 'cosine' instead.",
                stacklevel=2,
            )

    # ── Helpers ────────────────────────────────────────────────────

    def _get_beta_t(self, step, num_steps):
        """Scheduled beta value at this step (ramps 1 -> beta)."""
        frac = step / max(num_steps - 1, 1)
        if self.beta_schedule == "constant":
            return self.beta
        elif self.beta_schedule == "cosine":
            return 1.0 + (self.beta - 1.0) * 0.5 * (1 - math.cos(math.pi * frac))
        elif self.beta_schedule == "quadratic":
            # Accelerating: small d_beta early, large d_beta late.
            return 1.0 + (self.beta - 1.0) * frac * frac
        elif self.beta_schedule == "sqrt":
            # Decelerating: large d_beta early, small d_beta late.
            # Front-loads selection pressure, leaves late steps for refinement.
            return 1.0 + (self.beta - 1.0) * math.sqrt(frac)
        else:  # linear
            return 1.0 + (self.beta - 1.0) * frac

    def _get_noise_params(self, step, num_steps, device):
        """Get alpha_t and d_alpha/dt from the MDLM noise schedule."""
        # Map step index to diffusion time: step 0 = t=1 (fully masked)
        t = 1.0 - step / num_steps
        t_tensor = torch.tensor([t], device=device)
        ns = self.mdlm.noise_schedule
        sigma_t = ns.calculate_sigma(t_tensor, device)
        d_sigma_dt = ns.d_dt_sigma(t_tensor, device)
        alpha_t = ns.sigma_to_alpha(sigma_t)           # exp(-sigma)
        d_alpha_dt = -d_sigma_dt * alpha_t              # chain rule
        return alpha_t.item(), d_alpha_dt.item()

    def _annealing_weight_increment(self, logits, x, step, num_steps, beta_t):
        """Compute log-weight increment for annealing mode (Corollary 3.2, Eq 14).

        g_τ(i) = δ_{mi} (∂α_t/∂t)/α_t · Σ_j [p_t(j)/p_t(m) - p_t^β(j)/p_t^β(m)]

        Using Eq 72: p_t(j)/p_t(m) = α_t/(1-α_t) · p(x_0=j|x_t=m)  for j≠m
        Sharpened:  [p_t(j)/p_t(m)]^β = [α_t/(1-α_t)]^β · p(x_0=j|x_t=m)^β
        """
        device = x.device
        alpha_t, d_alpha_dt = self._get_noise_params(step, num_steps, device)

        mask = (x == self.model.mask_index)  # [K, seq]
        if not mask.any():
            return torch.zeros(x.shape[0], device=device)

        # p(x_0=j|x_t=m) from model output (already normalized logprobs)
        log_p_x0 = self.mdlm._subs_parameterization(logits.clone(), x)
        p_x0 = torch.exp(log_p_x0)  # [K, seq, V]

        # Noise schedule ratio c = α_t / (1 - α_t)
        c = alpha_t / max(1.0 - alpha_t, 1e-8)

        # Per-position: Σ_j [c·s_j - c^β·s_j^β] = c·1 - c^β·Σ s_j^β
        sum_s_beta = (p_x0 ** beta_t).sum(dim=-1)       # [K, seq]
        per_pos = c - (c ** beta_t) * sum_s_beta         # [K, seq]

        # Sum over masked positions only
        per_pos = (per_pos * mask.float()).sum(dim=-1)    # [K]

        # Scale by noise rate and dt  (Eq 14: (∂α_t/∂t)/α_t · dt)
        dt = 1.0 / num_steps
        rate = d_alpha_dt / max(alpha_t, 1e-8)           # keep sign (negative)
        g = rate * per_pos * dt
        return g

    def _rollout(self, x, start_step, num_steps, softmax_temp, randomness):
        """Complete denoising from start_step to end. Returns (rolled_x, fp_count)."""
        rolled = x.clone()
        fp = 0
        for j in range(start_step + 1, num_steps):
            attention_mask = (rolled != self.pad_index)
            logits = self.model(rolled, attention_mask)
            fp += rolled.shape[0]
            rolled = self.mdlm.step_confidence(
                logits, rolled, j, num_steps, softmax_temp, randomness)
        return rolled, fp

    def _reward_weight_increment(self, x, step, num_steps, beta_t,
                                 last_scored_step=None,
                                 softmax_temp=0.8, randomness=0.5):
        """Compute log-weight increment for reward-tilted mode.

        Rolls out particles to completion before scoring so that the
        reward function always sees fully-denoised, valid SMILES.
        Uses accumulated d_beta (telescoping property).

        Returns (log_weight_increment, n_evals, best_reward, smiles,
                 rewards_list, rollout_fp).
        """
        if self.forward_op is None:
            return torch.zeros(x.shape[0], device=x.device), 0, float("-inf"), [], [], 0

        # Score molecules: completion (rollout), tweedie (1 forward), or current (as-is)
        rollout_fp = 0
        if self.scoring_mode == "tweedie":
            scored_x, rollout_fp = self._tweedie_x0(x)
        elif self.scoring_mode == "completion":
            # Deterministic rollout for consistent scoring —
            # main loop keeps normal randomness for diversity
            scored_x, rollout_fp = self._rollout(x, step, num_steps,
                                                 softmax_temp, randomness=0)
        else:  # "current"
            scored_x = x
        smiles = decode_smiles(self.model, scored_x)
        rewards = self.forward_op(smiles)  # [K]
        rewards = rewards.to(x.device)

        best_r = rewards[rewards.isfinite()].max().item() if rewards.isfinite().any() else float("-inf")
        rewards_list = rewards.tolist()

        # Clamp -inf to a large negative value for weight stability
        rewards = rewards.clamp(min=-100.0)

        # Accumulated delta_beta since last scoring step
        if last_scored_step is not None:
            prev_beta = self._get_beta_t(last_scored_step, num_steps)
        else:
            prev_beta = self._get_beta_t(max(step - 1, 0), num_steps)
        d_beta = beta_t - prev_beta
        g = d_beta * rewards

        if self.verbose:
            n_valid = sum(1 for s in smiles if s)
            print(f"    score step {step}/{num_steps}: d_beta={d_beta:.4f} "
                  f"valid={n_valid}/{len(smiles)} best_r={best_r:.4f} "
                  f"rollout_fp={rollout_fp}")

        return g, len(smiles), best_r, smiles, rewards_list, rollout_fp

    def _encode_and_remask(self, smiles_list, min_add_len=40):
        """Encode SMILES back to token sequences and partially re-mask.

        Used for elite seeding: take known good molecules, re-mask a fraction
        of their tokens, and use as warm-start particles.
        Returns token tensor [N, seq_len] or None if encoding fails.
        """
        import safe as sf
        from genmol.utils.utils_chem import safe_to_smiles
        from genmol.utils.bracket_safe_converter import BracketSAFEConverter, bracketsafe2safe

        encoded = []
        for smi in smiles_list:
            try:
                if self.model.config.training.get("use_bracket_safe"):
                    safe_str = BracketSAFEConverter(slicer=self.slicer).encoder(smi, allow_empty=True)
                else:
                    safe_str = sf.SAFEConverter(slicer=self.slicer, ignore_stereo=True).encoder(smi, allow_empty=True)
                ids = self.model.tokenizer(
                    [safe_str], return_tensors='pt', truncation=True,
                    max_length=self.model.config.model.max_position_embeddings
                )['input_ids'][0]
                encoded.append(ids)
            except Exception:
                continue

        if not encoded:
            return None

        # Pad to same length
        max_len = max(len(e) for e in encoded)
        max_len = max(max_len, min_add_len + 2)
        padded = []
        for ids in encoded:
            pad_len = max_len - len(ids)
            if pad_len > 0:
                ids = torch.cat([ids, torch.full((pad_len,), self.pad_index)])
            padded.append(ids)
        x = torch.stack(padded)

        # Randomly mask a fraction of non-special tokens
        for i in range(x.shape[0]):
            # Find maskable positions (not BOS, EOS, PAD)
            maskable = (
                (x[i] != self.model.bos_index) &
                (x[i] != self.model.eos_index) &
                (x[i] != self.pad_index) &
                (x[i] != self.model.mask_index)
            )
            mask_positions = maskable.nonzero(as_tuple=True)[0]
            if len(mask_positions) == 0:
                continue
            n_mask = max(1, int(len(mask_positions) * self.elite_mask_ratio))
            chosen = mask_positions[torch.randperm(len(mask_positions))[:n_mask]]
            x[i, chosen] = self.model.mask_index

        return x

    def _resample(self, x, log_weights):
        """Resample particles and reset weights."""
        if self.resample_strategy == "multinomial":
            indices = multinomial_resample(log_weights)
        else:
            indices = systematic_resample(log_weights)
        x = x[indices].clone()
        log_weights = torch.zeros_like(log_weights)
        return x, log_weights

    # ── Main generation ────────────────────────────────────────────

    @torch.no_grad()
    def de_novo_generation(self, num_samples=1, softmax_temp=0.8,
                           randomness=0.5, min_add_len=40, timeout_sec=None,
                           max_reward_evals=None, **kwargs):
        K = self.num_particles
        device = self.model.device
        n_rounds = math.ceil(num_samples / K)

        self._reset_trajectory()

        all_smiles = []
        all_weights = []
        total_fp = 0
        total_reward_evals = 0
        elite_buffer = []  # list of (score, smiles) sorted desc
        t0 = time()

        for rnd in range(n_rounds):
            if timeout_sec is not None and time() - t0 > timeout_sec:
                break
            if max_reward_evals is not None and total_reward_evals >= max_reward_evals:
                break
            # Initialize K particles: some from elite buffer, rest from scratch
            n_elite = 0
            elite_x = None
            if self.elite_ratio > 0 and elite_buffer:
                n_elite = min(int(K * self.elite_ratio), len(elite_buffer))
                elite_smiles = [smi for _, smi in elite_buffer[:n_elite]]
                elite_x = self._encode_and_remask(elite_smiles, min_add_len)
                if elite_x is not None:
                    n_elite = elite_x.shape[0]
                else:
                    n_elite = 0

            n_fresh = K - n_elite
            x_proto = torch.hstack([
                torch.full((1, 1), self.model.bos_index),
                torch.full((1, 1), self.model.eos_index),
            ])
            x_fresh = self._insert_mask(x_proto, n_fresh, min_add_len=min_add_len)

            if n_elite > 0 and elite_x is not None:
                # Pad to same seq length and concatenate
                max_len = max(x_fresh.shape[1], elite_x.shape[1])
                if x_fresh.shape[1] < max_len:
                    x_fresh = torch.cat([x_fresh, torch.full(
                        (x_fresh.shape[0], max_len - x_fresh.shape[1]), self.pad_index)], dim=1)
                if elite_x.shape[1] < max_len:
                    elite_x = torch.cat([elite_x, torch.full(
                        (elite_x.shape[0], max_len - elite_x.shape[1]), self.pad_index)], dim=1)
                x = torch.cat([elite_x, x_fresh], dim=0)
            else:
                x = x_fresh
            x = x.to(device)

            num_steps = max(self.mdlm.get_num_steps_confidence(x), 2)
            log_weights = torch.zeros(K, device=device)
            last_scored_step = 0  # accumulate d_beta from step 0

            for i in range(num_steps):
                if max_reward_evals is not None and total_reward_evals >= max_reward_evals:
                    break
                beta_t = self._get_beta_t(i, num_steps)
                progress = i / num_steps

                # Forward pass
                attention_mask = (x != self.pad_index)
                logits = self.model(x, attention_mask)  # [K, seq, V]
                total_fp += K

                # Should we call the oracle this step?
                # Use denser scoring in late phase if score_interval_late is set
                interval = self.score_interval
                if self.score_interval_late is not None and progress >= self.late_phase:
                    interval = self.score_interval_late
                should_score = (
                    self.mode == "reward"
                    and self.forward_op is not None
                    and progress >= self.score_start
                    and i % interval == 0
                )

                # ── Weight update ──────────────────────────────
                if self.mode == "annealing":
                    log_weights += self._annealing_weight_increment(
                        logits, x, i, num_steps, beta_t)
                    self._log_point(0, K, float("-inf"))

                elif self.mode == "reward" and should_score:
                    g, n_evals, best_r, smiles_batch, scores_batch, rfp = \
                        self._reward_weight_increment(
                            x, i, num_steps, beta_t, last_scored_step,
                            softmax_temp, randomness)
                    if self.fresh_weight:
                        raw_rewards = torch.tensor(scores_batch, device=device)
                        raw_rewards = raw_rewards.clamp(min=-100.0)
                        log_weights = beta_t * raw_rewards
                    else:
                        log_weights += g
                    total_reward_evals += n_evals
                    total_fp += rfp
                    last_scored_step = i
                    self._log_point(n_evals, K, best_r,
                                    smiles_batch=smiles_batch,
                                    scores_batch=scores_batch)
                    # Update elite buffer
                    if self.elite_ratio > 0:
                        for smi, sc in zip(smiles_batch, scores_batch):
                            if smi and sc > -99:
                                elite_buffer.append((sc, smi))
                        elite_buffer.sort(reverse=True)
                        # Deduplicate, keep top N
                        seen_elite = set()
                        deduped = []
                        for sc, smi in elite_buffer:
                            if smi not in seen_elite:
                                seen_elite.add(smi)
                                deduped.append((sc, smi))
                            if len(deduped) >= self.elite_buffer_size:
                                break
                        elite_buffer = deduped

                # ── Modified logits for denoising step ─────────
                if self.mode == "annealing":
                    modified_logits = logits * beta_t
                else:
                    modified_logits = logits

                x = self.mdlm.step_confidence(
                    modified_logits, x, i, num_steps,
                    softmax_temp, randomness)

                # Numerical stability: shift log weights
                log_weights -= log_weights.max()

                # ESS-based resampling
                ess = compute_ess(log_weights)
                if ess < self.ess_threshold * K:
                    x, log_weights = self._resample(x, log_weights)
                    last_scored_step = i  # accumulate from resample point
                    if self.verbose:
                        print(f"  step {i}/{num_steps}: ESS={ess:.1f}, resampled")

            # Collect results from this round
            smiles = decode_smiles(self.model, x)
            for smi, lw in zip(smiles, log_weights.tolist()):
                all_smiles.append(smi)
                all_weights.append(lw)

        # Deduplicate and sort by weight
        seen = {}
        for smi, w in zip(all_smiles, all_weights):
            if smi and (smi not in seen or w > seen[smi]):
                seen[smi] = w
        result = sorted(seen.keys(), key=lambda s: seen[s], reverse=True)

        # Track budget
        self.last_forward_passes = total_fp
        self.last_fp_per_sample = total_fp / max(num_samples, 1)
        self.last_reward_evals = total_reward_evals
        self.last_budget_per_sample = total_reward_evals / max(num_samples, 1)

        return result[:num_samples]


# ---------------------------------------------------------------------------
# SMCSampler (vanilla particle filter)
# ---------------------------------------------------------------------------

class SMCSampler(Sampler):
    """Vanilla SMC sampler: standard denoising + periodic reward resampling.

    Propagates K particles with the unmodified MDLM denoiser. Every
    ``resample_interval`` steps (starting at ``resample_start`` fraction
    of denoising), decodes particles to SMILES, scores them with
    ``forward_op``, and resamples proportional to exp(beta * reward).
    """

    def __init__(
        self,
        path=None,
        forward_op=None,
        num_particles=8,
        resample_interval=5,
        resample_start=0.5,
        beta=10.0,
        ess_threshold=0.5,
        resample_strategy="systematic",
        seed=None,
        verbose=False,
        model=None,
        **kwargs,
    ):
        super().__init__(path=path, forward_op=forward_op, model=model, **kwargs)
        self.forward_op = forward_op
        self.num_particles = max(int(num_particles), 2)
        self.resample_interval = max(int(resample_interval), 1)
        self.resample_start = float(resample_start)
        self.beta = float(beta)
        self.ess_threshold = float(ess_threshold)
        self.resample_strategy = resample_strategy
        self.verbose = bool(verbose)
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

    @torch.no_grad()
    def de_novo_generation(self, num_samples=1, softmax_temp=0.8,
                           randomness=0.5, min_add_len=40, timeout_sec=None,
                           max_reward_evals=None, **kwargs):
        K = self.num_particles
        device = self.model.device
        n_rounds = math.ceil(num_samples / K)

        self._reset_trajectory()
        last_logged_fp = 0

        all_smiles = []
        all_rewards = []
        total_fp = 0
        total_reward_evals = 0
        t0 = time()

        for _ in range(n_rounds):
            if timeout_sec is not None and time() - t0 > timeout_sec:
                break
            if max_reward_evals is not None and total_reward_evals >= max_reward_evals:
                break
            # Initialize K particles
            x_proto = torch.hstack([
                torch.full((1, 1), self.model.bos_index),
                torch.full((1, 1), self.model.eos_index),
            ])
            x = self._insert_mask(x_proto, K, min_add_len=min_add_len)
            x = x.to(device)

            num_steps = max(self.mdlm.get_num_steps_confidence(x), 2)

            for i in range(num_steps):
                if max_reward_evals is not None and total_reward_evals >= max_reward_evals:
                    break
                # Standard forward pass + denoising
                attention_mask = (x != self.pad_index)
                logits = self.model(x, attention_mask)
                total_fp += K

                x = self.mdlm.step_confidence(
                    logits, x, i, num_steps, softmax_temp, randomness)

                # Periodic reward-based resampling
                progress = i / num_steps
                if (self.forward_op is not None
                        and progress >= self.resample_start
                        and i % self.resample_interval == 0):

                    smiles = decode_smiles(self.model, x)
                    rewards = self.forward_op(smiles).to(device)
                    total_reward_evals += K

                    best_r = rewards[rewards.isfinite()].max().item() if rewards.isfinite().any() else float("-inf")
                    self._log_point(K, total_fp - last_logged_fp, best_r,
                                    smiles_batch=smiles, scores_batch=rewards.tolist())
                    last_logged_fp = total_fp

                    # Replace -inf with minimum finite reward
                    finite = rewards.isfinite()
                    if finite.any():
                        rewards = rewards.clamp(min=rewards[finite].min().item())
                    else:
                        continue  # all invalid, skip resampling

                    log_w = self.beta * rewards
                    ess = compute_ess(log_w)

                    if ess < self.ess_threshold * K:
                        if self.resample_strategy == "multinomial":
                            indices = multinomial_resample(log_w)
                        else:
                            indices = systematic_resample(log_w)
                        x = x[indices].clone()
                        if self.verbose:
                            print(f"  step {i}/{num_steps}: ESS={ess:.1f}, "
                                  f"resampled, best_r={rewards.max():.3f}")

            # Collect results
            smiles = decode_smiles(self.model, x)
            # Skip end-of-round scoring if budget exhausted -- particles
            # were already scored at the last in-loop resample, and another
            # K calls here would overshoot the configured budget.
            budget_exhausted = (
                max_reward_evals is not None
                and total_reward_evals >= max_reward_evals
            )
            if self.forward_op is not None and not budget_exhausted:
                rewards = self.forward_op(smiles).to(device)
                total_reward_evals += K
                best_r = rewards[rewards.isfinite()].max().item() if rewards.isfinite().any() else float("-inf")
                self._log_point(K, total_fp - last_logged_fp, best_r,
                                smiles_batch=smiles, scores_batch=rewards.tolist())
                last_logged_fp = total_fp
            else:
                rewards = torch.zeros(K, device=device)

            for smi, r in zip(smiles, rewards.tolist()):
                all_smiles.append(smi)
                all_rewards.append(r)

        # Deduplicate, sort by reward
        seen = {}
        for smi, r in zip(all_smiles, all_rewards):
            if smi and (smi not in seen or r > seen[smi]):
                seen[smi] = r
        result = sorted(seen.keys(), key=lambda s: seen[s], reverse=True)

        # Track budget
        self.last_forward_passes = total_fp
        self.last_fp_per_sample = total_fp / max(num_samples, 1)
        self.last_reward_evals = total_reward_evals
        self.last_budget_per_sample = total_reward_evals / max(num_samples, 1)

        return result[:num_samples]
