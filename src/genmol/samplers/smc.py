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

    Implements Algorithm 1 with two modes:
      annealing - sample from p_t^β via scaled logits (Eq 15) with
                  Feynman-Kac weight correction (Corollary 3.2, Eq 14).
                  No reward function needed.
      reward    - tilt distribution toward high reward via Δβ_t·r(x)
                  weight increments (approximation of Corollary 3.6, Eq 20;
                  exact per-token formulation is O(V·d) per step).
                  Requires forward_op.
    """

    def __init__(
        self,
        path,
        forward_op=None,
        num_particles=8,
        mode="reward",
        beta=2.0,
        beta_schedule="linear",
        ess_threshold=0.5,
        resample_strategy="systematic",
        seed=None,
        verbose=False,
        **kwargs,
    ):
        super().__init__(path, forward_op=forward_op, **kwargs)
        self.forward_op = forward_op
        self.num_particles = max(int(num_particles), 2)
        self.mode = mode
        self.beta = float(beta)
        self.beta_schedule = beta_schedule
        self.ess_threshold = float(ess_threshold)
        self.resample_strategy = resample_strategy
        self.verbose = bool(verbose)
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

    # ── Helpers ────────────────────────────────────────────────────

    def _get_beta_t(self, step, num_steps):
        """Scheduled beta value at this step (ramps 1 -> beta)."""
        frac = step / max(num_steps - 1, 1)
        if self.beta_schedule == "constant":
            return self.beta
        elif self.beta_schedule == "cosine":
            return 1.0 + (self.beta - 1.0) * 0.5 * (1 - math.cos(math.pi * frac))
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

    def _reward_weight_increment(self, x, step, num_steps, beta_t):
        """Compute log-weight increment for reward-tilted mode.

        Approximation: decode particles, score with forward_op, scale
        by delta_beta_t. The exact per-token approach is O(V*d) reward
        evals per step -- impractical for molecules.
        """
        if self.forward_op is None:
            return torch.zeros(x.shape[0], device=x.device), 0, float("-inf")

        smiles = decode_smiles(self.model, x)
        rewards = self.forward_op(smiles)  # [K]
        rewards = rewards.to(x.device)

        best_r = rewards[rewards.isfinite()].max().item() if rewards.isfinite().any() else float("-inf")

        # Clamp -inf to a large negative value for weight stability
        rewards = rewards.clamp(min=-100.0)

        # delta_beta * r(x)
        dt = 1.0 / num_steps
        prev_beta = self._get_beta_t(max(step - 1, 0), num_steps)
        d_beta = beta_t - prev_beta
        g = d_beta * rewards
        return g, len(smiles), best_r

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
                           randomness=0.5, min_add_len=40, **kwargs):
        K = self.num_particles
        device = self.model.device
        n_rounds = math.ceil(num_samples / K)

        self._reset_trajectory()

        all_smiles = []
        all_weights = []
        total_fp = 0
        total_reward_evals = 0

        for _ in range(n_rounds):
            # Initialize K particles from fully masked sequences
            x_proto = torch.hstack([
                torch.full((1, 1), self.model.bos_index),
                torch.full((1, 1), self.model.eos_index),
            ])
            x = self._insert_mask(x_proto, K, min_add_len=min_add_len)
            x = x.to(device)

            num_steps = max(self.mdlm.get_num_steps_confidence(x), 2)
            log_weights = torch.zeros(K, device=device)

            for i in range(num_steps):
                beta_t = self._get_beta_t(i, num_steps)

                # Forward pass
                attention_mask = (x != self.pad_index)
                logits = self.model(x, attention_mask)  # [K, seq, V]
                total_fp += K

                # Weight update (before modifying logits)
                if self.mode == "annealing":
                    log_weights += self._annealing_weight_increment(
                        logits, x, i, num_steps, beta_t)
                    self._log_point(0, K, float("-inf"))
                elif self.mode == "reward":
                    g, n_evals, best_r = self._reward_weight_increment(
                        x, i, num_steps, beta_t)
                    log_weights += g
                    total_reward_evals += n_evals
                    self._log_point(n_evals, K, best_r)

                # Modified state update
                if self.mode == "annealing":
                    modified_logits = logits * beta_t
                elif self.mode == "reward" and self.forward_op is not None:
                    modified_logits = logits  # reward guidance via weights only
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
        path,
        forward_op=None,
        num_particles=8,
        resample_interval=5,
        resample_start=0.5,
        beta=10.0,
        ess_threshold=0.5,
        resample_strategy="systematic",
        seed=None,
        verbose=False,
        **kwargs,
    ):
        super().__init__(path, forward_op=forward_op, **kwargs)
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
                           randomness=0.5, min_add_len=40, **kwargs):
        K = self.num_particles
        device = self.model.device
        n_rounds = math.ceil(num_samples / K)

        self._reset_trajectory()
        last_logged_fp = 0

        all_smiles = []
        all_rewards = []
        total_fp = 0
        total_reward_evals = 0

        for _ in range(n_rounds):
            # Initialize K particles
            x_proto = torch.hstack([
                torch.full((1, 1), self.model.bos_index),
                torch.full((1, 1), self.model.eos_index),
            ])
            x = self._insert_mask(x_proto, K, min_add_len=min_add_len)
            x = x.to(device)

            num_steps = max(self.mdlm.get_num_steps_confidence(x), 2)

            for i in range(num_steps):
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
                    self._log_point(K, total_fp - last_logged_fp, best_r)
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
            if self.forward_op is not None:
                rewards = self.forward_op(smiles).to(device)
                total_reward_evals += K
                best_r = rewards[rewards.isfinite()].max().item() if rewards.isfinite().any() else float("-inf")
                self._log_point(K, total_fp - last_logged_fp, best_r)
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
