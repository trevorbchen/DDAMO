"""VIDD: Iterative Distillation for Reward-Guided Fine-Tuning.

Implements Wang et al., "Iterative Distillation for Reward-Guided Fine-Tuning
of Diffusion Models in Biomolecular Design" (arXiv 2507.00445).

Finetunes a GenMol masked diffusion model via policy distillation with
scheduled roll-in. Unlike DRAKES, VIDD does NOT require differentiable rewards,
making it directly compatible with FA and Boltz oracles.

Design space: VIDD is the simulation-based counterpart to DDPP-LB
(simulation-free) for non-differentiable rewards.

Four loss variants:
    - kl      : reward-weighted MLE at intermediate state x_s (default)
    - rw_mle  : reward-weighted MLE at x_0
    - ddpo    : PPO-style policy gradient
    - ddpp    : match reward differences to log-prob differences

Usage:
    from genmol.finetune import VIDDTrainer
    from genmol.rewards import get_reward

    trainer = VIDDTrainer(
        model_path="model_v2.ckpt",
        reward_fn=get_reward("qed"),
        loss_func="kl",
        teacher_alpha=1.0,
        gkd_lmbda=0.5,
    )
    trainer.train(num_steps=500, max_oracle_calls=800)
    trainer.save("vidd_finetuned.pt")
"""

import copy
import logging
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from genmol.samplers.base import load_model_from_path
from genmol.utils.utils_chem import safe_to_smiles
from genmol.utils.bracket_safe_converter import bracketsafe2safe

# VIDD's ReplayBuffer (minimal, self-contained — different API from DDPP-LB's
# ReplayBuffer in ddpp.py, kept local to avoid coupling).
from collections import deque
import random as _random


class ReplayBuffer:
    """Minimal FIFO buffer for VIDD: stores tokenised molecules + rewards."""

    def __init__(self, max_size=10_000):
        self.buffer = deque(maxlen=max_size)

    def add_batch(self, token_ids, attention_masks, smiles_list, rewards):
        for i in range(token_ids.shape[0]):
            self.buffer.append({
                "token_ids": token_ids[i].cpu(),
                "attention_mask": attention_masks[i].cpu(),
                "smiles": smiles_list[i],
                "reward": rewards[i].item(),
            })

    def sample(self, batch_size):
        items = _random.sample(list(self.buffer),
                               min(batch_size, len(self.buffer)))
        max_len = max(it["token_ids"].shape[0] for it in items)
        padded_ids, padded_mask = [], []
        for it in items:
            ids = it["token_ids"]
            mask = it["attention_mask"]
            pad_n = max_len - ids.shape[0]
            if pad_n > 0:
                ids = torch.cat([ids, ids.new_zeros(pad_n)])
                mask = torch.cat([mask, mask.new_zeros(pad_n)])
            padded_ids.append(ids)
            padded_mask.append(mask)
        token_ids = torch.stack(padded_ids)
        att_mask = torch.stack(padded_mask)
        smiles = [it["smiles"] for it in items]
        rewards = torch.tensor([it["reward"] for it in items],
                               dtype=torch.float32)
        return token_ids, att_mask, smiles, rewards

    def __len__(self):
        return len(self.buffer)

logger = logging.getLogger(__name__)


# ── VIDD Trainer ──────────────────────────────────────────────────────


class VIDDTrainer:
    """Fine-tune a GenMol MDM via VIDD iterative distillation.

    Uses three models: pretrained (frozen baseline), old_model (reference
    policy, periodically synced), and finetuned (trainable student).

    Interface is deliberately aligned with DDPPLBTrainer for drop-in
    replacement in CAT 5/7/8 experiment scripts.
    """

    def __init__(
        self,
        model_path: str,
        reward_fn,
        *,
        # === Shared with DDPP-LB ===
        lr: float = 1e-5,
        batch_size: int = 32,
        replay_buffer_size: int = 10_000,
        sampling_eps: float = 1e-3,
        initial_buffer_from_pretrained: int = 64,
        refill_interval: int = 0,   # VIDD generates on-policy, default no refill
        refill_batch_size: int = 16,
        softmax_temp: float = 0.8,
        randomness: float = 0.5,
        min_add_len: int = 60,
        ema_decay: float = 0.9999,
        seed: int | None = 0,
        verbose: bool = False,
        # === VIDD-specific ===
        loss_func: str = "rw_mle",      # "rw_mle" (cheap, default) | "kl" | "ddpo" | "ddpp"
        teacher_alpha: float = 1.0,
        reward_norm: str = "none",      # "none" (default; CVaR+raw) | "pos" | "normal"
        gkd_lmbda: float = 0.0,         # default: pure pretrain roll-in (matches DDPP oracle cost)
        use_schedule_rollin: bool = False,
        schedule_max_step: int = 1000,
        old_roll_in: bool = True,
        target_update_interval: int = 20,
        timesteps_per_epoch: int = 4,
        grad_clip: float = 1.0,
        ratio_clip: float = 1e-4,
        reward_shift: float = 0.0,
    ):
        """Initialize VIDD trainer.

        Default config (rw_mle + gkd_lmbda=0) matches DDPP-LB's oracle-call
        pattern exactly: all oracle calls come from initial buffer fill and
        optional periodic refill. This is suitable for tight budgets (Boltz).

        For larger budgets (FA ~100k), enable:
            --loss_func kl    (adds batch_size * timesteps_per_epoch calls/step for xs)
            --gkd_lmbda 0.5   (adds batch_size calls/step for on-policy x0)
        """
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path  # needed for active_loop compatibility

        # ── models ──────────────────────────────────────────────────
        logger.info("Loading pre-trained model (frozen) …")
        self.pretrained = load_model_from_path(model_path)
        self.pretrained.backbone.eval()
        for p in self.pretrained.parameters():
            p.requires_grad_(False)
        self.pretrained.to(self.device)

        logger.info("Loading fine-tuned model (trainable student) …")
        self.finetuned = load_model_from_path(model_path)
        self.finetuned.backbone.train()
        self.finetuned.to(self.device)

        logger.info("Loading reference policy (old_model, frozen between syncs) …")
        self.old_model = load_model_from_path(model_path)
        self.old_model.backbone.eval()
        for p in self.old_model.parameters():
            p.requires_grad_(False)
        self.old_model.to(self.device)

        # shared references
        self.mdlm = self.pretrained.mdlm
        self.mdlm.to_device(self.device)
        self.tokenizer = self.pretrained.tokenizer
        self.mask_idx = self.pretrained.mask_index
        self.pad_idx = self.tokenizer.pad_token_id
        self.bos_idx = self.pretrained.bos_index
        self.eos_idx = self.pretrained.eos_index
        self.vocab_size = self.pretrained.backbone.bert.embeddings.word_embeddings.num_embeddings
        self.max_len = self.pretrained.config.model.max_position_embeddings
        self.use_bracket_safe = self.pretrained.config.training.get(
            "use_bracket_safe", False
        )

        # ── reward function ─────────────────────────────────────────
        self.reward_fn = reward_fn

        # ── replay buffer (for pretrain roll-in data) ───────────────
        self.replay_buffer = ReplayBuffer(max_size=replay_buffer_size)
        self.buffer = self.replay_buffer  # alias for cross-API compat

        # ── optimizer ───────────────────────────────────────────────
        self.optimizer = Adam(self.finetuned.backbone.parameters(), lr=lr)

        # ── EMA for fine-tuned backbone ─────────────────────────────
        self.ema_decay = ema_decay
        self.ema_params = {
            n: p.data.clone()
            for n, p in self.finetuned.backbone.named_parameters()
        }

        # ── shared hparams ──────────────────────────────────────────
        self.batch_size = batch_size
        self.sampling_eps = sampling_eps
        self.initial_buffer_from_pretrained = initial_buffer_from_pretrained
        self.refill_interval = refill_interval
        self.refill_batch_size = refill_batch_size
        self.softmax_temp = softmax_temp
        self.randomness = randomness
        self.min_add_len = min_add_len
        self.verbose = verbose
        self.global_step = 0

        # ── VIDD-specific hparams ───────────────────────────────────
        assert loss_func in ("kl", "rw_mle", "ddpo", "ddpp"), \
            f"Unknown loss_func: {loss_func}"
        self.loss_func = loss_func
        self.teacher_alpha = teacher_alpha
        assert reward_norm in ("none", "pos", "normal", "top_k")
        self.reward_norm = reward_norm
        # K for reward_norm='top_k' — caller can override after construction.
        self.reward_norm_top_k = 10
        self.gkd_lmbda = gkd_lmbda
        self.use_schedule_rollin = use_schedule_rollin
        self.schedule_max_step = schedule_max_step
        self.old_roll_in = old_roll_in
        self.target_update_interval = target_update_interval
        self.timesteps_per_epoch = timesteps_per_epoch
        self.grad_clip = grad_clip
        self.ratio_clip = ratio_clip
        self.reward_shift = reward_shift

        # ── trajectory / checkpoints (populated by train()) ─────────
        self.trajectory = []
        self.checkpoints = {}
        self.cum_oracle_calls = 0
        self.cum_backward = 0
        # active_loop compatibility (mirror DDPP-LB)
        self.reward_clip_threshold = float("-inf")
        # Approach A: negative-score exploration. Same semantics as DDPP-LB:
        # at gradient time, log_r_eff = log_r - gamma * log p_theta(x).detach().
        # In VIDD's weight space (v = exp(r/alpha)), this becomes
        #   v_eff = v * exp(-gamma * log_p_theta.detach() / alpha).
        self.exploration_approach = "none"   # "none" | "A"
        self.exploration_gamma = 0.0
        self.negscore_estimator: str = "rough"
        self.negscore_n_mc: int = 1

    # ── helpers (identical to DDPP) ─────────────────────────────────

    def _compute_denoising_log_prob(self, model, x0, xt):
        """log q(x₀ | xₜ) summed over masked positions."""
        attention_mask = (xt != self.pad_idx).long()
        logits = model(xt, attention_mask)                    # [B, L, V]
        log_probs = F.log_softmax(logits, dim=-1)             # [B, L, V]
        lp = log_probs.gather(2, x0.clamp(0, log_probs.shape[-1] - 1).unsqueeze(-1)).squeeze(-1)  # [B, L]
        mask = (xt == self.mask_idx).float()
        return (lp * mask).sum(dim=1)                          # [B]

    def _compute_denoising_log_prob_alg3(self, model, x0, n_mc):
        """LLaDA Alg-3 unbiased conditional log-likelihood estimator (L/l weighted)."""
        import torch
        B = x0.shape[0]
        attn_mask = (x0 != self.pad_idx).long()
        valid_L = attn_mask.sum(dim=1).clamp(min=1).float()
        accum = torch.zeros(B, device=x0.device)
        for _ in range(int(n_mc)):
            t = self.mdlm.sample_time(B)
            xt = self.mdlm.forward_process(x0, t)
            masked_count = ((xt == self.mask_idx).float()
                            * attn_mask.float()).sum(dim=1).clamp(min=1)
            lp = self._compute_denoising_log_prob(model, x0, xt)
            accum = accum + (valid_L / masked_count) * lp
        return accum / float(n_mc)

    def _decode_tokens(self, x):
        """Token tensor → list of SMILES (None for failures)."""
        strs = self.tokenizer.batch_decode(x, skip_special_tokens=True)
        out = []
        for s in strs:
            if not s:
                out.append(None)
                continue
            if self.use_bracket_safe:
                try:
                    smi = safe_to_smiles(bracketsafe2safe(s), fix=True)
                except Exception:
                    smi = safe_to_smiles(s, fix=True)
            else:
                smi = safe_to_smiles(s, fix=True)
            if smi:
                smi = sorted(smi.split("."), key=len)[-1]
            out.append(smi)
        return out

    @torch.no_grad()

    # ── FinetuneTrainer interface ─────────────────────────────────────────

    @property
    def model(self):
        """Return a GenerativeModel view of the fine-tuned backbone (cached)."""
        if not hasattr(self, "_gen_model_cache"):
            from genmol.genmol_model import GenMolGenerativeModel
            object.__setattr__(
                self, "_gen_model_cache",
                GenMolGenerativeModel(self.finetuned, self.tokenizer, str(self.device), self.max_len)
            )
        return self._gen_model_cache

    @property
    def step(self):
        return getattr(self, "_train_step", 0)

    def add_scored_molecules(self, smiles_list, raw_scores):
        """Add externally scored molecules to the replay buffer.

        Same signature as DDPPLBTrainer.add_scored_molecules so active_loop
        and run_ddpp_search can use VIDDTrainer as a drop-in replacement.
        """
        import safe as sf

        if not isinstance(raw_scores, torch.Tensor):
            raw_scores = torch.tensor(raw_scores, dtype=torch.float32)

        safe_strings = []
        valid_idx = []
        for i, smi in enumerate(smiles_list):
            if smi is None or raw_scores[i].item() <= -99:
                continue
            try:
                safe_str = sf.encode(smi)
            except Exception:
                continue
            # sf.encode can return None / empty / non-string for pathological
            # inputs. Filter so the tokenizer never sees a non-string.
            if not isinstance(safe_str, str) or len(safe_str) == 0:
                continue
            safe_strings.append(safe_str)
            valid_idx.append(i)

        if not safe_strings:
            return 0

        encoded = self.tokenizer(
            safe_strings, return_tensors="pt", padding=True,
            truncation=True, max_length=self.max_len,
        )
        token_ids = encoded["input_ids"].to(self.device)
        att_masks = encoded["attention_mask"].to(self.device)

        scores = raw_scores[valid_idx]
        # CVaR: clip from below at reward_clip_threshold so low-reward samples
        # contribute small-but-uniform gradient instead of being effectively zero.
        # Mirrors DDPP-LB's reward_clip_threshold semantics.
        if self.reward_clip_threshold > float("-inf"):
            scores = torch.clamp(scores, min=self.reward_clip_threshold)
        # Keep raw scores (VIDD normalizes differently than DDPP)
        valid = torch.isfinite(scores)
        n_valid = int(valid.sum())
        if n_valid > 0:
            self.replay_buffer.add_batch(
                token_ids[valid], att_masks[valid],
                [smiles_list[valid_idx[j]] for j in range(len(valid_idx)) if valid[j]],
                scores[valid],
            )
        logger.info("add_scored_molecules: %d/%d added (buffer=%d)",
                    n_valid, len(smiles_list), len(self.replay_buffer))
        return n_valid

    @torch.no_grad()
    def _generate_tokens(self, model, num_samples):
        """Confidence-based denoising to get clean tokens."""
        was_training = model.backbone.training
        model.backbone.eval()
        # build fully-masked input: [BOS] [MASK…] [EOS]
        seqs = []
        for _ in range(num_samples):
            add_len = self.min_add_len
            seq = torch.cat([
                torch.tensor([self.bos_idx]),
                torch.full((add_len,), self.mask_idx),
                torch.tensor([self.eos_idx]),
            ])
            seqs.append(seq)
        max_l = max(len(s) for s in seqs)
        xt = torch.stack([
            F.pad(s, (0, max_l - len(s)), value=self.pad_idx) for s in seqs
        ]).to(self.device)

        num_steps = max(int(self.mdlm.get_num_steps_confidence(xt)), 2)
        att = (xt != self.pad_idx).long()
        for i in range(num_steps):
            logits = model(xt, att)
            xt = self.mdlm.step_confidence(
                logits, xt, i, num_steps,
                self.softmax_temp, self.randomness,
            )
        if was_training:
            model.backbone.train()
        return xt

    @torch.no_grad()
    def _fill_buffer(self, model, num_samples, label=""):
        """Generate molecules, evaluate reward, store in replay buffer."""
        xt = self._generate_tokens(model, num_samples)
        smiles = self._decode_tokens(xt)
        raw_scores = self.reward_fn(smiles)
        if not isinstance(raw_scores, torch.Tensor):
            raw_scores = torch.tensor(raw_scores, dtype=torch.float32)
        att = (xt != self.pad_idx).long()

        valid = torch.isfinite(raw_scores) & (raw_scores > -99)
        n_valid = int(valid.sum())
        if n_valid > 0:
            self.replay_buffer.add_batch(
                xt[valid], att[valid],
                [s for s, v in zip(smiles, valid) if v],
                raw_scores[valid],
            )
        if self.verbose or label:
            logger.info(
                "%s: %d/%d valid  (buffer=%d)",
                label or "fill", n_valid, num_samples, len(self.replay_buffer),
            )
        return n_valid

    # ── VIDD-specific helpers ───────────────────────────────────────

    def _sample_xs_from_logits(self, logits, xt):
        """Sample xs ~ p(·|xt) at masked positions, copy unmasked positions.

        Args:
            logits: [B, L, V] logits from a denoising model
            xt:     [B, L] current state (contains MASK at some positions)
        Returns:
            xs: [B, L] sampled intermediate state
        """
        probs = F.softmax(logits, dim=-1)                      # [B, L, V]
        sampled = torch.distributions.Categorical(probs=probs).sample()  # [B, L]
        is_masked = (xt == self.mask_idx)
        xs = torch.where(is_masked, sampled, xt)
        return xs

    def _normalize_reward(self, r):
        """Apply reward_norm strategy.

        - none:    return raw rewards (no scaling).
        - pos:     shift to non-negative, useful for log() downstream.
        - normal:  classic z-score over the full batch.
        - top_k:   z-score using mean/std of the TOP-K rewards only.
                   Calibrates against 'the good ones' so the normalization
                   isn't dominated by a tail of zero / invalid_penalty
                   sentinel values.
        """
        if self.reward_norm == "none":
            return r
        if self.reward_norm == "pos":
            return r - r.min() + 1e-6
        if self.reward_norm == "normal":
            return (r - r.mean()) / (r.std().clamp(min=1e-6))
        if self.reward_norm == "top_k":
            k = min(int(self.reward_norm_top_k), int(r.shape[0]))
            if k <= 1:
                # not enough samples for a meaningful std; fall back to raw.
                return r
            top_vals, _ = torch.topk(r, k)
            mu = top_vals.mean()
            std = top_vals.std().clamp(min=1e-6)
            return (r - mu) / std
        return r

    def _compute_reward_weight(self, smiles_list):
        """Compute exp(normalized_reward / alpha), detached.

        Args:
            smiles_list: list of SMILES or None
        Returns:
            w:     [B] reward weights (detached)
            valid: [B] bool mask of successful rewards
        """
        raw = self.reward_fn(smiles_list)
        if not isinstance(raw, torch.Tensor):
            # Coerce None -> NaN so downstream isfinite() filtering catches them.
            # Active_loop fa_oracle returns None for invalid SMILES / FA failures.
            raw = [float("nan") if x is None else float(x) for x in raw]
            raw = torch.tensor(raw, dtype=torch.float32)
        raw = raw.to(self.device)

        valid = torch.isfinite(raw) & (raw > -99)
        r = raw.clone()
        if valid.any():
            fallback = r[valid].mean()
            r = torch.where(valid, r, fallback.expand_as(r))
        else:
            r = torch.zeros_like(r)

        # CVaR floor (no-op when threshold == -inf).
        if self.reward_clip_threshold > float("-inf"):
            r = torch.clamp(r, min=self.reward_clip_threshold)

        r_norm = self._normalize_reward(r)
        w = torch.exp(r_norm / self.teacher_alpha)
        return w.detach(), valid

    # ── single VIDD training step ───────────────────────────────────

    def _rewards_to_weight(self, raw_rewards):
        """Normalize scalar rewards → exp(r/alpha) weights, detached.

        CVaR floor (reward_clip_threshold) is applied defensively in case
        rewards entered the buffer before a threshold was set.
        """
        r = raw_rewards.to(self.device).float()
        valid = torch.isfinite(r)
        if not valid.all():
            fallback = r[valid].mean() if valid.any() else torch.zeros([], device=self.device)
            r = torch.where(valid, r, fallback.expand_as(r))
        if self.reward_clip_threshold > float("-inf"):
            r = torch.clamp(r, min=self.reward_clip_threshold)
        r_norm = self._normalize_reward(r)
        w = torch.exp(r_norm / self.teacher_alpha)
        return w.detach()

    def _apply_exploration_A(self, v, log_p_theta_detached, log_p_pre_detached=None):
        """Approach A (negative-score exploration). De Santi 2025 form:

            r_eff = r - gamma * log( pi_theta(x) / pi_pre(x) )

        which expands to  r - gamma * (log_p_theta - log_p_pre).
        In VIDD weight space:

            v_eff = v * exp(-gamma * (log_p_theta - log_p_pre) / alpha)

        The `+gamma * log_p_pre` term anchors the policy toward the pretrained
        reference (KL regularization). Without it, neg-score is pure entropy
        and the policy drifts. DDPP-LB gets this anchor for free via its TB
        squared residual; VIDD-rw_mle does not, so we pass log_p_pre explicitly.

        For backward compat: if log_p_pre_detached is None we fall back to the
        old (anchorless) form, but caller is expected to always supply it.

        Caller MUST pass log_p as PER-TOKEN (sum / n_masked) for multi-token
        sequences — the exponential explodes on raw sums. The clamp at the end
        is a safety net for edge cases.
        """
        if self.exploration_approach != "A" or self.exploration_gamma == 0.0:
            return v
        alpha = max(float(self.teacher_alpha), 1e-6)
        if log_p_pre_detached is not None:
            log_ratio = log_p_theta_detached - log_p_pre_detached
        else:
            log_ratio = log_p_theta_detached
        log_mod = (-self.exploration_gamma * log_ratio / alpha).clamp(
            min=-20.0, max=20.0
        )
        return v * torch.exp(log_mod)

    def train_step(self):
        """One VIDD gradient step: roll-in → timestep losses → backward.

        Returns:
            dict with metrics (includes 'oracle_calls' used this step),
            or None if buffer is too small for pretrain roll-in.
        """
        step_oracle_calls = 0  # count oracle calls made in THIS train_step

        # === ROLL-IN: decide where x0 comes from ===
        if self.use_schedule_rollin:
            lmbda = min(self.global_step / max(1, self.schedule_max_step),
                        1.0) * self.gkd_lmbda
        else:
            lmbda = self.gkd_lmbda

        if random.random() < lmbda:
            # On-policy roll-in: generate fresh from old_model / finetuned
            # Need to call oracle on x0 (no stored reward) → costs batch_size calls
            roll_model = self.old_model if self.old_roll_in else self.finetuned
            with torch.no_grad():
                x0 = self._generate_tokens(roll_model, self.batch_size)
            x0_smiles = self._decode_tokens(x0)
            v_x0 = self._compute_reward_weight(x0_smiles)[0]
            step_oracle_calls += self.batch_size
            rollin_tag = "old" if self.old_roll_in else "student"
        else:
            # Pretrain roll-in: sample from replay buffer (reward already stored)
            if len(self.replay_buffer) < self.batch_size:
                return None
            x0, _, _, buffer_rewards = self.replay_buffer.sample(self.batch_size)
            x0 = x0.to(self.device)
            v_x0 = self._rewards_to_weight(buffer_rewards)  # no oracle call
            rollin_tag = "pretrain"

        # === LOSS OVER TIMESTEPS ===
        total_loss = 0.0
        eps = self.sampling_eps
        v_xs_mean = float("nan")
        v_xs_std = float("nan")

        self.finetuned.backbone.train()

        for _ in range(self.timesteps_per_epoch):
            t = torch.rand(x0.shape[0], device=self.device) * (1 - 2 * eps) + eps

            # Corrupt x0 → xt via MDLM forward process
            xt = self.mdlm.forward_process(x0, t)
            att_mask = (xt != self.pad_idx).long()

            # Student logits (always needed)
            logits_new = self.finetuned(xt, att_mask)                # [B, L, V]
            mask_float = (xt == self.mask_idx).float()

            # ── Loss variants ──
            needs_xs = self.loss_func in ("kl", "ddpo", "ddpp")

            if needs_xs:
                # Compute xs via old_model logits, call oracle on decoded xs
                with torch.no_grad():
                    logits_old = self.old_model(xt, att_mask)
                    xs = self._sample_xs_from_logits(logits_old, xt)
                xs_smiles = self._decode_tokens(xs)
                v_xs, _ = self._compute_reward_weight(xs_smiles)
                step_oracle_calls += self.batch_size
                v_xs_mean = v_xs.mean().item()
                v_xs_std = v_xs.std().item()

                log_probs_new = F.log_softmax(logits_new, dim=-1)
                lp_new_xs = log_probs_new.gather(2, xs.unsqueeze(-1)).squeeze(-1)
                lp_new_xs_masked = (lp_new_xs * mask_float).sum(dim=1)  # [B]

            if self.loss_func == "rw_mle":
                # Reward-weighted MLE at x_0, uses buffer reward for v_x0.
                # Approach A (De Santi-faithful): r_eff = r - gamma * log(pi_theta/pi_pre).
                # PER-TOKEN log-probs to keep the exp() weight bounded.
                lp_x0 = self._compute_denoising_log_prob(self.finetuned, x0, xt)
                _n_masked = mask_float.sum(dim=1).clamp(min=1.0)
                if self.negscore_estimator == "llada" and self.exploration_approach == "A" and self.exploration_gamma > 0.0:
                    with torch.no_grad():
                        lp_x0_for_w = self._compute_denoising_log_prob_alg3(
                            self.finetuned, x0, self.negscore_n_mc,
                        )
                    lp_x0_per_token = (lp_x0_for_w / _n_masked).detach()
                else:
                    lp_x0_per_token = (lp_x0 / _n_masked).detach()
                if self.exploration_approach == "A" and self.exploration_gamma > 0.0:
                    with torch.no_grad():
                        lp_x0_pre = self._compute_denoising_log_prob(self.pretrained, x0, xt)
                    lp_x0_pre_per_token = (lp_x0_pre / _n_masked).detach()
                    v_x0_eff = self._apply_exploration_A(
                        v_x0, lp_x0_per_token, lp_x0_pre_per_token,
                    )
                else:
                    v_x0_eff = v_x0
                loss_t = -(v_x0_eff * lp_x0).mean()

            elif self.loss_func == "kl":
                # Reward-weighted MLE at intermediate state x_s.
                # Approach A (De Santi-faithful): need both pi_theta and pi_pre at x_s.
                _n_masked_kl = mask_float.sum(dim=1).clamp(min=1.0)
                lp_new_xs_per_token = (lp_new_xs_masked / _n_masked_kl).detach()
                if self.exploration_approach == "A" and self.exploration_gamma > 0.0:
                    with torch.no_grad():
                        lp_pre_xs = self._compute_denoising_log_prob(self.pretrained, xs, xt)
                    lp_pre_xs_per_token = (lp_pre_xs / _n_masked_kl).detach()
                    v_xs_eff = self._apply_exploration_A(
                        v_xs, lp_new_xs_per_token, lp_pre_xs_per_token,
                    )
                else:
                    v_xs_eff = v_xs
                loss_t = -(v_xs_eff * lp_new_xs_masked).mean()

            elif self.loss_func == "ddpo":
                # PPO-style ratio clipping using xs rewards as advantage
                with torch.no_grad():
                    log_probs_old = F.log_softmax(logits_old, dim=-1)
                    lp_old_xs = log_probs_old.gather(2, xs.unsqueeze(-1)).squeeze(-1)
                    lp_old_masked = (lp_old_xs * mask_float).sum(dim=1)
                ratio = torch.exp(lp_new_xs_masked - lp_old_masked.detach())
                clamped = torch.clamp(ratio, 1 - self.ratio_clip, 1 + self.ratio_clip)
                adv = (v_xs - v_xs.mean()).detach()
                loss_t = torch.maximum(-adv * ratio, -adv * clamped).mean()

            elif self.loss_func == "ddpp":
                # Match reward-diff to log-prob-diff (MSE residual)
                lp_pre_x0 = self._compute_denoising_log_prob(self.pretrained, x0, xt)
                lp_new_x0 = self._compute_denoising_log_prob(self.finetuned, x0, xt)
                residual = (lp_new_x0 - lp_pre_x0.detach()
                            - (torch.log(v_x0 + 1e-10) - torch.log(v_xs + 1e-10)))
                loss_t = (residual ** 2).mean()

            else:
                raise ValueError(f"Unknown loss_func: {self.loss_func}")

            total_loss = total_loss + loss_t / self.timesteps_per_epoch

        # === BACKPROP + OPTIMIZER STEP ===
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.finetuned.backbone.parameters(), self.grad_clip
        )
        self.optimizer.step()

        # === EMA ===
        with torch.no_grad():
            for n, p in self.finetuned.backbone.named_parameters():
                self.ema_params[n].mul_(self.ema_decay).add_(
                    p.data, alpha=1 - self.ema_decay
                )

        self.global_step += 1

        # === Periodic old_model sync ===
        if (self.target_update_interval > 0
                and self.global_step % self.target_update_interval == 0):
            self.old_model.backbone.load_state_dict(
                self.finetuned.backbone.state_dict()
            )
            if self.verbose:
                logger.info("Synced old_model ← finetuned at step %d", self.global_step)

        # === Periodic buffer refill (optional) ===
        if (self.refill_interval > 0
                and self.global_step % self.refill_interval == 0):
            self._fill_buffer(
                self.finetuned, self.refill_batch_size,
                label=f"step-{self.global_step} on-policy refill",
            )

        return {
            "loss": float(total_loss.detach().item()),
            "rollin": rollin_tag,
            "v_xs_mean": v_xs_mean,
            "v_xs_std": v_xs_std,
            "v_x0_mean": float(v_x0.mean().item()),
            "oracle_calls": step_oracle_calls,
        }

    # ── main training loop ──────────────────────────────────────────

    def train(self, num_steps: int = None, max_steps: int = None,
              timeout_sec: float = None,
              checkpoint_oracle_interval: int = None,
              max_oracle_calls: int = None,
              checkpoint_steps: list = None):
        """VIDD training loop. Same signature as DDPPLBTrainer.train.

        Supports both `num_steps` and `max_steps` (active_loop uses max_steps).
        """
        from time import time

        if num_steps is None and max_steps is not None:
            num_steps = max_steps
        if num_steps is None:
            raise ValueError("Must specify num_steps or max_steps")

        t0 = time()
        if not self.trajectory:
            self.trajectory = []
            self.checkpoints = {}
            self.cum_oracle_calls = 0
            self.cum_backward = 0

        next_ckpt_oracle = checkpoint_oracle_interval if checkpoint_oracle_interval else None

        def _maybe_save_oracle_checkpoint(step_label=""):
            nonlocal next_ckpt_oracle
            if next_ckpt_oracle is None:
                return
            while self.cum_oracle_calls >= next_ckpt_oracle:
                ckpt = {n: p.clone() for n, p in self.ema_params.items()}
                self.checkpoints[f"oracle_{self.cum_oracle_calls}"] = ckpt
                logger.info("Saved checkpoint at %d oracle calls %s",
                            self.cum_oracle_calls, step_label)
                next_ckpt_oracle += checkpoint_oracle_interval

        # Seed replay buffer from pretrained (for pretrain roll-in)
        if len(self.replay_buffer) < self.batch_size:
            n_init = max(self.initial_buffer_from_pretrained, self.batch_size * 2)
            logger.info("Seeding replay buffer from pre-trained model (%d) …", n_init)
            self._fill_buffer(self.pretrained, n_init, label="init-pretrained")
            self.cum_oracle_calls += n_init
            _maybe_save_oracle_checkpoint("(init)")

        best_reward = max((it["reward"] for it in self.replay_buffer.buffer),
                          default=float("-inf"))
        self.trajectory.append({
            "wall_sec": time() - t0,
            "round_id": 0,
            "cum_oracle_calls": self.cum_oracle_calls,
            "cum_forward": 0,
            "running_best": best_reward,
            "cum_backward": 0,
            "phase": "train",
            "training_loss": 0.0,
        })

        if max_oracle_calls is not None and self.cum_oracle_calls >= max_oracle_calls:
            logger.info("Oracle budget exhausted after init (%d calls).",
                        self.cum_oracle_calls)
            return

        checkpoint_hours = set()

        for step in range(1, num_steps + 1):
            if timeout_sec is not None and time() - t0 > timeout_sec:
                logger.info("Timeout reached (%.0fs). Stopping at step %d.",
                            timeout_sec, step - 1)
                break

            metrics = self.train_step()
            if metrics is None:
                logger.warning("Buffer too small, skipping step %d", step)
                self._fill_buffer(self.pretrained, self.batch_size * 2,
                                  label="emergency-refill")
                self.cum_oracle_calls += self.batch_size * 2
                _maybe_save_oracle_checkpoint(f"(step {step})")
                if max_oracle_calls is not None and self.cum_oracle_calls >= max_oracle_calls:
                    logger.info("Oracle budget reached (%d calls).",
                                self.cum_oracle_calls)
                    break
                continue

            self.cum_backward += 1

            # Oracle calls made in this train_step depend on the loss variant:
            #   rw_mle + pretrain roll-in:  0 calls (reward from buffer)
            #   rw_mle + on-policy roll-in: batch_size calls (oracle on x0)
            #   kl / ddpo:                  batch_size * timesteps_per_epoch (oracle on xs)
            #   ddpp:                       same as kl (xs) + batch_size on-policy if on-policy
            self.cum_oracle_calls += metrics["oracle_calls"]
            _maybe_save_oracle_checkpoint(f"(step {step})")

            if max_oracle_calls is not None and self.cum_oracle_calls >= max_oracle_calls:
                logger.info("Oracle budget reached (%d calls).",
                            self.cum_oracle_calls)
                break

            # Trajectory logging (every 50 steps)
            if step % 50 == 0:
                best_reward = max(
                    (it["reward"] for it in self.replay_buffer.buffer),
                    default=float("-inf"),
                )
                self.trajectory.append({
                    "wall_sec": time() - t0,
                    "round_id": step,
                    "cum_oracle_calls": self.cum_oracle_calls,
                    "cum_forward": 0,
                    "running_best": best_reward,
                    "cum_backward": self.cum_backward,
                    "phase": "train",
                    "training_loss": metrics["loss"],
                })

            # Step-based checkpoints
            if checkpoint_steps and step in checkpoint_steps:
                ckpt = {n: p.clone() for n, p in self.ema_params.items()}
                self.checkpoints[f"step_{step}"] = ckpt
                logger.info("Saved checkpoint at step %d", step)

            # Hourly checkpoints
            elapsed_hr = int((time() - t0) / 3600)
            if elapsed_hr > 0 and elapsed_hr not in checkpoint_hours:
                checkpoint_hours.add(elapsed_hr)
                ckpt = {n: p.clone() for n, p in self.ema_params.items()}
                self.checkpoints[f"{elapsed_hr}hr"] = ckpt
                logger.info("Saved checkpoint at %dhr (step %d)", elapsed_hr, step)

            if step % 50 == 0 or step == 1:
                logger.info(
                    "step %5d/%d  loss=%.4f  rollin=%s  v_x0=%.3f  v_xs=%.3f±%.3f  "
                    "oracle(+%d→%d)  elapsed=%.0fs",
                    step, num_steps, metrics["loss"], metrics["rollin"],
                    metrics["v_x0_mean"], metrics["v_xs_mean"], metrics["v_xs_std"],
                    metrics["oracle_calls"], self.cum_oracle_calls, time() - t0,
                )

    # ── generation (identical to DDPP) ──────────────────────────────

    @torch.no_grad()
    def generate(self, num_samples=100, use_ema=True,
                 softmax_temp=None, randomness=None, min_add_len=None):
        """Sample molecules from the fine-tuned student."""
        # Accept optional overrides for active_loop compatibility
        if softmax_temp is not None:
            self.softmax_temp = softmax_temp
        if randomness is not None:
            self.randomness = randomness
        if min_add_len is not None:
            self.min_add_len = min_add_len

        if use_ema:
            orig = {}
            for n, p in self.finetuned.backbone.named_parameters():
                orig[n] = p.data.clone()
                p.data.copy_(self.ema_params[n])

        xt = self._generate_tokens(self.finetuned, num_samples)
        smiles = self._decode_tokens(xt)

        if use_ema:
            for n, p in self.finetuned.backbone.named_parameters():
                p.data.copy_(orig[n])

        return smiles

    # ── save / load ─────────────────────────────────────────────────

    def save(self, path):
        """Save fine-tuned checkpoint (EMA weights + old_model + optimizer)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state = {k: v.clone() for k, v in self.ema_params.items()}
        torch.save({
            "backbone_state_dict": state,
            "old_model_state_dict": self.old_model.backbone.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "loss_func": self.loss_func,
            "teacher_alpha": self.teacher_alpha,
            "gkd_lmbda": self.gkd_lmbda,
        }, path)
        logger.info("Saved VIDD checkpoint → %s", path)

    def load(self, path):
        """Resume from a VIDD checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.finetuned.backbone.load_state_dict(
            ckpt["backbone_state_dict"], strict=False
        )
        self.ema_params = {
            k: v.to(self.device) for k, v in ckpt["backbone_state_dict"].items()
        }
        if "old_model_state_dict" in ckpt:
            self.old_model.backbone.load_state_dict(
                ckpt["old_model_state_dict"], strict=False
            )
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
        logger.info("Loaded VIDD checkpoint ← %s (step %d)", path, self.global_step)

    # ── CLI entry point ─────────────────────────────────────────────

    @staticmethod
    def run_from_config(cfg):
        """Train → evaluate → save pipeline driven by a Hydra config."""
        import json
        import time as _time

        import hydra
        import pandas as pd
        from omegaconf import OmegaConf
        from rdkit import Chem
        from rdkit.Chem import Descriptors, QED as _QED

        from genmol.rewards import get_reward

        model_path = hydra.utils.to_absolute_path(cfg.model_path)
        output_dir = hydra.utils.to_absolute_path(cfg.output_dir)
        os.makedirs(output_dir, exist_ok=True)

        reward_name = cfg.get("reward", "qed")
        reward_fn = get_reward(reward_name)
        if reward_fn is None:
            raise ValueError(f"Reward '{reward_name}' not found")

        trainer = VIDDTrainer(
            model_path=model_path,
            reward_fn=reward_fn,
            lr=cfg.get("lr", 1e-5),
            batch_size=cfg.get("batch_size", 32),
            replay_buffer_size=cfg.get("replay_buffer_size", 10_000),
            ema_decay=cfg.get("ema_decay", 0.9999),
            seed=cfg.get("seed", 0),
            verbose=cfg.get("verbose", False),
            loss_func=cfg.get("loss_func", "kl"),
            teacher_alpha=cfg.get("teacher_alpha", 1.0),
            reward_norm=cfg.get("reward_norm", "normal"),
            gkd_lmbda=cfg.get("gkd_lmbda", 0.5),
            old_roll_in=cfg.get("old_roll_in", True),
            target_update_interval=cfg.get("target_update_interval", 20),
            timesteps_per_epoch=cfg.get("timesteps_per_epoch", 4),
            grad_clip=cfg.get("grad_clip", 1.0),
        )

        num_steps = cfg.get("num_steps", 500)
        logger.info("Starting VIDD training for %d steps …", num_steps)
        logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

        t0 = _time.time()
        trainer.train(num_steps)
        elapsed = _time.time() - t0

        ckpt_path = os.path.join(output_dir, "vidd_checkpoint.pt")
        trainer.save(ckpt_path)

        num_eval = cfg.get("num_eval_samples", 100)
        smiles = trainer.generate(num_eval, use_ema=True)

        def _mw(s):
            mol = Chem.MolFromSmiles(s) if s else None
            return float(Descriptors.MolWt(mol)) if mol else None

        df = pd.DataFrame({"smiles": smiles, "mol_wt": [_mw(s) for s in smiles]})
        df.to_csv(os.path.join(output_dir, "samples.csv"), index=False)

        valid_mols = [m for m in (Chem.MolFromSmiles(s) for s in smiles if s) if m]
        validity = len(valid_mols) / max(num_eval, 1)
        unique = len(set(s for s in smiles if s)) / max(len(valid_mols), 1)

        qeds = sorted([_QED.qed(m) for m in valid_mols], reverse=True)
        qed_mean = sum(qeds) / len(qeds) if qeds else 0.0
        top10_n = max(1, len(qeds) // 10)
        qed_top10 = sum(qeds[:top10_n]) / top10_n if qeds else 0.0

        reward_scores = reward_fn(smiles)
        if not isinstance(reward_scores, torch.Tensor):
            reward_scores = torch.tensor(reward_scores, dtype=torch.float32)
        finite = reward_scores.isfinite()
        reward_mean = reward_scores[finite].mean().item() if finite.any() else 0.0

        metrics = {
            "elapsed_sec": elapsed, "num_steps": num_steps,
            "loss_func": cfg.get("loss_func", "kl"),
            "teacher_alpha": cfg.get("teacher_alpha", 1.0),
            "reward": reward_name,
            "reward_mean": reward_mean, "validity": validity,
            "uniqueness": unique, "qed_mean": qed_mean,
            "qed_top10": qed_top10, "qed_max": qeds[0] if qeds else 0.0,
        }
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        with open(os.path.join(output_dir, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(cfg))

        logger.info("Time: %.1f sec  Validity: %.4f  Reward: %.4f  QED: %.4f",
                    elapsed, validity, reward_mean, qed_mean)
        logger.info("Output: %s", output_dir)
