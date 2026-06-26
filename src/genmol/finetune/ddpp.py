"""
DDPP-LB (Denoising Diffusion Policy Posterior — Lower Bound variant).

Loss (per batch):
    L(θ, φ) = E_t E_{x0~buffer} E_{xt~p(xt|x0,t)} [
        (log q_θ(x0|xt) − log p_pre(x0|xt) − (1/β)·log R(x0) + log Ẑ_φ(xt, t))²
    ]

Both θ (finetuned backbone) and φ (LogZNetwork) are trained jointly.
p_pre is the pretrained model and stays frozen throughout.
log Ẑ_φ is a learned slack variable that absorbs the per-(xt,t) normalisation
constant; it is NOT a value/quality predictor.

Reference: https://arxiv.org/pdf/2410.08134
"""

import copy
import itertools
import random
from collections import deque
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from genmol.utils.ema import ExponentialMovingAverage


# ---------------------------------------------------------------------------
# LogZ network  φ: (x_t, t) → scalar log Ẑ
# ---------------------------------------------------------------------------

class LogZNetwork(nn.Module):
    """
    Lightweight transformer encoder that maps a noisy sequence x_t and
    diffusion time t to a scalar log partition estimate log Ẑ.

    Architecture:
        token_emb(x_t)  +  time_proj(t)   →  TransformerEncoder  →  mean-pool  →  linear  →  scalar
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.time_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, 1)

    def forward(self, xt: torch.Tensor, t: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            xt:        [B, L] int token ids
            t:         [B]    float diffusion times in [0, 1]
            attn_mask: [B, L] float/long attention mask (1 = real token)

        Returns:
            log_z: [B] scalar per example
        """
        x = self.token_emb(xt.clamp(0, self.token_emb.num_embeddings - 1))  # [B, L, d]; clamp mask_idx OOB
        t_emb = self.time_proj(t.unsqueeze(-1).float())     # [B, d]
        x = x + t_emb.unsqueeze(1)                         # [B, L, d]

        key_padding_mask = None
        if attn_mask is not None:
            key_padding_mask = ~attn_mask.bool()            # True = ignore

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)  # [B, L, d]

        if attn_mask is not None:
            mask_f = attn_mask.unsqueeze(-1).float()        # [B, L, 1]
            pooled = (x * mask_f).sum(1) / mask_f.sum(1).clamp(min=1.0)
        else:
            pooled = x.mean(1)                              # [B, d]

        return self.out(pooled).squeeze(-1)                 # [B]


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    Fixed-capacity FIFO buffer.  Entries are stored as CPU tensors and padded
    on the fly when a batch is sampled.
    """

    def __init__(self, capacity: int = 10_000, pad_token_id: int = 0, eviction: str = "fifo"):
        self.capacity = capacity
        # For FIFO we use deque (O(1) evict); for priority we use list + min-search on add
        self.buf = deque(maxlen=capacity) if eviction == "fifo" else []
        self.pad_token_id = pad_token_id
        self.eviction = eviction

    def add(
        self,
        input_ids: torch.Tensor,    # [1, L] or [L]
        attention_mask: torch.Tensor,
        smiles: str,
        reward: float,
    ) -> None:
        ids = input_ids.squeeze(0).cpu()
        mask = attention_mask.squeeze(0).cpu()
        entry = {"input_ids": ids, "attention_mask": mask, "smiles": smiles, "reward": reward}
        if self.eviction == "priority":
            if len(self.buf) < self.capacity:
                self.buf.append(entry)
            else:
                # Find min-reward entry; replace only if new reward is higher
                min_i = 0
                min_r = self.buf[0]["reward"]
                for i in range(1, len(self.buf)):
                    if self.buf[i]["reward"] < min_r:
                        min_r = self.buf[i]["reward"]
                        min_i = i
                if reward > min_r:
                    # remove min and append new to end (preserves "fresh at tail" semantics)
                    self.buf.pop(min_i)
                    self.buf.append(entry)
                else:
                    # New reward is lower than everything in buffer; still append to tail
                    # but evict oldest (index 0) to keep capacity
                    self.buf.pop(0)
                    self.buf.append(entry)
        else:
            self.buf.append(entry)

    def sample(self, batch_size: int, fresh_count: int = 0, fresh_fraction: float = 0.0) -> dict:
        if fresh_count > 0 and fresh_fraction > 0.0 and fresh_count < len(self.buf):
            n_fresh = min(int(round(batch_size * fresh_fraction)), fresh_count)
            n_old = batch_size - n_fresh
            buf_list = list(self.buf)
            fresh_pool = buf_list[-fresh_count:]
            old_pool = buf_list[:-fresh_count]
            fresh_batch = random.sample(fresh_pool, min(n_fresh, len(fresh_pool)))
            if len(fresh_batch) < n_fresh:
                n_old += n_fresh - len(fresh_batch)
            old_batch = random.sample(old_pool, min(n_old, len(old_pool))) if old_pool else []
            batch = fresh_batch + old_batch
        else:
            batch = random.sample(self.buf, min(batch_size, len(self.buf)))
        max_len = max(item["input_ids"].shape[0] for item in batch)

        ids_list, mask_list, rewards = [], [], []
        for item in batch:
            ids = item["input_ids"]
            L = ids.shape[0]
            pad = max_len - L
            ids_list.append(torch.cat([ids, torch.full((pad,), self.pad_token_id, dtype=ids.dtype)]))
            mask_list.append(torch.cat([item["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
            rewards.append(item["reward"])

        return {
            "input_ids": torch.stack(ids_list),
            "attention_mask": torch.stack(mask_list),
            "rewards": torch.tensor(rewards, dtype=torch.float),
        }

    def __len__(self) -> int:
        return len(self.buf)


# ---------------------------------------------------------------------------
# DDPP-LB Trainer
# ---------------------------------------------------------------------------

class DDPPLBTrainer:
    """
    Fine-tunes a GenMol model using the DDPP-LB objective.

    Args:
        pretrained_model:           Loaded GenMol instance (will be frozen).
                                    Mutually exclusive with model_path.
        reward_fn:                  callable(list[str]) -> list[float | None]
        beta:                       Inverse temperature scaling log R.
        lr:                         Learning rate for the finetuned backbone θ.
        lr_logz:                    Learning rate for the LogZNetwork φ.
        buffer_size:                Maximum replay buffer capacity.
        batch_size:                 Training batch size.
        warmup_logz_steps:          Train only φ for this many steps before
                                    unfreezing θ (stabilises log Z first).
        refill_interval:            Auto-refill buffer every N steps (0 = disabled).
        refill_num_samples:         Molecules to generate per auto-refill.
        ema_decay:                  EMA decay for the finetuned backbone.
        logz_d_model:               Hidden size of LogZNetwork.
        logz_nhead:                 Attention heads of LogZNetwork.
        logz_num_layers:            Transformer depth of LogZNetwork.
        model_path:                 Path to GenMol checkpoint (alternative to pretrained_model).
        replay_buffer_size:         Alias for buffer_size.
        initial_buffer_from_pretrained: Generate this many molecules from the
                                    pretrained model at init to seed the buffer.
    """

    def __init__(
        self,
        pretrained_model=None,
        reward_fn=None,
        beta: float = 1.0,
        lr: float = 1e-5,
        lr_logz: float = 1e-3,
        buffer_size: int = 10_000,
        batch_size: int = 32,
        warmup_logz_steps: int = 100,
        refill_interval: int = 50,
        refill_num_samples: int = 16,
        ema_decay: float = 0.999,
        logz_d_model: int = 256,
        logz_nhead: int = 4,
        logz_num_layers: int = 2,
        # comparison.py-compatible kwargs
        model_path: str = None,
        replay_buffer_size: int = None,
        initial_buffer_from_pretrained: int = 0,
    ):
        # ---- resolve model -----------------------------------------------
        try:
            from genmol.sampler import load_model_from_path
        except ModuleNotFoundError:
            from genmol.samplers import load_model_from_path
        if model_path is not None:
            # Load twice to get independent backbone weights without deepcopy
            # (deepcopy fails for models with custom Rust tokenizer PreTokenizers)
            pretrained_model = load_model_from_path(model_path)
            finetuned_model  = load_model_from_path(model_path)
        elif pretrained_model is not None:
            finetuned_model = copy.deepcopy(pretrained_model)
        else:
            raise ValueError("Provide either pretrained_model or model_path")

        # replay_buffer_size is an alias for buffer_size
        if replay_buffer_size is not None:
            buffer_size = replay_buffer_size

        self.model_path = model_path
        self.reward_fn = reward_fn
        self.beta = beta
        self.batch_size = batch_size
        self.warmup_logz_steps = warmup_logz_steps
        self.refill_interval = refill_interval
        self.refill_num_samples = refill_num_samples
        self.step = 0
        # Clipped-reward threshold (set by active loop each epoch).
        # Rewards below this value are clipped to near-zero before computing
        # log_r, focusing DDPP on molecules above the threshold.
        # Units: same as the rewards stored in the buffer (shifted FA scores).
        self.reward_clip_threshold: float = 0.0
        self.fresh_fraction: float = 0.0
        self.last_fresh_count: int = 0
        # Floor for (reward - threshold).clamp(min=reward_floor). Default 1e-8 = log_r ~ -18 (strong push-down).
        # Setting to 1.0 gives log_r=0 (neutral) for below-threshold samples (no aggressive suppression).
        self.reward_floor: float = 1e-8
        # Exploration bonus (A/B/C per De Santi 2025 / Sendera 2024 analysis)
        self.exploration_approach: str = "none"   # none | A | B | C
        self.exploration_gamma: float = 0.0        # γ coefficient (approach A & B); also μ in approach C
        self.kl_lambda: float = 0.0                # λ (approach C only)
        # Neg-score log-prob estimator (paired with exploration_approach A/B/C):
        # "rough" = single-sample Σ 1[mask] log p (current, biased high-var);
        # "llada" = LLaDA Alg-3 — n_mc fresh masks with L/l weight (unbiased).
        self.negscore_estimator: str = "rough"
        self.negscore_n_mc: int = 1
        # MARA: Mode Anchored Reward Augmentation (GX-Chen 2025)
        self.use_mara: bool = False

        # ---- finetuned model (trainable copy) ----------------------------
        self.finetuned = finetuned_model
        self.finetuned.backbone.train()

        # ---- pretrained model (frozen) -----------------------------------
        self.pretrained = pretrained_model
        self.pretrained.backbone.eval()
        for p in self.pretrained.backbone.parameters():
            p.requires_grad_(False)

        # Shared references (same across both copies after deepcopy)
        self.tokenizer = pretrained_model.tokenizer
        self.mdlm = pretrained_model.mdlm
        self.device = pretrained_model.device

        # Keep mdlm on the right device
        self.mdlm.to_device(self.device)

        # ---- replay buffer -----------------------------------------------
        self.buffer = ReplayBuffer(
            capacity=buffer_size,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # ---- LogZ network ------------------------------------------------
        self.logz_net = LogZNetwork(
            vocab_size=self.tokenizer.vocab_size,
            d_model=logz_d_model,
            nhead=logz_nhead,
            num_layers=logz_num_layers,
        ).to(self.device)

        # ---- optimizers --------------------------------------------------
        self.opt_theta = torch.optim.Adam(self.finetuned.backbone.parameters(), lr=lr)
        self.opt_phi = torch.optim.Adam(self.logz_net.parameters(), lr=lr_logz)

        # ---- EMA of finetuned backbone -----------------------------------
        self.ema = ExponentialMovingAverage(
            self.finetuned.backbone.parameters(), decay=ema_decay
        )
        self.ema.move_shadow_params_to_device(self.device)

        # ---- optional initial buffer fill --------------------------------
        if initial_buffer_from_pretrained > 0:
            self._fill_buffer(self.pretrained, initial_buffer_from_pretrained)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def global_step(self) -> int:
        """Alias for self.step — matches comparison.py API."""
        return self.step

    @property
    def buffer_size(self) -> int:
        return len(self.buffer)

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _compute_denoising_log_prob(
        self,
        model,
        x0: torch.Tensor,
        xt: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sum of log q(x0_i | xt) over positions i where xt_i == MASK.

        Args:
            model:     GenMol instance whose forward() is used.
            x0:        [B, L] clean token ids.
            xt:        [B, L] noisy (masked) token ids.
            attn_mask: [B, L] attention mask.

        Returns:
            [B] per-example summed log probabilities.
        """
        logits = model.forward(xt, attn_mask)              # [B, L, V]
        log_probs = F.log_softmax(logits, dim=-1)          # [B, L, V]
        token_lp = log_probs.gather(-1, x0.clamp(0, logits.shape[-1] - 1).unsqueeze(-1)).squeeze(-1)  # [B, L]
        # Only count positions that were masked in xt
        masked = (xt == model.mask_index).float()          # [B, L]
        return (token_lp * masked).sum(-1)                 # [B]

    def _compute_denoising_log_prob_alg3(
        self,
        model,
        x0: torch.Tensor,
        attn_mask: torch.Tensor,
        n_mc: int,
    ) -> torch.Tensor:
        """LLaDA Algorithm 3 marginal log-likelihood estimator.

        Resamples n_mc fresh masked xt's via mdlm.forward_process(t ~ U(0,1)),
        applies the L/l importance weight per realization, averages. Unbiased
        estimator of the MDM ELBO bound on log p_θ(x0).
        """
        B = x0.shape[0]
        valid_L = attn_mask.sum(dim=1).clamp(min=1).float()       # [B]
        accum = torch.zeros(B, device=x0.device)
        for _ in range(int(n_mc)):
            t = self.mdlm.sample_time(B)
            xt = self.mdlm.forward_process(x0, t)
            masked_count = ((xt == model.mask_index).float()
                            * attn_mask.float()).sum(dim=1).clamp(min=1)
            lp = self._compute_denoising_log_prob(model, x0, xt, attn_mask)
            accum = accum + (valid_L / masked_count) * lp
        return accum / float(n_mc)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self) -> float | None:
        """
        Execute one DDPP-LB gradient step.

        Returns:
            Loss value (float), or None if the buffer is too small.
        """
        if len(self.buffer) < self.batch_size:
            return None

        # On-policy auto-refill (skip step 0; refill_interval=0 disables)
        if self.refill_interval > 0 and self.step > 0 and self.step % self.refill_interval == 0:
            self._fill_buffer(self.finetuned, self.refill_num_samples)

        batch = self.buffer.sample(self.batch_size, fresh_count=self.last_fresh_count, fresh_fraction=self.fresh_fraction)
        input_ids = batch["input_ids"].to(self.device)
        attn_mask = batch["attention_mask"].to(self.device)
        rewards = batch["rewards"].to(self.device)

        B = input_ids.shape[0]
        t = self.mdlm.sample_time(B)
        xt = self.mdlm.forward_process(input_ids, t)

        # log q_θ(x0 | xt)  — finetuned model (differentiable)
        log_q = self._compute_denoising_log_prob(self.finetuned, input_ids, xt, attn_mask)

        # log p_pre(x0 | xt) — frozen pretrained (no grad)
        with torch.no_grad():
            log_p = self._compute_denoising_log_prob(self.pretrained, input_ids, xt, attn_mask)

        # MARA reward augmentation (GX-Chen 2025, Algorithm 1):
        # For above-threshold samples, set R̄(y) = R(z) + β*(log p_pre(z) - log p_pre(y))
        # where z is the most-prior-likely above-threshold sample in the batch.
        # Makes the optimal target distribution uniform over above-threshold modes.
        if self.use_mara:
            above = rewards >= self.reward_clip_threshold
            if above.sum() >= 1:
                above_idx = above.nonzero(as_tuple=True)[0]
                anchor_local = log_p[above_idx].argmax()
                anchor_idx = above_idx[anchor_local]
                r_anchor = rewards[anchor_idx].detach()
                log_p_anchor = log_p[anchor_idx].detach()
                rewards = rewards.clone()
                rewards[above] = r_anchor + self.beta * (log_p_anchor - log_p[above].detach())

        # (1/β) · log R(x0) with optional exploration bonus (A/B/C)
        clipped_rewards = (rewards - self.reward_clip_threshold).clamp(min=self.reward_floor)
        log_r_cvar = torch.log(clipped_rewards) / self.beta
        if self.exploration_approach in ("A", "B", "C") and self.exploration_gamma > 0:
            # log p_k(x): current-model log-likelihood of x0|xt (detached — no grad).
            # Estimator: "rough" = single-sample raw Σ 1[mask] log p (current, biased
            # high-variance); "llada" = LLaDA Alg-3, average n_mc fresh mask
            # realizations with L/l importance weight (unbiased lower-variance).
            with torch.no_grad():
                if self.negscore_estimator == "llada":
                    log_p_k = self._compute_denoising_log_prob_alg3(
                        self.finetuned, input_ids, attn_mask, self.negscore_n_mc,
                    ).detach()
                else:
                    log_p_k = self._compute_denoising_log_prob(
                        self.finetuned, input_ids, xt, attn_mask,
                    ).detach()
            if self.exploration_approach == "A":
                # additive: log_r = log_r_cvar - γ * log_p_k
                log_r = log_r_cvar - self.exploration_gamma * log_p_k
            elif self.exploration_approach == "B":
                # multiplicative reweight in tail: log_r = log_r_cvar + γ * log(-log p_k)
                neg_log_p = (-log_p_k).clamp(min=1e-6)
                log_r = log_r_cvar + self.exploration_gamma * torch.log(neg_log_p)
            else:  # C: three-functional FDC in linear space + log at end
                r_cvar_lin = clipped_rewards  # already clamped to reward_floor
                C_const = 10.0  # stabilizing constant to keep argument positive
                r_eff = (self.exploration_gamma * r_cvar_lin
                         + (1.0 + self.kl_lambda) * (-log_p_k)
                         + self.kl_lambda * log_p.detach()
                         + C_const)
                log_r = torch.log(r_eff.clamp(min=self.reward_floor)) / self.beta
        else:
            log_r = log_r_cvar

        # log Ẑ_φ(xt, t)
        log_z = self.logz_net(xt, t, attn_mask)

        # DDPP-LB squared residual
        residual = log_q - log_p - log_r + log_z          # [B]
        loss = (residual ** 2).mean()

        # Gradient step on φ (always)
        self.opt_phi.zero_grad()
        # Gradient step on θ (after warmup)
        if self.step >= self.warmup_logz_steps:
            self.opt_theta.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.logz_net.parameters(), 1.0)
        self.opt_phi.step()
        if self.step >= self.warmup_logz_steps:
            torch.nn.utils.clip_grad_norm_(self.finetuned.backbone.parameters(), 1.0)
            self.opt_theta.step()
            self.ema.update(itertools.chain(self.finetuned.backbone.parameters()))

        self.step += 1
        return loss.item()


    # ── FinetuneTrainer interface ─────────────────────────────────────────

    @property
    def model(self):
        """Return a GenerativeModel view of the fine-tuned backbone.

        Cached so repeated calls in the active loop don't create new wrappers.
        The wrapper holds a reference to self.finetuned, so it always reflects
        the current (post-training) backbone weights.
        """
        if not hasattr(self, "_gen_model_cache"):
            from genmol.genmol_model import GenMolGenerativeModel
            max_len = getattr(
                getattr(self.finetuned, "config", None), "model", {}
            )
            if hasattr(max_len, "max_position_embeddings"):
                max_len = max_len.max_position_embeddings
            else:
                max_len = 128
            object.__setattr__(
                self, "_gen_model_cache",
                GenMolGenerativeModel(self.finetuned, self.tokenizer, str(self.device), max_len)
            )
        return self._gen_model_cache

    @property
    def step(self):
        return getattr(self, "_step", 0)

    def train(self, max_steps: int, timeout_sec: float = None) -> None:
        """
        Run training loop for up to max_steps steps or timeout_sec seconds.

        Matches comparison.py's: trainer.train(max_steps, timeout_sec=budget)

        Bootstrap: if the buffer is too small to start training and auto-refill
        is enabled, do one immediate fill from the finetuned model before the
        loop.  This handles the run_ddpp_method pattern where no external data
        is added before train() is called.  run_mcts_surrogate_method avoids
        this by passing refill_interval=0 and filling the buffer manually.
        """
        if len(self.buffer) < self.batch_size and self.refill_interval > 0:
            n = max(self.refill_num_samples, self.batch_size * 4)
            for _ in range(10):  # retry until buffer is large enough
                self._fill_buffer(self.finetuned, n)
                if len(self.buffer) >= self.batch_size:
                    break

        t0 = time()
        for _ in range(max_steps):
            if timeout_sec is not None and (time() - t0) >= timeout_sec:
                break
            self.train_step()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate(
        self,
        model,
        num_samples: int,
        use_ema: bool = True,
        softmax_temp: float = 0.8,
        randomness: float = 0.5,
        min_add_len: int = 40,
    ) -> list[str | None]:
        """
        Generate molecules from `model` using the confidence-based MDLM denoising loop.
        Returns a list of SMILES (or None for invalid).
        """
        pad_index = self.tokenizer.pad_token_id
        mask_index = model.mask_index
        bos_index = model.bos_index
        eos_index = model.eos_index

        ema_active = (use_ema and model is self.finetuned
                      and self.step > self.warmup_logz_steps)
        if ema_active:
            self.ema.store(itertools.chain(model.backbone.parameters()))
            self.ema.copy_to(itertools.chain(model.backbone.parameters()))

        try:
            seqs = []
            for _ in range(num_samples):
                seq = torch.cat([
                    torch.tensor([bos_index], device=self.device),
                    torch.full((min_add_len,), mask_index, device=self.device),
                    torch.tensor([eos_index], device=self.device),
                ], dim=0)
                seqs.append(seq)

            x = torch.stack(seqs)                          # [N, L]
            attention_mask = (x != pad_index).long()

            num_steps = max(self.mdlm.get_num_steps_confidence(x), 2)
            for i in range(num_steps):
                logits = model(x, attention_mask)
                x = self.mdlm.step_confidence(logits, x, i, num_steps, softmax_temp, randomness)

            decoded = self.tokenizer.batch_decode(x, skip_special_tokens=True)
            from genmol.utils.utils_chem import safe_to_smiles
            try:
                from genmol.utils.bracket_safe_converter import bracketsafe2safe
                _convert = bracketsafe2safe
            except ImportError:
                _convert = lambda s: s  # noqa: E731  # local model, no bracket tokens
            smiles_list = []
            for s in decoded:
                smi = safe_to_smiles(_convert(s), fix=True)
                if smi:
                    smi = sorted(smi.split("."), key=len)[-1]
                else:
                    smi = None
                smiles_list.append(smi)
            return smiles_list

        finally:
            if ema_active:
                self.ema.restore(itertools.chain(model.backbone.parameters()))

    # ------------------------------------------------------------------
    # Buffer population
    # ------------------------------------------------------------------

    def _fill_buffer(self, model_or_num, num_samples=None, label=None) -> None:
        """
        Generate molecules and insert scored ones into the buffer.

        Two calling conventions (both supported):
            _fill_buffer(num_samples)                           — generates from finetuned
            _fill_buffer(model, num_samples, label=...)         — comparison.py style
        """
        if num_samples is None:
            # Called as _fill_buffer(num_samples)
            model, n = self.finetuned, model_or_num
        else:
            # Called as _fill_buffer(model, num_samples, label=...)
            model, n = model_or_num, num_samples

        smiles_list = self._generate(model, n)
        valid = [s for s in smiles_list if s is not None]
        if not valid:
            return
        rewards = self.reward_fn(valid)
        self.add_scored_molecules(valid, rewards)

    def add_scored_molecules(self, smiles_list: list[str], rewards: list[float | None]) -> None:
        """
        Tokenise and insert pre-scored molecules directly into the replay buffer.

        This is the entry point for offline data (e.g. oracle-scored candidates
        from an outer MCTS/beam-search loop).
        """
        # track fresh additions for stratified sampling
        _n_before = len(self.buffer.buf)
        max_len = self.finetuned.config.model.max_position_embeddings
        vocab_size = getattr(self.finetuned.config.model, "vocab_size", None) or self.tokenizer.vocab_size
        for smi, r in zip(smiles_list, rewards):
            if smi is None or r is None:
                continue
            try:
                enc = self.tokenizer(
                    [smi],
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_len,
                )
                ids = enc["input_ids"]
                # sanity: token ids must be in valid embedding range
                if ids.numel() == 0 or ids.max().item() >= vocab_size or ids.min().item() < 0:
                    raise ValueError("out-of-vocab token id from tokenizer (smi=" + repr(smi)[:80] + ")")
                self.buffer.add(enc["input_ids"], enc["attention_mask"], smi, float(r))
            except Exception as e:
                # Fallback: store penalty against a safe placeholder ([BOS][MASK][EOS]).
                # Preserves the negative-reward signal without crashing the embedding lookup.
                import torch as _torch
                bos = getattr(self.finetuned, "bos_index", 1)
                eos = getattr(self.finetuned, "eos_index", 2)
                mask = getattr(self.finetuned, "mask_index", 0)
                ids = _torch.tensor([[bos, mask, eos]], dtype=_torch.long)
                attn = _torch.ones_like(ids)
                self.buffer.add(ids, attn, "<INVALID>", float(r))

    # ------------------------------------------------------------------
    # Public generation API
    # ------------------------------------------------------------------
        self.last_fresh_count = max(0, len(self.buffer.buf) - _n_before)

    def generate(self, num_samples: int, **kwargs) -> list[str]:
        """Generate from the finetuned model, return valid SMILES only.

        Side effects: stash raw output (with None for invalids) in
        self._last_raw_smiles for downstream diversity logging.
        """
        raw = self._generate(self.finetuned, num_samples, **kwargs)
        self._last_n_attempted = len(raw)
        self._last_n_valid = sum(1 for s in raw if s is not None)
        self._last_raw_smiles = list(raw)
        return [s for s in raw if s is not None]
