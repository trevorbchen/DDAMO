"""DDPP-LB: Discrete Denoising Posterior Prediction (Lower Bound) for GenMol.

Implements Algorithm 1 from "Steering Masked Discrete Diffusion Models
via Discrete Denoising Posterior Prediction" (Rector-Brooks et al., 2024).

Fine-tunes a pre-trained GenMol masked diffusion model to sample from the
reward-induced Bayesian posterior: π₀(x₀) ∝ p_pre(x₀) · R(x₀)^{1/β}

Usage:
    from genmol.ddpp_trainer import DDPPLBTrainer
    from genmol.rewards import get_reward

    trainer = DDPPLBTrainer(
        model_path="model_v2.ckpt",
        reward_fn=get_reward("qed"),
        beta=0.25,
    )
    trainer.train(num_steps=5000)
    trainer.save("ddpp_finetuned.pt")
"""

import copy
import logging
import os
import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from genmol.samplers.base import load_model_from_path
from genmol.utils.utils_chem import safe_to_smiles
from genmol.utils.bracket_safe_converter import bracketsafe2safe

logger = logging.getLogger(__name__)


# ── Log partition function network ────────────────────────────────────


class LogZNetwork(nn.Module):
    """Small network to estimate log Z_πt(xt).

    Takes a partially masked token sequence xt and diffusion time t,
    outputs a scalar log partition function estimate.  Architecture is
    a lightweight transformer encoder (2 layers) followed by mean-pool
    and a linear head.
    """

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256,
                 n_heads=4, n_layers=2, max_len=256):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        self.time_proj = nn.Linear(1, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=hidden_dim, batch_first=True,
            dropout=0.1, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, xt, t, padding_mask=None):
        """
        Args:
            xt: [B, L] token IDs (partially masked)
            t:  [B] diffusion time in (0, 1)
            padding_mask: [B, L] bool, True = pad (ignored positions)
        Returns:
            [B] scalar log-partition-function estimates
        """
        B, L = xt.shape
        positions = torch.arange(L, device=xt.device).unsqueeze(0).expand(B, -1)
        h = self.tok_embed(xt) + self.pos_embed(positions)
        h = h + self.time_proj(t.unsqueeze(-1)).unsqueeze(1)  # broadcast time
        if padding_mask is not None:
            h = self.encoder(h, src_key_padding_mask=padding_mask)
        else:
            h = self.encoder(h)
        # mean-pool over non-padding positions
        if padding_mask is not None:
            mask_float = (~padding_mask).float().unsqueeze(-1)  # [B, L, 1]
            h = (h * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
        else:
            h = h.mean(dim=1)
        return self.head(h).squeeze(-1)  # [B]


# ── Replay buffer ─────────────────────────────────────────────────────


class ReplayBuffer:
    """Stores tokenised molecule sequences for off-policy DDPP-LB training."""

    def __init__(self, max_size=10_000):
        self.buffer = deque(maxlen=max_size)

    def add_batch(self, token_ids, attention_masks, smiles_list, rewards):
        """Add a batch of valid samples."""
        for i in range(token_ids.shape[0]):
            self.buffer.append({
                "token_ids": token_ids[i].cpu(),
                "attention_mask": attention_masks[i].cpu(),
                "smiles": smiles_list[i],
                "reward": rewards[i].item(),
            })

    def sample(self, batch_size):
        """Sample a mini-batch, returned as stacked tensors."""
        items = random.sample(list(self.buffer),
                              min(batch_size, len(self.buffer)))
        token_ids = torch.stack([it["token_ids"] for it in items])
        att_mask = torch.stack([it["attention_mask"] for it in items])
        smiles = [it["smiles"] for it in items]
        rewards = torch.tensor([it["reward"] for it in items],
                               dtype=torch.float32)
        return token_ids, att_mask, smiles, rewards

    def __len__(self):
        return len(self.buffer)


# ── DDPP-LB Trainer ───────────────────────────────────────────────────


class DDPPLBTrainer:
    """Fine-tune a GenMol MDM via DDPP-LB.

    The single-step DDPP-LB loss (paper Eq. 11, Algorithm 1) is:

        L = || log q_θ(x₀|xₜ) − log p_pre(x₀|xₜ) − (1/β)·log R(x₀) + log Ẑ_πt(xₜ) ||²

    where q_θ is the fine-tuned denoiser, p_pre is the frozen pre-trained
    model, R is the reward function, and Ẑ is a learned log-partition
    function (lower-bound estimator).
    """

    def __init__(
        self,
        model_path: str,
        reward_fn,
        *,
        lr: float = 1e-4,
        lr_logz: float = 1e-3,
        batch_size: int = 16,
        beta: float = 1.0,
        replay_buffer_size: int = 10_000,
        warmup_logz_steps: int = 0,
        sampling_eps: float = 1e-3,
        refill_interval: int = 250,
        refill_batch_size: int = 16,
        initial_buffer_from_pretrained: int = 64,
        softmax_temp: float = 0.8,
        randomness: float = 0.5,
        min_add_len: int = 60,
        ema_decay: float = 0.9999,
        logz_embed_dim: int = 128,
        logz_hidden_dim: int = 256,
        seed: int | None = 0,
        verbose: bool = False,
    ):
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── models ──────────────────────────────────────────────────
        logger.info("Loading pre-trained model (frozen) …")
        self.pretrained = load_model_from_path(model_path)
        self.pretrained.backbone.eval()
        for p in self.pretrained.parameters():
            p.requires_grad_(False)
        self.pretrained.to(self.device)

        logger.info("Loading fine-tuned model (trainable) …")
        self.finetuned = load_model_from_path(model_path)
        self.finetuned.backbone.train()
        self.finetuned.to(self.device)

        # shared references
        self.mdlm = self.pretrained.mdlm
        self.mdlm.to_device(self.device)
        self.tokenizer = self.pretrained.tokenizer
        self.mask_idx = self.pretrained.mask_index
        self.pad_idx = self.tokenizer.pad_token_id
        self.bos_idx = self.pretrained.bos_index
        self.eos_idx = self.pretrained.eos_index
        # actual vocab = tokenizer + any added tokens; use the embedding size
        self.vocab_size = self.pretrained.backbone.bert.embeddings.word_embeddings.num_embeddings
        self.max_len = self.pretrained.config.model.max_position_embeddings
        self.use_bracket_safe = self.pretrained.config.training.get(
            "use_bracket_safe", False
        )

        # ── log-Z network ──────────────────────────────────────────
        self.log_z_net = LogZNetwork(
            vocab_size=self.vocab_size,
            embed_dim=logz_embed_dim,
            hidden_dim=logz_hidden_dim,
            max_len=self.max_len,
        ).to(self.device)

        # ── reward / temperature ────────────────────────────────────
        self.reward_fn = reward_fn
        self.beta = beta

        # ── replay buffer ───────────────────────────────────────────
        self.replay_buffer = ReplayBuffer(max_size=replay_buffer_size)

        # ── optimiser ───────────────────────────────────────────────
        self.optimizer = Adam([
            {"params": self.finetuned.backbone.parameters(), "lr": lr},
            {"params": self.log_z_net.parameters(), "lr": lr_logz},
        ])

        # ── EMA for fine-tuned backbone ─────────────────────────────
        self.ema_decay = ema_decay
        self.ema_params = {
            n: p.data.clone()
            for n, p in self.finetuned.backbone.named_parameters()
        }

        # ── hparams ─────────────────────────────────────────────────
        self.batch_size = batch_size
        self.warmup_logz_steps = warmup_logz_steps
        self.sampling_eps = sampling_eps
        self.refill_interval = refill_interval
        self.refill_batch_size = refill_batch_size
        self.initial_buffer_from_pretrained = initial_buffer_from_pretrained
        self.softmax_temp = softmax_temp
        self.randomness = randomness
        self.min_add_len = min_add_len
        self.verbose = verbose
        self.global_step = 0

    # ── helpers ──────────────────────────────────────────────────────

    def _compute_denoising_log_prob(self, model, x0, xt):
        """log q(x₀ | xₜ) = Σ_{i : xₜⁱ = MASK} log softmax(logits)[i, x₀ⁱ].

        Only masked positions contribute; unmasked positions are
        deterministic (log-prob = 0).
        """
        attention_mask = (xt != self.pad_idx).long()
        logits = model(xt, attention_mask)                    # [B, L, V]
        log_probs = F.log_softmax(logits, dim=-1)            # [B, L, V]
        # gather the log-prob assigned to the true clean token at each pos
        lp = log_probs.gather(2, x0.unsqueeze(-1)).squeeze(-1)  # [B, L]
        mask = (xt == self.mask_idx).float()
        return (lp * mask).sum(dim=1)                         # [B]

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
    def _generate_tokens(self, model, num_samples):
        """Run the confidence-based denoising loop and return raw tokens.

        Returns:
            xt: [N, L] final (fully-denoised) token tensor
        """
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
        model.backbone.train()
        return xt

    @torch.no_grad()
    def _fill_buffer(self, model, num_samples, label=""):
        """Generate molecules, evaluate reward, store valid ones."""
        xt = self._generate_tokens(model, num_samples)
        smiles = self._decode_tokens(xt)
        rewards = self.reward_fn(smiles)          # [N]
        att = (xt != self.pad_idx).long()

        valid = torch.isfinite(rewards) & (rewards > 0)
        n_valid = int(valid.sum())
        if n_valid > 0:
            self.replay_buffer.add_batch(
                xt[valid], att[valid],
                [s for s, v in zip(smiles, valid) if v],
                rewards[valid],
            )
        if self.verbose or label:
            logger.info(
                "%s: %d/%d valid  (buffer=%d)",
                label or "fill", n_valid, num_samples, len(self.replay_buffer),
            )
        return n_valid

    # ── single training step ────────────────────────────────────────

    def train_step(self):
        """One gradient step of single-step DDPP-LB (Algorithm 1).

        Returns:
            dict of scalar metrics, or None if buffer is too small.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        x0, att_mask, smiles, rewards = self.replay_buffer.sample(self.batch_size)
        x0 = x0.to(self.device)
        rewards = rewards.to(self.device)
        B = x0.shape[0]

        # 1. sample time  t ∼ U[ε, 1−ε]
        eps = self.sampling_eps
        t = torch.rand(B, device=self.device) * (1.0 - 2 * eps) + eps

        # 2. forward (masking) process:  xₜ ∼ p_t(xₜ | x₀)
        xt = self.mdlm.forward_process(x0, t)

        # 3. log q_θ(x₀ | xₜ)  — fine-tuned model
        log_q = self._compute_denoising_log_prob(self.finetuned, x0, xt)

        # 4. log p_pre(x₀ | xₜ)  — frozen pre-trained model
        with torch.no_grad():
            log_p = self._compute_denoising_log_prob(self.pretrained, x0, xt)

        # 5. (1/β) · log R(x₀)
        log_r = torch.log(rewards.clamp(min=1e-10)) / self.beta

        # 6. log Ẑ_πt(xₜ)  — learned lower-bound
        pad_mask = (xt == self.pad_idx)  # True = ignore
        log_z = self.log_z_net(xt, t, padding_mask=pad_mask)

        # 7. DDPP-LB loss
        residual = log_q - log_p - log_r + log_z
        loss = (residual ** 2).mean()

        # 8. backward + step
        self.optimizer.zero_grad()
        if self.global_step < self.warmup_logz_steps:
            # warmup: freeze denoiser, only train log-Z head
            for p in self.finetuned.backbone.parameters():
                p.requires_grad_(False)
            loss.backward()
            for p in self.finetuned.backbone.parameters():
                p.requires_grad_(True)
        else:
            loss.backward()
        self.optimizer.step()

        # 9. EMA update
        with torch.no_grad():
            for n, p in self.finetuned.backbone.named_parameters():
                self.ema_params[n].mul_(self.ema_decay).add_(
                    p.data, alpha=1 - self.ema_decay
                )

        self.global_step += 1

        # 10. periodic buffer refill (on-policy from fine-tuned model)
        if (self.refill_interval > 0
                and self.global_step % self.refill_interval == 0):
            self._fill_buffer(
                self.finetuned, self.refill_batch_size,
                label=f"step-{self.global_step} on-policy refill",
            )

        return {
            "loss": loss.item(),
            "residual_abs": residual.abs().mean().item(),
            "log_q": log_q.mean().item(),
            "log_p": log_p.mean().item(),
            "log_r": log_r.mean().item(),
            "log_z": log_z.mean().item(),
        }

    # ── main training loop ──────────────────────────────────────────

    def train(self, num_steps: int):
        """Run the full DDPP-LB fine-tuning loop."""
        # seed buffer from pre-trained model
        n_init = max(self.initial_buffer_from_pretrained, self.batch_size * 2)
        logger.info("Seeding replay buffer from pre-trained model (%d) …", n_init)
        self._fill_buffer(self.pretrained, n_init, label="init-pretrained")

        for step in range(1, num_steps + 1):
            metrics = self.train_step()
            if metrics is None:
                logger.warning("Buffer too small, skipping step %d", step)
                self._fill_buffer(self.pretrained, self.batch_size * 2,
                                  label="emergency-refill")
                continue

            if step % 100 == 0 or step == 1:
                logger.info(
                    "step %5d/%d  loss=%.4f  |res|=%.4f  log_z=%.2f",
                    step, num_steps,
                    metrics["loss"], metrics["residual_abs"], metrics["log_z"],
                )

    # ── generation with fine-tuned model ────────────────────────────

    @torch.no_grad()
    def generate(self, num_samples=100, use_ema=True):
        """Sample molecules from the fine-tuned model.

        Returns:
            list[str | None]  SMILES strings
        """
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
        """Save fine-tuned checkpoint (EMA weights)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # snapshot EMA weights
        state = {k: v.clone() for k, v in self.ema_params.items()}
        torch.save({
            "backbone_state_dict": state,
            "log_z_state_dict": self.log_z_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "beta": self.beta,
        }, path)
        logger.info("Saved DDPP-LB checkpoint → %s", path)

    def load(self, path):
        """Resume from a DDPP-LB checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.finetuned.backbone.load_state_dict(
            ckpt["backbone_state_dict"], strict=False
        )
        self.ema_params = {
            k: v.to(self.device) for k, v in ckpt["backbone_state_dict"].items()
        }
        self.log_z_net.load_state_dict(ckpt["log_z_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
        logger.info("Loaded DDPP-LB checkpoint ← %s  (step %d)", path, self.global_step)

    # ── CLI entry point ──────────────────────────────────────────────

    @staticmethod
    def run_from_config(cfg):
        """Full train → evaluate → save pipeline driven by a Hydra config.

        Used by ``scripts/train_ddpp.py`` (thin CLI wrapper).
        """
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

        trainer = DDPPLBTrainer(
            model_path=model_path,
            reward_fn=reward_fn,
            lr=cfg.get("lr", 1e-4),
            lr_logz=cfg.get("lr_logz", 1e-3),
            batch_size=cfg.get("batch_size", 16),
            beta=cfg.get("beta", 1.0),
            replay_buffer_size=cfg.get("replay_buffer_size", 10_000),
            warmup_logz_steps=cfg.get("warmup_logz_steps", 0),
            sampling_eps=cfg.get("sampling_eps", 1e-3),
            refill_interval=cfg.get("refill_interval", 250),
            refill_batch_size=cfg.get("refill_batch_size", 16),
            initial_buffer_from_pretrained=cfg.get("initial_buffer_from_pretrained", 64),
            softmax_temp=cfg.get("softmax_temp", 0.8),
            randomness=cfg.get("randomness", 0.5),
            min_add_len=cfg.get("min_add_len", 60),
            ema_decay=cfg.get("ema_decay", 0.9999),
            seed=cfg.get("seed", 0),
            verbose=cfg.get("verbose", False),
        )

        num_steps = cfg.get("num_steps", 5000)
        logger.info("Starting DDPP-LB training for %d steps …", num_steps)
        logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

        t0 = _time.time()
        trainer.train(num_steps)
        elapsed = _time.time() - t0

        ckpt_path = os.path.join(output_dir, "ddpp_checkpoint.pt")
        trainer.save(ckpt_path)

        # ── evaluate ─────────────────────────────────────────────────
        num_eval = cfg.get("num_eval_samples", 100)
        logger.info("Generating %d evaluation samples …", num_eval)
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
        finite = reward_scores.isfinite()
        reward_mean = reward_scores[finite].mean().item() if finite.any() else 0.0

        metrics = {
            "elapsed_sec": elapsed, "num_steps": num_steps,
            "beta": cfg.get("beta", 1.0), "reward": reward_name,
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
