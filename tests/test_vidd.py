"""
Unit tests for VIDDTrainer.

Covers all the bugs hit during the VIDD ablation campaign so they don't
regress before the massive run:

  1. ReplayBuffer API mismatch (max_size kwarg, add_batch method, 4-tuple sample)
  2. _apply_exploration_A doesn't blow up for multi-token log-prob sums
     (per-token semantics + clamp safety net)
  3. add_scored_molecules tolerates None / non-string sf.encode output
  4. _compute_reward_weight tolerates None oracle returns (coerce -> NaN)
  5. CVaR floor (reward_clip_threshold) is actually applied at storage and
     reward-conversion time
  6. exploration_approach / exploration_gamma attributes plumb through cleanly
"""

import math
import os
import sys
import types
from unittest import mock

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Stub the bits of genmol.samplers.base and genmol.utils.utils_chem that
# vidd.py imports — without triggering the (heavy) full samplers/model chain.
_fake_utils_chem = types.ModuleType("genmol.utils.utils_chem")
_fake_utils_chem.safe_to_smiles = lambda s, fix=True: "CC"
sys.modules.setdefault("genmol.utils.utils_chem", _fake_utils_chem)

_fake_bsc = types.ModuleType("genmol.utils.bracket_safe_converter")
_fake_bsc.bracketsafe2safe = lambda s: s
sys.modules.setdefault("genmol.utils.bracket_safe_converter", _fake_bsc)

# Pre-register a fake samplers.base so the `from genmol.samplers.base import
# load_model_from_path` line in vidd.py succeeds without pulling the real
# samplers tree (which has heavy deps).
_fake_samplers = types.ModuleType("genmol.samplers")
_fake_samplers_base = types.ModuleType("genmol.samplers.base")
_fake_samplers_base.load_model_from_path = lambda path: None
sys.modules.setdefault("genmol.samplers", _fake_samplers)
sys.modules.setdefault("genmol.samplers.base", _fake_samplers_base)

# Same for genmol.finetune.__init__ — it imports DDPP and VIDD; if DDPP's
# heavy deps fail we still want VIDD tests to run. Import VIDD module directly.
import importlib
_vidd_mod = importlib.import_module("genmol.finetune.vidd")
VIDDTrainer = _vidd_mod.VIDDTrainer
ReplayBuffer = _vidd_mod.ReplayBuffer


# ── Helpers ──────────────────────────────────────────────────────────────────

VOCAB = 50
PAD = 0
MASK = 1
BOS = 2
EOS = 3


def _fake_pretrained(vocab=VOCAB, max_len=64, hidden=8):
    """Minimal mock that matches the GenMol API surface VIDD reaches into."""

    class FakeTokenizer:
        vocab_size = vocab
        pad_token_id = PAD
        mask_token_id = MASK
        bos_token_id = BOS
        eos_token_id = EOS

        def __call__(self, texts, return_tensors="pt", truncation=True,
                     max_length=64, padding=True):
            ids = []
            for t in texts:
                n = min(len(t) + 2, max_length)
                row = [BOS] + [5] * (n - 2) + [EOS]
                ids.append(row)
            maxl = max(len(r) for r in ids)
            input_ids = torch.zeros(len(ids), maxl, dtype=torch.long)
            attn = torch.zeros(len(ids), maxl, dtype=torch.long)
            for i, row in enumerate(ids):
                input_ids[i, : len(row)] = torch.tensor(row)
                attn[i, : len(row)] = 1
            return {"input_ids": input_ids, "attention_mask": attn}

        def batch_decode(self, x, skip_special_tokens=True):
            return ["C" * 5 for _ in range(x.shape[0])]

    class FakeMDLM:
        def to_device(self, device):
            pass

        def forward_process(self, x0, t):
            xt = x0.clone()
            B, L = xt.shape
            for b in range(B):
                for pos in range(1, L - 1):
                    if torch.rand(1).item() < 0.4:
                        xt[b, pos] = MASK
            return xt

        def get_num_steps_confidence(self, x):
            return 3

        def step_confidence(self, logits, x, step, num_steps, temp, rand):
            xt = x.clone()
            mp = (xt == MASK).nonzero(as_tuple=False)
            if len(mp):
                row, col = mp[0]
                xt[row, col] = logits[row, col].argmax()
            return xt

    class FakeWordEmbeddings:
        num_embeddings = vocab

    class FakeBertEmb:
        word_embeddings = FakeWordEmbeddings()

    class FakeBert:
        embeddings = FakeBertEmb()

        class config:
            hidden_size = hidden

    class FakeBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(vocab, vocab)
            self.bert = FakeBert()

        def forward(self, x, attn_mask=None, **kw):
            B, L = x.shape
            emb = torch.zeros(B, L, vocab)
            emb.scatter_(-1, x.unsqueeze(-1), 1.0)
            return self.linear(emb.float())

    class FakeConfigModel:
        max_position_embeddings = max_len

    class FakeConfigTraining:
        def get(self, k, default=None):
            return default

    class FakeConfig:
        model = FakeConfigModel
        training = FakeConfigTraining()

    class FakeGenMol(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = FakeBackbone()
            self.tokenizer = FakeTokenizer()
            self.mdlm = FakeMDLM()
            self.mask_index = MASK
            self.bos_index = BOS
            self.eos_index = EOS
            self.config = FakeConfig()
            self.ema = None

        @property
        def device(self):
            return next(self.backbone.parameters()).device

        def forward(self, x, attn_mask=None):
            return self.backbone(x, attn_mask)

    return FakeGenMol()


def _dummy_reward(smiles_list):
    return [0.5] * len(smiles_list)


def _build_trainer(reward_fn=None, **overrides):
    """Construct VIDDTrainer with fake models patched in (no checkpoint load)."""
    fake = _fake_pretrained()
    with mock.patch(
        "genmol.finetune.vidd.load_model_from_path",
        return_value=fake,
    ):
        kw = dict(
            model_path="fake.ckpt",
            reward_fn=reward_fn or _dummy_reward,
            lr=1e-3,
            batch_size=4,
            replay_buffer_size=200,
            initial_buffer_from_pretrained=8,
            ema_decay=0.9,
            seed=0,
            verbose=False,
            loss_func="rw_mle",
            teacher_alpha=1.0,
            reward_norm="none",
            gkd_lmbda=0.0,
            target_update_interval=20,
            timesteps_per_epoch=2,
        )
        kw.update(overrides)
        return VIDDTrainer(**kw)


# ── ReplayBuffer (VIDD's local one) ──────────────────────────────────────────

class TestVIDDReplayBuffer:
    """The buffer is INLINED in vidd.py (different API from DDPP's). Must
    stay self-contained — no cross-file dep on ddpp.py's ReplayBuffer."""

    def test_max_size_kwarg(self):
        # The exact bug that crashed the first VIDD ablation campaign.
        buf = ReplayBuffer(max_size=10)
        assert len(buf) == 0

    def test_add_batch_signature(self):
        buf = ReplayBuffer(max_size=10)
        token_ids = torch.zeros(3, 6, dtype=torch.long)
        masks = torch.ones(3, 6, dtype=torch.long)
        rewards = torch.tensor([0.1, 0.2, 0.3])
        buf.add_batch(token_ids, masks, ["C", "CC", "CCC"], rewards)
        assert len(buf) == 3

    def test_sample_returns_4_tuple(self):
        buf = ReplayBuffer(max_size=10)
        token_ids = torch.zeros(3, 6, dtype=torch.long)
        masks = torch.ones(3, 6, dtype=torch.long)
        rewards = torch.tensor([0.1, 0.2, 0.3])
        buf.add_batch(token_ids, masks, ["C", "CC", "CCC"], rewards)
        out = buf.sample(2)
        assert len(out) == 4, "VIDD's sample() must return a 4-tuple"
        ids, attn, smis, rs = out
        assert ids.shape[0] == 2
        assert attn.shape == ids.shape
        assert isinstance(smis, list) and len(smis) == 2
        assert rs.shape == (2,)

    def test_self_buffer_attr_exists(self):
        # VIDD trainer accesses replay_buffer.buffer (deque), not .buf.
        buf = ReplayBuffer(max_size=10)
        assert hasattr(buf, "buffer")
        assert not hasattr(buf, "buf"), "VIDD's buffer must not collide with DDPP's .buf"

    def test_capacity_fifo(self):
        buf = ReplayBuffer(max_size=3)
        for i in range(5):
            buf.add_batch(
                torch.zeros(1, 4, dtype=torch.long),
                torch.ones(1, 4, dtype=torch.long),
                ["C"],
                torch.tensor([float(i)]),
            )
        assert len(buf) == 3
        rewards = sorted(it["reward"] for it in buf.buffer)
        assert rewards == [2.0, 3.0, 4.0]


# ── VIDDTrainer init ─────────────────────────────────────────────────────────

class TestVIDDInit:
    def test_default_attrs(self):
        t = _build_trainer()
        assert t.reward_clip_threshold == float("-inf")
        assert t.exploration_approach == "none"
        assert t.exploration_gamma == 0.0
        assert t.loss_func == "rw_mle"
        assert hasattr(t, "replay_buffer") and len(t.replay_buffer) == 0

    def test_attrs_settable_after_init(self):
        # active_loop / run_vidd / run_active_loop set these AFTER construction.
        t = _build_trainer()
        t.reward_clip_threshold = 0.3
        t.exploration_approach = "A"
        t.exploration_gamma = 1.0
        assert t.reward_clip_threshold == 0.3
        assert t.exploration_approach == "A"
        assert t.exploration_gamma == 1.0


# ── add_scored_molecules — None / non-string handling ────────────────────────

SMI_A = "O=C(O)c1ccccc1O"               # salicylic acid (2 fragments)
SMI_B = "CC(=O)Nc1ccc(O)cc1"            # acetaminophen
SMI_C = "CC(C)Cc1ccc(C(C)C(=O)O)cc1"    # ibuprofen


class TestAddScoredMolecules:
    def test_basic_add(self):
        t = _build_trainer()
        n = t.add_scored_molecules([SMI_A, SMI_B, SMI_C], [0.1, 0.2, 0.3])
        assert n == 3
        assert len(t.replay_buffer) == 3

    def test_none_smiles_skipped(self):
        t = _build_trainer()
        n = t.add_scored_molecules([SMI_A, None, SMI_B], [0.1, 0.5, 0.2])
        # None entry filtered out — should add 2.
        assert n == 2
        assert len(t.replay_buffer) == 2

    def test_inf_score_skipped(self):
        t = _build_trainer()
        n = t.add_scored_molecules([SMI_A, SMI_B], [-100.0, 0.5])
        # -100 score (sentinel for invalid) should be filtered.
        assert n == 1

    def test_sf_encode_returning_none(self):
        # Patch sf.encode to return None for one of the inputs (simulating
        # a pathological SMILES that bypasses RDKit but breaks SAFE).
        t = _build_trainer()
        with mock.patch("safe.encode", side_effect=[None, "ok_safe", "ok2"]):
            n = t.add_scored_molecules(["weird1", "valid1", "valid2"],
                                        [0.1, 0.2, 0.3])
        # First entry (sf.encode -> None) must be filtered, not crash the tokenizer.
        assert n == 2

    def test_sf_encode_returning_empty_string(self):
        t = _build_trainer()
        with mock.patch("safe.encode", side_effect=["", "ok_safe"]):
            n = t.add_scored_molecules(["weird", "valid"], [0.1, 0.2])
        assert n == 1

    def test_sf_encode_returning_int(self):
        # Sanity: even non-string returns are filtered.
        t = _build_trainer()
        with mock.patch("safe.encode", side_effect=[42, "ok_safe"]):
            n = t.add_scored_molecules(["weird", "valid"], [0.1, 0.2])
        assert n == 1

    def test_cvar_clip_at_storage(self):
        """CVaR floor: rewards below threshold should be clipped UP at storage."""
        t = _build_trainer()
        t.reward_clip_threshold = 0.3
        t.add_scored_molecules([SMI_A, SMI_B, SMI_C], [0.0, 0.1, 0.5])
        rewards = sorted(it["reward"] for it in t.replay_buffer.buffer)
        # 0.0 and 0.1 should both be raised to 0.3; 0.5 unchanged.
        assert rewards == pytest.approx([0.3, 0.3, 0.5])

    def test_cvar_inactive_when_threshold_minus_inf(self):
        t = _build_trainer()
        # Default threshold is -inf → no clipping.
        t.add_scored_molecules([SMI_A, SMI_B], [0.0, 0.5])
        rewards = sorted(it["reward"] for it in t.replay_buffer.buffer)
        assert rewards == pytest.approx([0.0, 0.5])


# ── _compute_reward_weight: handles None / NaN gracefully ────────────────────

class TestComputeRewardWeight:
    def test_none_in_oracle_output(self):
        """fa_oracle returns None for invalid SMILES; trainer must coerce -> NaN."""
        def oracle(smis):
            return [0.5 if s == "ok" else None for s in smis]
        t = _build_trainer(reward_fn=oracle)
        w, valid = t._compute_reward_weight(["ok", "bad", "ok"])
        assert w.shape == (3,)
        # 'ok' samples are finite, 'bad' is masked out.
        assert valid.tolist() == [True, False, True]

    def test_all_valid(self):
        t = _build_trainer()
        w, valid = t._compute_reward_weight(["CC", "CCC", "CCCC"])
        assert valid.all()
        assert torch.isfinite(w).all()

    def test_cvar_clip_in_compute_reward_weight(self):
        """CVaR floor applied to on-policy rewards before normalization."""
        def oracle(smis):
            return [0.0, 0.5]   # below + above threshold
        t = _build_trainer(reward_fn=oracle)
        t.reward_clip_threshold = 0.3
        # With reward_norm="none" and alpha=1: w = exp(r). Clipped 0.0->0.3.
        w, _ = t._compute_reward_weight(["a", "b"])
        # Both above threshold → both at least exp(0.3) ≈ 1.35
        assert w[0].item() >= math.exp(0.3) - 1e-4

    def test_returns_tensor(self):
        # Caller may pass a torch tensor directly (active_loop does this).
        def oracle_tensor(smis):
            return torch.tensor([0.1, 0.2, 0.3])
        t = _build_trainer(reward_fn=oracle_tensor)
        w, valid = t._compute_reward_weight(["a", "b", "c"])
        assert w.shape == (3,)


# ── _rewards_to_weight (buffer-side, defensive CVaR) ─────────────────────────

class TestRewardsToWeight:
    def test_basic(self):
        t = _build_trainer()
        r = torch.tensor([0.1, 0.5, 0.9])
        w = t._rewards_to_weight(r)
        assert w.shape == (3,)
        assert torch.isfinite(w).all()

    def test_nan_handling(self):
        t = _build_trainer()
        r = torch.tensor([float("nan"), 0.5, 0.3])
        w = t._rewards_to_weight(r)
        # Should not propagate NaN out.
        assert torch.isfinite(w).all()

    def test_defensive_cvar_clip(self):
        t = _build_trainer()
        t.reward_clip_threshold = 0.4
        r = torch.tensor([0.1, 0.5, 0.9])
        w = t._rewards_to_weight(r)
        # The two below-threshold inputs should produce identical weights
        # (both clipped to 0.4 before normalization).
        assert torch.allclose(w[0], torch.tensor(math.exp(0.4)), atol=1e-4)


# ── _apply_exploration_A: per-token semantics + no inf ───────────────────────

class TestApplyExplorationA:
    def test_disabled_returns_v_unchanged(self):
        t = _build_trainer()
        v = torch.tensor([1.0, 2.0, 3.0])
        log_p = torch.tensor([-1.0, -2.0, -5.0])
        out = t._apply_exploration_A(v, log_p)
        assert torch.allclose(out, v)

    def test_disabled_when_gamma_zero(self):
        t = _build_trainer()
        t.exploration_approach = "A"
        t.exploration_gamma = 0.0
        v = torch.tensor([1.0, 2.0])
        log_p = torch.tensor([-1.0, -2.0])
        out = t._apply_exploration_A(v, log_p)
        assert torch.allclose(out, v)

    def test_enabled_modifies_v(self):
        t = _build_trainer()
        t.exploration_approach = "A"
        t.exploration_gamma = 1.0
        v = torch.tensor([1.0])
        log_p = torch.tensor([-1.0])  # per-token
        out = t._apply_exploration_A(v, log_p)
        # v_eff = v * exp(-1.0 * -1.0 / 1.0) = exp(1.0) ≈ 2.718
        assert torch.allclose(out, torch.tensor([math.exp(1.0)]), atol=1e-4)

    def test_no_blowup_for_huge_negative_log_p(self):
        """Multi-token sums can be very negative (e.g. -465). The clamp safety
        net at [-20, 20] must prevent inf."""
        t = _build_trainer()
        t.exploration_approach = "A"
        t.exploration_gamma = 1.0
        v = torch.tensor([1.0])
        log_p = torch.tensor([-1000.0])   # pathological
        out = t._apply_exploration_A(v, log_p)
        assert torch.isfinite(out).all()
        # Should be bounded by exp(20) ≈ 4.85e8, not inf
        assert out.item() < 1e10

    def test_no_blowup_for_huge_positive_log_p(self):
        t = _build_trainer()
        t.exploration_approach = "A"
        t.exploration_gamma = 1.0
        v = torch.tensor([1.0])
        log_p = torch.tensor([1000.0])
        out = t._apply_exploration_A(v, log_p)
        assert torch.isfinite(out).all()
        # Should be bounded above 0 by exp(-20)
        assert out.item() > 0

    def test_alpha_scales_modifier(self):
        t = _build_trainer()
        t.exploration_approach = "A"
        t.exploration_gamma = 1.0
        t.teacher_alpha = 2.0
        v = torch.tensor([1.0])
        log_p = torch.tensor([-1.0])
        out = t._apply_exploration_A(v, log_p)
        # exp(-1.0 * -1.0 / 2.0) = exp(0.5)
        assert torch.allclose(out, torch.tensor([math.exp(0.5)]), atol=1e-4)


# ── Integration: API surface that active_loop relies on ──────────────────────

class TestActiveLoopAPI:
    """active_loop.py treats VIDDTrainer as a drop-in for DDPPLBTrainer.
    These tests pin down the API surface so refactors don't break it."""

    def test_has_required_attrs(self):
        t = _build_trainer()
        # Used by active_loop's plumbing.
        for attr in [
            "tokenizer", "mask_idx", "pad_idx", "max_len",
            "reward_clip_threshold", "exploration_approach",
            "exploration_gamma", "replay_buffer",
            "step", "global_step",
        ]:
            assert hasattr(t, attr), f"missing required attr: {attr}"

    def test_has_required_methods(self):
        t = _build_trainer()
        for method in ["add_scored_molecules", "train"]:
            assert callable(getattr(t, method)), f"missing method: {method}"

    def test_step_aliases_global_step(self):
        t = _build_trainer()
        # active_loop reads .step expecting it tracks training progress.
        assert t.step == t.global_step
