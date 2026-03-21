"""Axis 3: Reward / forward operator functions for guided generation.

Each reward is a callable: ``List[str] -> torch.Tensor`` of scores.

Available rewards (increasing compute cost):
    - QEDReward, LogPReward, MolecularWeightReward, TPSAReward  (~free, RDKit)
    - FlashAffinityForwardOp  (~100-500ms/mol, neural binding prediction)
    - BoltzAffinityReward     (~3-10s/mol, structure-based oracle)

Registry:
    >>> from genmol.rewards import get_reward
    >>> reward = get_reward("qed")
    >>> reward = get_reward("flash_affinity", protein_id="2VT4")
"""

from .threshold import ThresholdReward
from .properties import (
    RewardFunction,
    MolecularWeightReward,
    QEDReward,
    LogPReward,
    TPSAReward,
    MolecularWeightForwardOp,  # backward-compat alias
    QEDForwardOp,              # backward-compat alias
    _safe_mol,
)

# ── Registry ──────────────────────────────────────────────────────────

REWARD_REGISTRY = {
    "mw": MolecularWeightReward,
    "qed": QEDReward,
    "logp": LogPReward,
    "tpsa": TPSAReward,
    # Expensive rewards — lazy-loaded to avoid heavy imports when not needed.
    # String values are "module.ClassName" resolved on first use.
    "flash_affinity": "genmol.rewards.flash_affinity.FlashAffinityForwardOp",
    "boltz": "genmol.rewards.boltz.BoltzAffinityReward",
}


def get_reward(name: str, **kwargs):
    """Look up a reward by short name.  Returns an *instance*.

    For parameterised rewards (e.g. flash_affinity), pass constructor
    kwargs directly::

        get_reward("flash_affinity", protein_id="2VT4", task="binary")
    """
    key = name.lower().strip()
    if key in ("none", ""):
        return None
    entry = REWARD_REGISTRY.get(key)
    if entry is None:
        raise ValueError(
            f"Unknown reward {name!r}. "
            f"Available: {', '.join(REWARD_REGISTRY)}"
        )
    if isinstance(entry, str):
        import importlib
        module_path, class_name = entry.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
    else:
        cls = entry
    return cls(**kwargs)


__all__ = [
    "RewardFunction",
    "MolecularWeightReward",
    "QEDReward",
    "LogPReward",
    "TPSAReward",
    "MolecularWeightForwardOp",
    "QEDForwardOp",
    "REWARD_REGISTRY",
    "get_reward",
    "ThresholdReward",
]
