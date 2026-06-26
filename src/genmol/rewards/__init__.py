"""Reward / forward operator registry for GenMol guided generation.

Each reward is a callable:  List[str] -> torch.Tensor  of scores.

Built-in rewards (by oracle cost):
    - qed, logp, mw, tpsa   — near-zero cost (RDKit)
    - flash_affinity                     — ~100-500 ms/mol  (neural binding)
    - boltz                              — ~3-10 s/mol      (structure-based)

Plug in your own oracle
-----------------------
Option 1 — subclass RewardFunction (recommended for per-molecule scoring)::

    from genmol.rewards import RewardFunction, register_reward

    class MyOracle(RewardFunction):
        def _score_mol(self, mol) -> float:
            ...  # mol is an RDKit Mol object
            return my_score

    register_reward("my_oracle", MyOracle)

Option 2 — register any callable that maps List[str] -> Tensor directly::

    def my_batch_oracle(smiles_list):
        ...
        return torch.tensor(scores)

    register_reward("my_oracle", my_batch_oracle)

Then use it in any run script::

    python scripts/run_active_loop.py --oracle custom --protein_id TARGET         --custom_oracle_module mypackage.mymodule --custom_oracle_class MyOracle
"""

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
from .kl_penalty import KLPenalizedReward

# ── Registry ──────────────────────────────────────────────────────────

REWARD_REGISTRY: dict = {
    "mw": MolecularWeightReward,
    "qed": QEDReward,
    "logp": LogPReward,
    "tpsa": TPSAReward,
    # Expensive rewards — lazy-loaded to avoid heavy imports when not needed.
    "flash_affinity": "genmol.rewards.flash_affinity.FlashAffinityForwardOp",
    "boltz": "genmol.rewards.boltz.BoltzAffinityReward",
}


def register_reward(name: str, cls_or_fn) -> None:
    """Register a custom reward under *name* so it can be retrieved via :func:.

    Args:
        name:       Short name used in config / CLI (e.g. "my_oracle").
        cls_or_fn:  A class (will be instantiated with kwargs from :func:)
                    or any callable List[str] -> torch.Tensor.

    Example::

        register_reward("my_dock", MyDockingOracle)
        oracle = get_reward("my_dock", protein_pdb="target.pdb")
    """
    REWARD_REGISTRY[name.lower().strip()] = cls_or_fn


def get_reward(name: str, **kwargs):
    """Instantiate a reward by short name.

    For parameterised rewards pass constructor kwargs directly::

        get_reward("flash_affinity", protein_id="2VT4", task="binary")
        get_reward("my_oracle", some_param=value)

    Returns:
        An instance of the reward (or None if *name* is "none").
    """
    key = name.lower().strip()
    if key in ("none", ""):
        return None
    entry = REWARD_REGISTRY.get(key)
    if entry is None:
        raise ValueError(
            f"Unknown reward {name!r}.  "
            f"Available: {', '.join(REWARD_REGISTRY)}.  "
            f"Register custom rewards with register_reward()."
        )
    if isinstance(entry, str):
        import importlib
        module_path, class_name = entry.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
    else:
        cls = entry
    # If it's a plain callable (not a class), return it directly
    import inspect
    if inspect.isclass(cls):
        return cls(**kwargs)
    return cls


__all__ = [
    "RewardFunction",
    "MolecularWeightReward",
    "QEDReward",
    "LogPReward",
    "TPSAReward",
    "MolecularWeightForwardOp",
    "QEDForwardOp",
    "KLPenalizedReward",
    "REWARD_REGISTRY",
    "get_reward",
    "register_reward",
]
