"""
Reward / forward operator functions for guided molecular generation.

Each reward is a callable: List[str] -> torch.Tensor of scores.
Used by DAPSSampler's Metropolis-Hastings step to bias generation
toward molecules with desirable properties.

Usage:
    from genmol.rewards import MolecularWeightReward, QEDReward

    reward = MolecularWeightReward()
    scores = reward(["CCO", "c1ccccc1"])  # tensor([0.046, 0.078])
"""

import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, QED


# ── Helpers ───────────────────────────────────────────────────────────

def _safe_mol(smi):
    """Try to parse SMILES, with a SAFE/bracket-safe fallback."""
    if not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        return mol
    # Fallback: maybe it's still a SAFE or bracket-safe string
    try:
        from genmol.utils.utils_chem import safe_to_smiles
        from genmol.utils.bracket_safe_converter import bracketsafe2safe
        candidate = safe_to_smiles(bracketsafe2safe(smi), fix=True)
        return Chem.MolFromSmiles(candidate) if candidate else None
    except Exception:
        return None


# ── Base class ────────────────────────────────────────────────────────

class RewardFunction:
    """Base class for reward / forward operators.

    Subclasses must implement ``_score_mol(mol) -> float``.
    Invalid molecules receive ``-inf`` automatically.
    An optional ``scale`` divides all *finite* scores to keep MH
    log-ratios in a numerically safe range.
    """

    scale: float = 1.0  # override in subclass if needed

    def _score_mol(self, mol) -> float:
        raise NotImplementedError

    def __call__(self, smiles_list):
        scores = []
        for smi in smiles_list:
            mol = _safe_mol(smi)
            if mol is None:
                scores.append(float("-inf"))
            else:
                try:
                    scores.append(self._score_mol(mol))
                except Exception:
                    scores.append(float("-inf"))
        if self.scale != 1.0:
            scores = [s / self.scale if s != float("-inf") else s for s in scores]
        return torch.tensor(scores, dtype=torch.float32)


# ── Concrete rewards ──────────────────────────────────────────────────

class MolecularWeightReward(RewardFunction):
    """Score = MW / 1000.  Higher MW → higher score."""
    scale = 1000.0

    def _score_mol(self, mol):
        return float(Descriptors.MolWt(mol))


class QEDReward(RewardFunction):
    """Quantitative Estimate of Drug-likeness (0–1)."""

    def _score_mol(self, mol):
        return float(QED.qed(mol))


class LogPReward(RewardFunction):
    """Crippen LogP.  Unscaled — typical range ~ -2 to +8."""

    def _score_mol(self, mol):
        return float(Descriptors.MolLogP(mol))


class TPSAReward(RewardFunction):
    """Topological Polar Surface Area / 200 (normalised to ~0–1)."""
    scale = 200.0

    def _score_mol(self, mol):
        return float(Descriptors.TPSA(mol))


# ── Backward-compat alias ────────────────────────────────────────────
# So that ``genmol.rewards.MolecularWeightForwardOp`` still resolves
# (used by Hydra configs).
MolecularWeightForwardOp = MolecularWeightReward


# ── Registry ──────────────────────────────────────────────────────────

REWARD_REGISTRY = {
    "mw": MolecularWeightReward,
    "qed": QEDReward,
    "logp": LogPReward,
    "tpsa": TPSAReward,
}


def get_reward(name: str) -> RewardFunction:
    """Look up a reward by short name.  Returns an *instance*."""
    key = name.lower().strip()
    if key in ("none", ""):
        return None
    cls = REWARD_REGISTRY.get(key)
    if cls is None:
        raise ValueError(
            f"Unknown reward {name!r}. "
            f"Available: {', '.join(REWARD_REGISTRY)}"
        )
    return cls()
