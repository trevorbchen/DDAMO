"""Simple molecular property rewards (RDKit-based, near-zero cost)."""

import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, QED


def _safe_mol(smi):
    """Try to parse SMILES, with a SAFE/bracket-safe fallback."""
    if not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        return mol
    try:
        from genmol.utils.utils_chem import safe_to_smiles
        from genmol.utils.bracket_safe_converter import bracketsafe2safe
        candidate = safe_to_smiles(bracketsafe2safe(smi), fix=True)
        return Chem.MolFromSmiles(candidate) if candidate else None
    except Exception:
        return None


class RewardFunction:
    """Base class for reward / forward operators.

    Subclasses implement ``_score_mol(mol) -> float``.
    Invalid molecules receive ``-inf`` automatically.
    """

    scale: float = 1.0

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


class MolecularWeightReward(RewardFunction):
    scale = 1000.0
    def _score_mol(self, mol):
        return float(Descriptors.MolWt(mol))


class QEDReward(RewardFunction):
    def _score_mol(self, mol):
        return float(QED.qed(mol))


class LogPReward(RewardFunction):
    def _score_mol(self, mol):
        return float(Descriptors.MolLogP(mol))


class TPSAReward(RewardFunction):
    scale = 200.0
    def _score_mol(self, mol):
        return float(Descriptors.TPSA(mol))


# backward-compat aliases
MolecularWeightForwardOp = MolecularWeightReward
QEDForwardOp = QEDReward
