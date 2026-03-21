"""Lightweight online surrogate for active learning in GenMol.

FingerprintSurrogate wraps a small sklearn MLP on 2048-bit Morgan (ECFP4)
fingerprints. Designed for online use: fast inference (<1ms/mol), fast
retraining (<1s on hundreds of oracle-labeled molecules).

The surrogate is reward-compatible: __call__(smiles_list) -> torch.Tensor,
same interface as RewardFunction subclasses.
"""

import numpy as np
import torch

from rdkit import Chem
from rdkit.Chem import AllChem


class FingerprintSurrogate:
    """Online MLP surrogate on Morgan fingerprints.

    Usage:
        surrogate = FingerprintSurrogate()
        # after accumulating oracle labels:
        surrogate.fit(smiles_list, scores)
        # use as reward (same interface as genmol.rewards.RewardFunction):
        scores_tensor = surrogate(candidate_smiles)
    """

    def __init__(self, radius: int = 2, nbits: int = 2048,
                 hidden_layer_sizes=(512, 128)):
        self.radius = radius
        self.nbits = nbits
        self.hidden_layer_sizes = hidden_layer_sizes
        self._mlp = None          # sklearn MLPRegressor, None until first fit
        self._n_fitted = 0        # number of (smiles, score) pairs seen

    # ── fingerprint helper ─────────────────────────────────────────────

    def _fps(self, smiles_list):
        """SMILES list -> float32 numpy array [N, nbits]."""
        fps = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is None:
                fps.append(np.zeros(self.nbits, dtype=np.float32))
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, self.radius, self.nbits
                )
                fps.append(np.array(fp, dtype=np.float32))
        return np.array(fps, dtype=np.float32)

    # ── training ───────────────────────────────────────────────────────

    def fit(self, smiles_list, scores):
        """Train (or warm-start update) the surrogate on (smiles, score) pairs.

        Filters out None / non-finite scores before fitting.
        No-ops if fewer than 4 valid points (not enough to fit).
        """
        from sklearn.neural_network import MLPRegressor

        scores_arr = np.array(
            [float(s) if s is not None else float("nan") for s in scores],
            dtype=np.float32,
        )
        X = self._fps(smiles_list)
        valid = np.isfinite(scores_arr)
        if valid.sum() < 4:
            return  # not enough data

        X_v, y_v = X[valid], scores_arr[valid]

        if self._mlp is None:
            self._mlp = MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation="relu",
                solver="adam",
                max_iter=500,
                warm_start=True,
                random_state=0,
                n_iter_no_change=20,
            )
        # warm_start=True means subsequent calls continue from current weights
        self._mlp.fit(X_v, y_v)
        self._n_fitted += int(valid.sum())

    # ── inference ──────────────────────────────────────────────────────

    @property
    def is_fitted(self):
        return self._mlp is not None

    def predict(self, smiles_list):
        """Return numpy array of predictions. Returns zeros if not fitted."""
        if not self.is_fitted:
            return np.zeros(len(smiles_list), dtype=np.float32)
        X = self._fps(smiles_list)
        return self._mlp.predict(X).astype(np.float32)

    def __call__(self, smiles_list):
        """Reward-compatible interface. Returns torch.Tensor [N]."""
        preds = self.predict(smiles_list)
        t = torch.tensor(preds, dtype=torch.float32)
        # Mark invalid SMILES (zero fingerprint → often noisy prediction)
        for i, smi in enumerate(smiles_list):
            if not smi or Chem.MolFromSmiles(smi) is None:
                t[i] = float("-inf")
        return t
