"""Sequence surrogate for active learning in GenMol.

SequenceSurrogate encodes SMILES as mean-pooled character-level one-hot vectors,
then passes through a 4-layer MLP (hidden_dim=64). Designed for online use:
fast inference, fast retraining on small oracle-labeled datasets.

Same __call__ interface as genmol.rewards.RewardFunction:
    surrogate(smiles_list) -> torch.Tensor [N]
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# Standard SMILES character vocabulary (covers all common organic molecules)
_SMILES_CHARS = list(
    "BCFHINOPS"           # uppercase atoms (single-char)
    "bcnos"               # aromatic atoms
    "ClBrSi"              # two-char atoms flattened (handled in tokenizer)
    "()[]=#+-./@\%"      # bonds, rings, charges, stereo
    "0123456789"          # ring closure digits
    " "                   # padding sentinel
)
# Build a clean unique-ordered set
_CHAR2IDX: dict[str, int] = {}
for _c in _SMILES_CHARS:
    if _c not in _CHAR2IDX:
        _CHAR2IDX[_c] = len(_CHAR2IDX)
_UNK_IDX = len(_CHAR2IDX)   # index for unknown chars
VOCAB_SIZE = len(_CHAR2IDX) + 1  # +1 for UNK


def smiles_to_onehot(smiles_list: list[str]) -> torch.Tensor:
    """Encode a list of SMILES as mean-pooled one-hot vectors.

    Returns: float32 tensor [N, VOCAB_SIZE]
    """
    out = torch.zeros(len(smiles_list), VOCAB_SIZE)
    for i, smi in enumerate(smiles_list):
        if not smi:
            continue
        counts = torch.zeros(VOCAB_SIZE)
        n = 0
        j = 0
        while j < len(smi):
            # Try two-char tokens first (Cl, Br, Si, ...)
            two = smi[j:j+2]
            if two in _CHAR2IDX:
                counts[_CHAR2IDX[two]] += 1
                j += 2
            else:
                one = smi[j]
                idx = _CHAR2IDX.get(one, _UNK_IDX)
                counts[idx] += 1
                j += 1
            n += 1
        if n > 0:
            out[i] = counts / n   # mean-pool (token frequencies)
    return out


class SequenceSurrogate(nn.Module):
    """4-layer MLP on mean-pooled character one-hot sequences.

    Architecture:
        [VOCAB_SIZE] -> Linear(64) -> ReLU -> Linear(64) -> ReLU
                     -> Linear(64) -> ReLU -> Linear(64) -> ReLU -> Linear(1)

    That is 4 hidden layers of width 64, followed by a scalar head.
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = nn.Sequential(
            nn.Linear(VOCAB_SIZE, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(self.device)

        self.optimizer = Adam(self.parameters(), lr=1e-3)
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    # ── training ───────────────────────────────────────────────────────

    def fit(self, smiles_list: list[str], scores: list,
            n_epochs: int = 100, batch_size: int = 32):
        """Train the MLP on (smiles, score) pairs.

        Filters out None / non-finite scores. No-ops if < 4 valid points.
        Subsequent calls continue training from current weights (warm-start).
        """
        scores_arr = np.array(
            [float(s) if s is not None else float("nan") for s in scores],
            dtype=np.float32,
        )
        valid = np.isfinite(scores_arr)
        if valid.sum() < 4:
            return

        smiles_v = [s for s, v in zip(smiles_list, valid) if v]
        y = torch.tensor(scores_arr[valid], dtype=torch.float32,
                         device=self.device)

        self.train()
        for _ in range(n_epochs):
            perm = torch.randperm(len(smiles_v))
            for start in range(0, len(smiles_v), batch_size):
                idx = perm[start : start + batch_size]
                batch_smi = [smiles_v[i] for i in idx.tolist()]
                batch_y = y[idx]
                X = smiles_to_onehot(batch_smi).to(self.device)
                pred = self.net(X).squeeze(-1)
                loss = F.mse_loss(pred, batch_y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self._is_fitted = True

    # ── inference ──────────────────────────────────────────────────────

    def predict(self, smiles_list: list[str]) -> np.ndarray:
        """Return numpy float32 array of predictions."""
        self.eval()
        with torch.no_grad():
            X = smiles_to_onehot(smiles_list).to(self.device)
            preds = self.net(X).squeeze(-1)
        return preds.cpu().numpy().astype(np.float32)

    def __call__(self, smiles_list: list[str]) -> torch.Tensor:
        """Reward-compatible interface. Returns torch.Tensor [N].

        Invalid / empty SMILES are marked -inf.
        Returns zeros (not -inf) for all if not yet fitted, so the sampler
        treats all candidates as equally neutral before training begins.
        """
        if not self._is_fitted:
            return torch.zeros(len(smiles_list))
        preds = self.predict(smiles_list)
        t = torch.tensor(preds, dtype=torch.float32)
        for i, smi in enumerate(smiles_list):
            if not smi:
                t[i] = float("-inf")
        return t
