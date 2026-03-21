"""ThresholdReward: max(r(x) - t, 0) reward wrapper.

Implements the threshold-clipped reward from the "best-arm identification
under budget constraints" framework. The threshold t is set to the 80th
percentile of all rewards seen so far, updated every `update_every` batch
calls. Only molecules that beat the current threshold contribute gradient
(positive reward); the rest receive zero reward.

Reference: arxiv 2602.16796
"""

import numpy as np
import torch


class ThresholdReward:
    """Wraps any reward fn with max(r(x) - t, 0) clipping.

    t is updated every `update_every` calls to this reward to be the
    `percentile`-th percentile of all valid rewards seen so far. This
    naturally focuses training on the top (100 - percentile)% molecules.

    By default: percentile=80, so t splits top 20% from bottom 80%.
    With DDPP, only molecules scoring above t contribute positive reward,
    which means only top-20% molecules accumulate in the replay buffer
    (since ReplayBuffer stores entries where reward > 0).

    Args:
        base_reward_fn:  any callable smiles_list -> torch.Tensor [N]
        update_every:    update t after this many __call__ invocations
        percentile:      percentile for threshold (default 80 = top 20%)
        min_samples:     minimum samples before threshold becomes non-zero
    """

    def __init__(self, base_reward_fn, update_every: int = 10,
                 percentile: float = 80.0, min_samples: int = 20):
        self.base = base_reward_fn
        self.update_every = update_every
        self.percentile = percentile
        self.min_samples = min_samples

        self.threshold = 0.0
        self._all_rewards: list[float] = []
        self._call_count = 0

    def update_threshold(self):
        """Force a threshold update from accumulated rewards."""
        if len(self._all_rewards) >= self.min_samples:
            self.threshold = float(
                np.percentile(self._all_rewards, self.percentile)
            )

    def __call__(self, smiles_list):
        """Returns max(r(x) - t, 0), with -inf for invalid molecules."""
        scores = self.base(smiles_list)

        # Accumulate valid rewards for threshold tracking
        valid_mask = scores.isfinite()
        self._all_rewards.extend(scores[valid_mask].tolist())

        self._call_count += 1
        if (self._call_count % self.update_every == 0
                and len(self._all_rewards) >= self.min_samples):
            self.threshold = float(
                np.percentile(self._all_rewards, self.percentile)
            )

        # Apply threshold clipping
        clipped = torch.clamp(scores - self.threshold, min=0.0)
        clipped[~valid_mask] = float("-inf")
        return clipped
