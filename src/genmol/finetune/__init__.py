"""Axis 2: Pretrain/finetune algorithms.

Available training methods:
    - DDPPLBTrainer  (Discrete Denoising Posterior Prediction, lower bound)
"""

from .ddpp import DDPPLBTrainer

__all__ = ["DDPPLBTrainer"]
