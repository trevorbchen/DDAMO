"""Axis 2: Pretrain/finetune algorithms.

Available training methods:
    - DDPPLBTrainer  (Discrete Denoising Posterior Prediction, lower bound)
    - VIDDTrainer    (Iterative Distillation, Wang et al. arXiv 2507.00445)
"""

from .ddpp import DDPPLBTrainer
from .vidd import VIDDTrainer

__all__ = ["DDPPLBTrainer", "VIDDTrainer"]

