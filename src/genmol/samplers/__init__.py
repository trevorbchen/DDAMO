"""Axis 1: Inference-time sampling and search algorithms.

All samplers share the ``Sampler`` base class and expose
``de_novo_generation(num_samples, ...)`` as their main entry point.

Available samplers:
    - Sampler           (unconditional / baseline)
    - BeamSearchSampler (branch-prune with rollout scoring)
    - MCTSSampler       (Monte Carlo tree search with UCB)
    - SMCSampler        (sequential Monte Carlo particle filter)
    - DFKCSampler       (discrete Feynman-Kac corrector)
    - DAPSSampler       (decoupled annealing posterior sampling)
"""

from .base import Sampler, load_model_from_path, decode_smiles
from .beam_search import BeamSearchSampler
from .mcts import MCTSSampler
from .smc import SMCSampler, DFKCSampler
from .daps import DAPSSampler

__all__ = [
    "Sampler",
    "load_model_from_path",
    "BeamSearchSampler",
    "MCTSSampler",
    "SMCSampler",
    "DFKCSampler",
    "DAPSSampler",
    "QEDForwardOp",
    "decode_smiles",
]
