"""Abstract interfaces for swapping in alternative generative models.

The default and primary setup is GenMol (masked discrete diffusion over SAFE
representations) driven by ``DDPPLBTrainer`` or ``VIDDTrainer``.  These ABCs
exist so researchers can drop in a different backbone without forking the loop.

Default GenMol usage (no changes needed)::

    trainer = DDPPLBTrainer(model_path="model_v2.ckpt", ...)
    run_active_loop(ddpp_trainer=trainer, fa_oracle=oracle, cfg=cfg)

Custom backbone (implement these ABCs, then pass your trainer in)::

    class MyModel(GenerativeModel):
        def generate(self, n, **kwargs): ...
        def get_embeddings(self, smiles): ...
        @property
        def embedding_dim(self): return 512
        @property
        def device(self): return "cuda"

    class MyTrainer(FinetuneTrainer):
        @property
        def model(self): return self._model
        def add_scored_molecules(self, smiles, scores): ...
        def train(self, max_steps, **kwargs): ...

See ``examples/custom_model.py`` for a full template.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import torch


class GenerativeModel(ABC):
    """Interface for a molecule generative model.

    Subclass this to plug any backbone (autoregressive LM, flow model,
    VAE, diffusion model, ...) into the GenMol active loop.
    """

    @abstractmethod
    def generate(self, n: int, **kwargs) -> List[str]:
        """Sample *n* molecules.

        Args:
            n: Number of molecules to generate.
            **kwargs: Architecture-specific generation parameters
                      (temperature, beam width, etc.).  Unused kwargs
                      should be silently ignored.

        Returns:
            List of SMILES strings.  Invalid molecules are allowed;
            the active loop filters them.
        """

    @abstractmethod
    def get_embeddings(self, smiles: List[str]) -> torch.Tensor:
        """Encode molecules to fixed-size vectors for the surrogate ensemble.

        Args:
            smiles: List of SMILES strings (may contain invalid entries;
                    return zero vectors for those).

        Returns:
            Tensor of shape ``[len(smiles), embedding_dim]``.
        """

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of vectors returned by ``get_embeddings``."""

    @property
    @abstractmethod
    def device(self) -> str:
        """Device the model lives on (e.g. ``"cuda"`` or ``"cpu"``)."""

    # Optional hooks — override if your model needs explicit mode switching
    def set_eval(self) -> None:
        """Switch to evaluation mode (called before embedding extraction)."""

    def set_train(self) -> None:
        """Switch back to training mode (called after embedding extraction)."""


class FinetuneTrainer(ABC):
    """Interface for a fine-tuner that wraps a ``GenerativeModel``.

    Encapsulates the replay buffer, optimizer, and gradient updates so
    that ``run_active_loop`` stays model-agnostic.
    """

    @property
    @abstractmethod
    def model(self) -> GenerativeModel:
        """The underlying ``GenerativeModel`` being fine-tuned."""

    @abstractmethod
    def add_scored_molecules(self, smiles: List[str], scores: List[float]) -> None:
        """Add oracle-scored molecules to the replay buffer.

        Args:
            smiles: SMILES strings evaluated by the oracle.
            scores: Corresponding oracle scores (higher = better).
        """

    @abstractmethod
    def train(self, max_steps: int, **kwargs) -> None:
        """Run ``max_steps`` gradient updates using the current buffer."""

    # ── Convenience delegates (no need to override) ──────────────────────

    def generate(self, n: int, **kwargs) -> List[str]:
        return self.model.generate(n, **kwargs)

    def get_embeddings(self, smiles: List[str]) -> torch.Tensor:
        return self.model.get_embeddings(smiles)

    @property
    def embedding_dim(self) -> int:
        return self.model.embedding_dim

    @property
    def device(self) -> str:
        return self.model.device

    # Optional — trainers that track a step counter should override this
    @property
    def step(self) -> int:
        return 0
