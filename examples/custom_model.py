"""
Custom Model Example
====================
Shows how to plug any generative model into the GenMol active loop
by implementing the GenerativeModel and FinetuneTrainer interfaces.

The active loop (run_active_loop) only calls:
    trainer.generate(n)
    trainer.model.get_embeddings(smiles)
    trainer.model.set_eval() / set_train()
    trainer.add_scored_molecules(smiles, scores)
    trainer.train(max_steps)

Everything else (architecture, tokenizer, fine-tuning algorithm) is
entirely up to you.

Usage:
    from examples.custom_model import MyTrainer
    from genmol.active_loop import ActiveLoopConfig, run_active_loop

    trainer = MyTrainer(model_path="my_checkpoint.pt")
    run_active_loop(trainer, oracle_fn=my_oracle, cfg=ActiveLoopConfig())
"""

from __future__ import annotations
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from genmol.base_model import GenerativeModel, FinetuneTrainer


# ==========================================================================
# Step 1 — Implement GenerativeModel
# ==========================================================================

class MyGenerativeModel(GenerativeModel):
    """Replace this stub with your actual model.

    The only hard requirements are:
    - generate() returns a list of SMILES strings
    - get_embeddings() returns a [N, D] float tensor
    - embedding_dim matches D
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self._device = device
        self._embedding_dim = 256   # set to match your model's hidden size

        # Load your model here, e.g.:
        #   self.backbone = MyModel.from_pretrained(checkpoint_path)
        #   self.backbone.to(device)
        self.backbone = nn.Linear(1, self._embedding_dim)   # placeholder

        self.backbone.to(device)

    # ── Required interface ────────────────────────────────────────────────

    def generate(self, n: int, **kwargs) -> List[str]:
        """Sample n molecules from your model.

        Implement your generation logic here (beam search, sampling, etc.).
        Invalid / duplicate SMILES are fine — the active loop handles them.
        """
        # ---- replace with your generation logic ----
        return ["CCO"] * n   # placeholder
        # --------------------------------------------

    @torch.no_grad()
    def get_embeddings(self, smiles: List[str]) -> torch.Tensor:
        """Encode SMILES to vectors for the surrogate ensemble.

        The surrogate needs fixed-size, informative embeddings.
        Good choices: last hidden states, pooled encoder outputs, fingerprints.
        """
        # ---- replace with your encoder ----
        n = len(smiles)
        return torch.zeros(n, self._embedding_dim, device=self._device)
        # -----------------------------------

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def device(self) -> str:
        return self._device

    # Optional — only needed if your model has explicit train/eval modes
    def set_eval(self):
        self.backbone.eval()

    def set_train(self):
        self.backbone.train()


# ==========================================================================
# Step 2 — Implement FinetuneTrainer
# ==========================================================================

class MyTrainer(FinetuneTrainer):
    """Wraps MyGenerativeModel with a simple fine-tuning loop.

    Replace the buffer and train() body with your actual training algorithm
    (RLHF, DPO, GFlowNet, etc.).
    """

    def __init__(self, model_path: str, device: str = "cuda", lr: float = 1e-4):
        self._model = MyGenerativeModel(model_path, device)
        self._optimizer = torch.optim.Adam(
            self._model.backbone.parameters(), lr=lr
        )
        self._buffer: list[tuple[str, float]] = []   # (smiles, score)
        self._step = 0

    # ── FinetuneTrainer interface ────────────────────────────────────────

    @property
    def model(self) -> GenerativeModel:
        return self._model

    def add_scored_molecules(self, smiles: List[str], scores: List[float]) -> None:
        """Add oracle-scored molecules to the replay buffer."""
        for smi, sc in zip(smiles, scores):
            if smi and sc is not None:
                self._buffer.append((smi, float(sc)))

    def train(self, max_steps: int, **kwargs) -> None:
        """Fine-tune the model on the current buffer for max_steps steps.

        Replace this body with your actual training algorithm.
        """
        if not self._buffer:
            return
        for _ in range(max_steps):
            # ---- replace with your training logic ----
            # Example sketch for REINFORCE-style update:
            #   smiles, scores = sample_batch(self._buffer)
            #   log_probs = self._model.log_prob(smiles)
            #   loss = -(log_probs * scores).mean()
            #   self._optimizer.zero_grad()
            #   loss.backward()
            #   self._optimizer.step()
            self._step += 1
            # ------------------------------------------

    @property
    def step(self) -> int:
        return self._step


# ==========================================================================
# Run with the active loop
# ==========================================================================

if __name__ == "__main__":
    from genmol.active_loop import ActiveLoopConfig, run_active_loop
    from genmol.rewards import get_reward

    oracle = get_reward("qed")   # swap in any oracle

    trainer = MyTrainer(model_path="model_v2.ckpt")

    cfg = ActiveLoopConfig(
        n_epochs=5,
        M=100,
        K=10,
        output_dir="outputs/custom_model_test",
    )

    run_active_loop(
        ddpp_trainer=trainer,
        fa_oracle=lambda smiles: oracle(smiles).tolist(),
        cfg=cfg,
    )
