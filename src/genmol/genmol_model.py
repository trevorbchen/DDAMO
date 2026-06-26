"""GenMolGenerativeModel -- wraps the existing GenMol checkpoint as a GenerativeModel.

This is the concrete implementation used when running with the default GenMol
masked-diffusion backbone.  Users who want a different backbone should instead
subclass ``GenerativeModel`` directly (see ``examples/custom_model.py``).
"""

from __future__ import annotations

from typing import List

import torch

from genmol.base_model import GenerativeModel


class GenMolGenerativeModel(GenerativeModel):
    """Wraps a loaded GenMol instance to satisfy the GenerativeModel interface.

    Args:
        genmol_instance: A loaded GenMol Lightning module.
        tokenizer:       The HuggingFace tokenizer attached to the model.
        device:          Device the model lives on.
        max_len:         Maximum sequence length for tokenization.
    """

    def __init__(self, genmol_instance, tokenizer, device: str, max_len: int = 128):
        self._model = genmol_instance
        self._tokenizer = tokenizer
        self._device = device
        self._max_len = max_len

    # ── GenerativeModel interface ─────────────────────────────────────────

    def generate(self, n: int, **kwargs) -> List[str]:
        """Unconditional fallback. For fine-tuned GenMol, call trainer.generate() instead."""
        from genmol.samplers.base import BaseSampler
        return BaseSampler(self._model).sample(n, **kwargs)


    @torch.no_grad()
    def get_embeddings(self, smiles: List[str]) -> torch.Tensor:
        """Mean-pool BERT last hidden states over non-pad positions."""
        H = self._model.backbone.bert.config.hidden_size
        valid_mask = [isinstance(s, str) and len(s) > 0 for s in smiles]
        valid_idx = [i for i, ok in enumerate(valid_mask) if ok]
        if not valid_idx:
            return torch.zeros((len(smiles), H), device=self._device)

        safe_list = [smiles[i] for i in valid_idx]
        enc = self._tokenizer(
            safe_list,
            padding=True,
            truncation=True,
            max_length=self._max_len,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self._device)
        attention_mask = enc["attention_mask"].to(self._device)

        outputs = self._model.backbone.bert(input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # [N, L, H]
        mask_f = attention_mask.unsqueeze(-1).float()
        embeddings = (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1.0)

        full = torch.zeros(
            (len(smiles), H), device=embeddings.device, dtype=embeddings.dtype
        )
        full[valid_idx] = embeddings
        return full

    @property
    def embedding_dim(self) -> int:
        return self._model.backbone.bert.config.hidden_size

    @property
    def device(self) -> str:
        return self._device

    def set_eval(self) -> None:
        self._model.backbone.eval()

    def set_train(self) -> None:
        self._model.backbone.train()

    # ── Pass-through access for code that still needs the raw model ───────

    @property
    def backbone(self):
        """Direct access to the BERT backbone (for trainer internals)."""
        return self._model.backbone

    @property
    def raw(self):
        """The underlying GenMol LightningModule."""
        return self._model
