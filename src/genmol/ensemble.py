"""
Ensemble of MLPs trained on backbone embeddings for Thompson Sampling.

Usage in active loop:
    embeddings = encode_smiles(smiles_list, ddpp_trainer.finetuned, ...)
    mu, sigma = ensemble.predict(embeddings)
    sampled = mu + sigma * torch.randn_like(mu)   # Thompson Sampling
    top_k_idx = sampled.topk(K).indices

Reinit each epoch to maintain diversity of beliefs across the 10 MLPs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Backbone embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_smiles(smiles_list: list[str], model, tokenizer=None, device=None, max_len: int = 128) -> torch.Tensor:
    """Encode SMILES to fixed-size embeddings for the surrogate ensemble.

    Accepts either:
    - A ``GenerativeModel`` instance (preferred): ``encode_smiles(smiles, gen_model)``
    - The legacy tuple signature:               ``encode_smiles(smiles, genmol, tokenizer, device)``

    Returns:
        Tensor of shape ``[N, embedding_dim]``.
    """
    from genmol.base_model import GenerativeModel
    if isinstance(model, GenerativeModel):
        return model.get_embeddings(smiles_list)

    # Legacy path: raw GenMol instance + HuggingFace tokenizer
    import torch as _torch
    H = model.backbone.bert.config.hidden_size
    valid_mask = [isinstance(s, str) and len(s) > 0 for s in smiles_list]
    valid_idx = [i for i, ok in enumerate(valid_mask) if ok]
    if not valid_idx:
        return _torch.zeros((len(smiles_list), H), device=device)
    safe_list = [smiles_list[i] for i in valid_idx]
    enc = tokenizer(safe_list, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    outputs = model.backbone.bert(input_ids, attention_mask=attention_mask)
    hidden = outputs.last_hidden_state
    mask_f = attention_mask.unsqueeze(-1).float()
    embeddings = (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1.0)
    full = _torch.zeros((len(smiles_list), embeddings.shape[-1]),
                        device=embeddings.device, dtype=embeddings.dtype)
    full[valid_idx] = embeddings
    return full

# ---------------------------------------------------------------------------
# Single MLP
# ---------------------------------------------------------------------------

class EnsembleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def reinit(self):
        """Reset all weights to fresh random values."""
        for layer in self.net:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()


# ---------------------------------------------------------------------------
# Ensemble of 10 MLPs
# ---------------------------------------------------------------------------

class EnsembleScorer:
    """
    Deep ensemble of `n_models` MLPs for reward prediction + uncertainty.

    Diversity comes from random initialization only (no bootstrapping).
    Reinit every epoch to prevent ensemble collapse over time.

    Args:
        input_dim:   backbone hidden size (e.g. 768 for BERT-base)
        n_models:    number of ensemble members (default 10)
        hidden_dim:  MLP hidden layer size
        device:      torch device
    """

    def __init__(
        self,
        input_dim: int,
        n_models: int = 10,
        hidden_dim: int = 256,
        device: str = "cuda",
    ):
        self.n_models = n_models
        self.device = device
        self.models = nn.ModuleList(
            [EnsembleMLP(input_dim, hidden_dim) for _ in range(n_models)]
        ).to(device)

    @torch.no_grad()
    def predict(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mu:    [N] ensemble mean score
            sigma: [N] ensemble std (uncertainty)
        """
        preds = torch.stack([m(embeddings) for m in self.models], dim=0)  # [K, N]
        return preds.mean(0), preds.std(0)

    @torch.no_grad()
    def thompson_sample(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Sample one score per candidate: s_i ~ N(mu_i, sigma_i²).
        Used to select top-K candidates for oracle evaluation.
        """
        mu, sigma = self.predict(embeddings)
        return mu + sigma * torch.randn_like(mu)

    def reinit_and_train(
        self,
        all_embeddings: torch.Tensor,
        all_scores: torch.Tensor,
        n_epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 256,
    ) -> None:
        """
        Reinitialize all MLPs with fresh random weights, then train on all
        accumulated (embedding, score) data. Called once per active loop epoch.

        Random reinit is critical: it prevents ensemble collapse over time,
        which would cause sigma → 0 and kill Thompson Sampling exploration.

        Args:
            all_embeddings: [N, H] float tensor
            all_scores:     [N]   float tensor (FA scores, any scale)
            n_epochs:        gradient steps per MLP
            lr:              Adam learning rate
            batch_size:      mini-batch size for training
        """
        # Normalise targets to zero mean, unit std for stable MLP training
        mu_s = all_scores.mean()
        std_s = all_scores.std().clamp(min=1e-6)
        y_norm = (all_scores - mu_s) / std_s

        dataset = TensorDataset(
            all_embeddings.detach().cpu().float(),
            y_norm.detach().cpu().float(),
        )
        loader = DataLoader(
            dataset, batch_size=min(batch_size, len(dataset)), shuffle=True, drop_last=False
        )

        for m in self.models:
            m.reinit()
            m.train()
            opt = torch.optim.Adam(m.parameters(), lr=lr)
            for _ in range(n_epochs):
                for xb, yb in loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    opt.zero_grad()
                    F.mse_loss(m(xb), yb).backward()
                    opt.step()
            m.eval()

        # Store normalisation constants so predict() returns scores on original scale
        self._score_mean = mu_s
        self._score_std = std_s

    @torch.no_grad()
    def predict_unnorm(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Like predict(), but returns scores on the original FA scale.
        Only valid after at least one reinit_and_train call.
        """
        mu_norm, sigma_norm = self.predict(embeddings)
        mu = mu_norm * self._score_std + self._score_mean
        sigma = sigma_norm * self._score_std
        return mu, sigma

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def spearman_rho(
        self,
        embeddings: torch.Tensor,
        true_scores: torch.Tensor,
    ) -> float:
        """
        Spearman ρ between ensemble mean predictions and true FA scores.
        Used to track screening quality across epochs.
        """
        if len(true_scores) < 4:
            return float("nan")
        mu, _ = self.predict(embeddings)
        mu_np = mu.detach().cpu().numpy()
        y_np = true_scores.detach().cpu().numpy()
        rho, _ = spearmanr(mu_np, y_np)
        return float(rho)
