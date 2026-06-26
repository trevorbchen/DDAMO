"""KL-penalized reward wrapper for inference-time methods.

Wraps any reward with a KL penalty relative to the pretrained model,
making inference-time methods comparable to DDPP which has built-in KL
regularization (log_q - log_p in its loss function).

    guided_score(x) = base_reward(x) + λ * log_p(x)

where log_p(x) is the log-likelihood of molecule x under the pretrained
model (higher = more likely under pretrained dist = less penalty).
"""

import torch
import torch.nn.functional as F


class KLPenalizedReward:
    """Wraps a reward function with a KL penalty from the pretrained model.

    Args:
        base_reward: The underlying reward callable (e.g., FlashAffinityForwardOp).
        model: The pretrained GenMol model (used to compute log-likelihood).
        lam: KL penalty weight. λ = 1/β matches DDPP's reward temperature.
    """

    def __init__(self, base_reward, model, lam=0.01):
        self.base_reward = base_reward
        self.model = model
        self.tokenizer = model.tokenizer
        self.mask_idx = model.mask_index
        self.pad_idx = self.tokenizer.pad_token_id
        self.lam = lam

    def __call__(self, smiles_list):
        """Score molecules with base reward + KL penalty.

        Args:
            smiles_list: list of SMILES strings.

        Returns:
            torch.Tensor of shape [N] with penalized scores.
        """
        # Replace None/empty with placeholder to avoid C++ crashes
        clean = [s if (s and isinstance(s, str) and len(s) > 0) else "C" for s in smiles_list]
        null_mask = [s is None or not isinstance(s, str) or len(s) == 0 for s in smiles_list]

        # Base reward
        base_scores = self.base_reward(clean)
        if not isinstance(base_scores, torch.Tensor):
            base_scores = torch.tensor([float(s) if s is not None else -5.0 for s in base_scores], dtype=torch.float32)

        # KL penalty: log p(x) under pretrained model
        try:
            log_p = self._compute_sequence_log_prob(clean)
            log_p = log_p.to(base_scores.device)
        except Exception:
            log_p = torch.zeros_like(base_scores)

        result = base_scores + self.lam * log_p
        # Zero out null entries
        for i, is_null in enumerate(null_mask):
            if is_null:
                result[i] = -5.0
        return result

    @torch.no_grad()
    def _compute_sequence_log_prob(self, smiles_list):
        """Compute log p(x0 | x_masked) under the pretrained model.

        Tokenizes each SMILES, creates a fully-masked version, and computes
        the sum of log-probs at each position. This is the same computation
        DDPP uses in _compute_denoising_log_prob (ddpp.py:246), applied to
        fully-masked input.

        Returns:
            torch.Tensor of shape [N] with log-likelihood values.
        """
        # Convert SMILES back to SAFE for tokenization
        from genmol.utils.utils_chem import safe_to_smiles
        import safe as sf

        safe_strings = []
        for smi in smiles_list:
            if smi is None:
                safe_strings.append("")
                continue
            try:
                safe_str = sf.encode(smi)
                safe_strings.append(safe_str)
            except Exception:
                safe_strings.append("")

        # Tokenize
        encoded = self.tokenizer(
            safe_strings,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model.config.model.max_position_embeddings,
        )
        x0 = encoded["input_ids"].to(self.model.device)  # [B, L]

        # Create fully masked version (mask all non-special tokens)
        special = {self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, self.pad_idx}
        is_content = torch.ones_like(x0, dtype=torch.bool)
        for tok_id in special:
            if tok_id is not None:
                is_content &= (x0 != tok_id)

        xt = x0.clone()
        xt[is_content] = self.mask_idx

        # Forward pass through pretrained model
        attention_mask = (xt != self.pad_idx).long()
        logits = self.model(xt, attention_mask)               # [B, L, V]
        log_probs = F.log_softmax(logits, dim=-1)             # [B, L, V]

        # Gather log-prob of true token at each masked position
        lp = log_probs.gather(2, x0.unsqueeze(-1)).squeeze(-1)  # [B, L]
        mask = is_content.float()
        return (lp * mask).sum(dim=1)                          # [B]
