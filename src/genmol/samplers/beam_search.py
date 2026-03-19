import math
import os
import warnings
import torch
from rdkit import Chem
from rdkit.Chem import QED, DataStructs
from rdkit.Chem import rdMolDescriptors

from .base import Sampler, decode_smiles
from genmol.rewards import QEDForwardOp  # noqa: F401 — re-exported for backward-compat


os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")


class EliteBuffer:
    """
    Fixed-size buffer that retains the highest-reward molecules seen
    across all beam search rollouts.

    Optionally filters out near-duplicate entries via Tanimoto similarity.

    Args:
        max_size:         Maximum number of (smiles, reward) pairs to keep.
        diversity_cutoff: If set (0–1), a new molecule is only inserted when
                          its max Tanimoto similarity to existing buffer
                          members is below this threshold.  Set to None to
                          disable the diversity check.
    """

    def __init__(self, max_size: int, diversity_cutoff: float = None):
        self.max_size = max_size
        self.diversity_cutoff = diversity_cutoff
        self.buffer = []  # sorted ascending by reward; index 0 = lowest
        # TODO: switch to heapq or bisect.insort if max_size grows large

    # -- public API --------------------------------------------------------

    def update(self, smiles: str, reward: float) -> bool:
        """
        Try to insert (smiles, reward).  Returns True if inserted.
        Insertion rules:
          - Buffer not full → always insert (subject to diversity check).
          - Buffer full → insert only if reward > current minimum reward
            (subject to diversity check), replacing that minimum.
        """
        if not smiles:
            return False

        # Diversity check: skip if too similar to anything already in buffer
        if self.diversity_cutoff is not None and self.buffer:
            if self._too_similar(smiles):
                return False

        if len(self.buffer) < self.max_size:
            self.buffer.append((smiles, reward))
            self.buffer.sort(key=lambda x: x[1])
            return True

        if reward > self.buffer[0][1]:          # better than current worst
            self.buffer[0] = (smiles, reward)
            self.buffer.sort(key=lambda x: x[1])
            return True

        return False

    def best_smiles(self):
        """Return all SMILES sorted best-first (highest reward first)."""
        return [smi for smi, _ in reversed(self.buffer)]

    def __len__(self):
        return len(self.buffer)

    # -- private -----------------------------------------------------------

    def _too_similar(self, smiles: str) -> bool:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        for existing_smi, _ in self.buffer:
            existing_mol = Chem.MolFromSmiles(existing_smi)
            if existing_mol is None:
                continue
            existing_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                existing_mol, radius=2, nBits=2048
            )
            if DataStructs.TanimotoSimilarity(fp, existing_fp) >= self.diversity_cutoff:
                return True
        return False


class BeamSearchSampler(Sampler):
    """
    Beam Search sampler for GenMol (MDLM), adapted from §3.4 of Complexa
    (ICLR 2026) to discrete confidence-based unmasking.

    Algorithm:
      Maintain N beam trajectories.  Every K denoising steps:
        1. Branch: from each of the N beams, sample L children (K more steps each).
        2. Rollout: run every N×L candidate to completion (disposable).
        3. Score: evaluate completed sequences with forward_op.
        4. Prune: keep the top-N candidates as the new beam.
      Repeat until fully denoised, then return the beam molecules.

    Beam collapse in MDLM:
      MDLM stochasticity comes from two sources:
        - Layer 1 (token selection): Categorical(logits/temp).sample()
          -> low temp ≈ argmax, children become near-identical.
        - Layer 2 (position selection): Gumbel noise on log-confidence,
          scaled by randomness * (1 - step/total_steps).
          -> noise anneals to zero by end of denoising.
      Both vanish toward the end of denoising, so children branched from the
      same parent at mid-to-late steps tend to produce near-identical sequences.
      This compounds with reward pressure pulling all beams toward the same
      high-scoring scaffolds. Empirically: bare beam -> uniqueness ~67%;
      adding elite buffer recovers uniqueness to ~99.7%.

      Implemented mitigations:
        - elite_buffer: preserves the best unique molecules across all rollouts,
          recovering uniqueness even when the beam itself has collapsed.
        - diversity_penalty (lambda): soft diverse beam search -- penalizes
          pruning scores by max Tanimoto similarity to other selected members.
          Degrades gracefully at high budget, unlike hard diversity_cutoff which
          rejects too many candidates and forces the beam to accept low-quality
          fill (QED collapsed to 0.833 at budget=200 with hard cutoff).

      Possible future improvements (not yet implemented):
        - Entropy-adaptive branching: monitor BERT output entropy over MASK
          positions, branch only when entropy is high (multiple positions with
          similar confidence). Skip branching when model is confident -- children
          would be near-identical anyway. Concentrates budget on decision points
          that actually matter.
        - Position-level branching: instead of relying on Gumbel noise for
          diversity (which anneals away), directly enumerate different position
          subsets to unmask. E.g. if 20 MASKs remain and schedule says unmask 5,
          each child unmasks a different top-5 subset. Diversity comes from
          combinatorics, completely independent of annealing.

    Batching note:
      Each beam run processes [N*L, seq_len] in one forward pass. This is a
      real algorithmic advantage over MCTS: beam branching/pruning is
      embarrassingly parallel, while MCTS iterations are inherently sequential
      (UCB depends on backpropagated visit counts from prior iterations).
      At matched forward-pass budget, beam walltime << MCTS walltime.
    """

    def __init__(
        self,
        path,
        beam_width: int = 8,
        branching_factor: int = 4,
        steps_per_interval: int = None,  # None → total_steps // 4 (≈4 updates, matching Complexa)
        forward_op=None,
        elite_buffer_size: int = None,
        diversity_cutoff: float = None,
        diversity_penalty: float = 0.0,
        **kwargs,
    ):
        super().__init__(path, **kwargs)
        self.beam_width = beam_width                  # N
        self.branching_factor = branching_factor      # L
        self.steps_per_interval = steps_per_interval  # K
        self.forward_op = forward_op or QEDForwardOp()
        # Optional elite buffer (None = disabled)
        self.elite_buffer_size = elite_buffer_size
        self.diversity_cutoff = diversity_cutoff
        self.diversity_penalty = diversity_penalty    # λ for Tanimoto penalty

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decode_smiles(self, x, fix=True):
        return decode_smiles(self.model, x, fix=fix)

    @torch.no_grad()
    def _rollout(self, x, start_step: int, total_steps: int,
                 softmax_temp: float, randomness: float):
        """
        Complete denoising from start_step → total_steps.
        Returns (denoised token tensor, forward_pass_count).

        This is the "simulate" half of branch-and-bound: given a partially
        unmasked sequence, run MDLM's confidence-based unmasking to completion
        so we can decode and score it. The result is disposable — we only keep
        the reward, not the rolled-out tokens (the beam retains the partial
        sequences at the branch point).
        """
        fp = 0
        for i in range(start_step, total_steps):
            attention_mask = x != self.pad_index
            logits = self.model(x, attention_mask)
            fp += x.shape[0]  # one forward pass per sequence in the batch
            x = self.mdlm.step_confidence(
                logits, x, i, total_steps, softmax_temp, randomness
            )
        return x, fp

    def _score_candidates(self, candidates, current_step, total_steps,
                          softmax_temp, randomness):
        """
        Rollout all candidates to completion, decode, and score with forward_op.
        Returns (scores tensor [N], smiles list [N], forward_pass_count).
        """
        rolled, fp = self._rollout(
            candidates.clone(), current_step, total_steps, softmax_temp, randomness
        )
        smiles = self._decode_smiles(rolled)
        scores = self.forward_op(smiles).to(candidates.device)
        return scores, smiles, fp

    def _diversity_penalized_scores(self, scores, smiles_list):
        """Soft diverse beam search: s_i = R(c_i) - λ * max_j Tanimoto(c_i, c_j).

        Preferred over hard diversity_cutoff which collapsed at high budget
        (QED dropped to 0.833 at budget=200). The soft penalty degrades
        gracefully because it never outright rejects candidates.
        """
        if self.diversity_penalty <= 0:
            return scores

        n = len(smiles_list)
        # Compute Morgan fingerprints
        fps = []
        for smi in smiles_list:
            if smi:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    fps.append(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
                else:
                    fps.append(None)
            else:
                fps.append(None)

        penalties = torch.zeros(n)
        for i in range(n):
            if fps[i] is None:
                continue
            max_sim = 0.0
            for j in range(n):
                if i == j or fps[j] is None:
                    continue
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                if sim > max_sim:
                    max_sim = sim
            penalties[i] = max_sim

        return scores - self.diversity_penalty * penalties.to(scores.device)

    # ------------------------------------------------------------------
    # Generation entry point
    # ------------------------------------------------------------------

    def _run_single_beam(self, N, L, K, total_steps, softmax_temp, randomness,
                         min_add_len, elite):
        """Run one beam search of width N. Returns (smiles_list, reward_evals, fp_count)."""
        # Start from a fully masked prototype: [BOS] [MASK...] [EOS]
        x_proto = torch.hstack([
            torch.full((1, 1), self.model.bos_index),
            torch.full((1, 1), self.model.eos_index),
        ])
        x_proto = self._insert_mask(x_proto, num_samples=1, min_add_len=min_add_len)
        x_proto = x_proto.to(self.model.device)
        beam = x_proto.repeat(N, 1)  # N identical copies to start

        step = 0
        reward_evals = 0
        fp_count = 0
        last_logged_fp = 0
        last_logged_re = 0
        while step < total_steps:
            k = min(K, total_steps - step)

            # --- Branch: expand each beam into L children via K denoising steps ---
            candidates = beam.repeat_interleave(L, dim=0)  # [N*L, seq_len]
            for j in range(k):
                attention_mask = candidates != self.pad_index
                logits = self.model(candidates, attention_mask)
                fp_count += candidates.shape[0]
                candidates = self.mdlm.step_confidence(
                    logits, candidates, step + j, total_steps, softmax_temp, randomness
                )
            step += k

            # --- Score: rollout each candidate to completion, then evaluate ---
            # At the final interval we're already fully denoised, so skip rollout.
            if step < total_steps:
                scores, rollout_smiles, rollout_fp = self._score_candidates(
                    candidates, step, total_steps, softmax_temp, randomness
                )
                fp_count += rollout_fp
            else:
                rollout_smiles = self._decode_smiles(candidates)
                scores = self.forward_op(rollout_smiles).to(candidates.device)
            reward_evals += len(candidates)

            # Trajectory logging
            self._log_point(
                reward_evals - last_logged_re,
                fp_count - last_logged_fp,
                scores.max().item(),
            )
            last_logged_re = reward_evals
            last_logged_fp = fp_count

            # Elite buffer sees raw (unpenalized) scores — it's a global best-of
            # across all iterations, independent of the diversity pressure applied
            # during pruning.
            if elite is not None:
                for smi, r in zip(rollout_smiles, scores.tolist()):
                    elite.update(smi, r)

            # --- Prune: keep top-N candidates (with optional diversity penalty) ---
            pruning_scores = self._diversity_penalized_scores(scores, rollout_smiles)
            n_keep = min(N, candidates.shape[0])
            top_idx = pruning_scores.topk(n_keep).indices
            beam = candidates[top_idx]

        # Merge elite buffer (best unique molecules across all rollouts) with
        # the final beam. Elite-first ordering means high-reward molecules from
        # earlier rollouts aren't lost to late-stage beam collapse.
        beam_smiles = [s for s in self._decode_smiles(beam) if s]
        if elite is not None:
            seen = set()
            result = []
            for s in elite.best_smiles() + beam_smiles:
                if s and s not in seen:
                    seen.add(s)
                    result.append(s)
        else:
            result = beam_smiles

        return result, reward_evals, fp_count

    @torch.no_grad()
    def de_novo_generation(
        self,
        num_samples: int = 1,
        softmax_temp: float = 0.8,
        randomness: float = 0.5,
        min_add_len: int = 40,
        beam_width: int = None,
        branching_factor: int = None,
        steps_per_interval: int = None,
        **kwargs,
    ):
        N = beam_width or self.beam_width
        L = branching_factor or self.branching_factor
        _K = steps_per_interval if steps_per_interval is not None else self.steps_per_interval

        # total_steps depends on sequence length (set by _insert_mask).
        # We probe it once here; each _run_single_beam call creates its own
        # prototype, which may differ slightly in length due to randomness in
        # _insert_mask, but total_steps is typically stable.
        x_proto = torch.hstack([
            torch.full((1, 1), self.model.bos_index),
            torch.full((1, 1), self.model.eos_index),
        ])
        x_proto = self._insert_mask(x_proto, num_samples=1, min_add_len=min_add_len)
        x_proto = x_proto.to(self.model.device)
        total_steps = max(self.mdlm.get_num_steps_confidence(x_proto.repeat(N, 1)), 2)
        # Default K ≈ T/4 gives ~4 branch-prune cycles, matching Complexa §3.4
        K = _K if _K is not None else max(1, total_steps // 4)

        self._reset_trajectory()

        # Each beam run yields up to N molecules; run enough beams to cover
        # the requested num_samples. Each run has its own elite buffer so that
        # diversity pressure is independent across runs.
        n_runs = math.ceil(num_samples / N)
        all_smiles = []
        total_reward_evals = 0
        total_fp = 0

        for _ in range(n_runs):
            elite = (
                EliteBuffer(self.elite_buffer_size, self.diversity_cutoff)
                if self.elite_buffer_size else None
            )
            run_smiles, re, fp = self._run_single_beam(
                N, L, K, total_steps, softmax_temp, randomness, min_add_len, elite
            )
            all_smiles.extend(run_smiles)
            total_reward_evals += re
            total_fp += fp

        # Deduplicate and score to return top num_samples
        seen = set()
        final_smiles = []
        for s in all_smiles:
            if s and s not in seen:
                seen.add(s)
                final_smiles.append(s)

        # Budget = total rollouts / num_samples (consistent with MCTS)
        self.last_reward_evals = total_reward_evals
        self.last_budget_per_sample = total_reward_evals / max(num_samples, 1)
        self.last_forward_passes = total_fp
        self.last_fp_per_sample = total_fp / max(num_samples, 1)
        return final_smiles[:num_samples]
