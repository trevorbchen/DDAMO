import math
import os
import warnings
import torch

from genmol.sampler import Sampler
from genmol.beam_search_sampler import QEDForwardOp, decode_smiles

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")


class MCTSNode:
    """One node in the MCTS denoising tree."""

    def __init__(self, x, step, parent=None):
        self.x = x            # [1, seq_len] token tensor (on GPU)
        self.step = step      # current denoising step index
        self.children = []    # list[MCTSNode]
        self.V = 0            # visit / rollout count
        self.total_R = 0.0    # cumulative reward backpropagated through this node
        self.parent = parent


class MCTSSampler(Sampler):
    """
    MCTS sampler for GenMol (MDLM), adapted from §3.4 of Complexa
    (ICLR 2026) to discrete confidence-based unmasking.

    Tree structure:
      State (node)   = partially unmasked SAFE sequence at denoising step t
      Action (edge)  = run K steps of step_confidence with stochastic noise
      Terminal       = fully decoded sequence → SMILES → reward

    Each MCTS iteration:
      1. SELECT:    descend tree via UCB until an unexpanded node.
      2. EXPAND:    run K denoising steps × L branches → L child nodes.
      3. SIMULATE:  rollout all L children to completion in one GPU batch.
      4. BACKPROP:  propagate each child's reward up to root.

    UCB formula:
        UCB(node) = mean_R + c_uct * sqrt(ln(parent.V) / V)

    Key advantage over beam search:
      MCTS retains pruned subtrees -- a low-reward branch stays in the tree
      and can be revisited when better branches exhaust their exploration
      bonus. In beam search, pruned candidates are permanently discarded.
      Example: a branch scoring 0.54 on first visit would be killed by beam
      search, but MCTS may revisit it later and find a 0.88 child.

    Key disadvantage:
      Iterations are inherently sequential -- each simulation depends on
      backpropagated visit counts from the previous iteration, so we cannot
      batch across iterations like beam search can. Same forward-pass count,
      but beam walltime << MCTS walltime. This is not a fixable implementation
      issue; it's fundamental to how UCB works.

    Design notes:
      - No elite buffer or diversity penalty here (unlike BeamSearchSampler).
        MCTS's UCB exploration bonus naturally encourages visiting different
        subtrees, so explicit diversity pressure is less necessary. The tree
        also accumulates all rollout results and picks the best unique ones
        at the end, rather than maintaining a fixed-width beam.
      - Budget is unified with beam search as forward passes per output
        molecule (fp_per_sample) for fair Pareto comparisons. The default
        formula (L * T / K) matches beam search's total candidate count.

    Budget per output molecule:
        B_mol = num_rollouts / num_samples  (default = L * T / K)
    """

    def __init__(
        self,
        path,
        branching_factor: int = 4,
        steps_per_interval: int = 5,
        c_uct: float = 1.0,
        rollout_budget_per_sample: int = None,
        forward_op=None,
        **kwargs,
    ):
        super().__init__(path, **kwargs)
        self.branching_factor = branching_factor
        self.steps_per_interval = steps_per_interval
        self.c_uct = c_uct
        self.rollout_budget_per_sample = rollout_budget_per_sample  # explicit budget cap
        self.forward_op = forward_op or QEDForwardOp()

    def _decode_smiles(self, x, fix=True):
        return decode_smiles(self.model, x, fix=fix)

    @torch.no_grad()
    def _rollout(self, x, start_step, total_steps, softmax_temp, randomness):
        """Complete denoising from start_step → total_steps.
        Returns (denoised tensor, forward_pass_count)."""
        fp = 0
        for i in range(start_step, total_steps):
            attention_mask = x != self.pad_index
            logits = self.model(x, attention_mask)
            fp += x.shape[0]
            x = self.mdlm.step_confidence(
                logits, x, i, total_steps, softmax_temp, randomness
            )
        return x, fp

    # ------------------------------------------------------------------
    # MCTS phases
    # ------------------------------------------------------------------

    def _ucb(self, node: MCTSNode) -> float:
        """Upper Confidence Bound: Q(node) + c * sqrt(ln(parent.V) / V).
        Balances exploitation (high mean reward Q) vs exploration (low visit
        count V relative to parent). Unvisited nodes get +inf to guarantee
        they're tried before any revisit."""
        if node.V == 0:
            return float("inf")
        Q = node.total_R / node.V
        return Q + self.c_uct * math.sqrt(math.log(node.parent.V) / node.V)

    def _select(self, root: MCTSNode, total_steps: int) -> MCTSNode:
        """Descend via UCB until reaching an unexpanded or terminal node."""
        node = root
        while node.children and node.step < total_steps:
            node = max(node.children, key=self._ucb)
        return node

    @torch.no_grad()
    def _expand_simulate_backprop(
        self,
        node: MCTSNode,
        total_steps: int,
        softmax_temp: float,
        randomness: float,
    ):
        """
        Expand node → L children (K denoising steps each),
        rollout all L children to completion in one GPU batch,
        backprop each child's reward up to root.
        Returns list of (reward, smiles).
        """
        L = self.branching_factor
        K = min(self.steps_per_interval, total_steps - node.step)
        if K <= 0:
            return [], 0  # shouldn't happen; guard for edge cases

        # --- Expand: run K denoising steps on L copies of the node's state ---
        # Each copy takes a different stochastic path thanks to MDLM sampling
        # noise (softmax_temp + Gumbel position noise scaled by randomness).
        fp = 0
        xi = node.x.repeat(L, 1)  # [L, seq_len]
        for j in range(K):
            attn = xi != self.pad_index
            logits = self.model(xi, attn)
            fp += xi.shape[0]
            xi = self.mdlm.step_confidence(
                logits, xi, node.step + j, total_steps, softmax_temp, randomness
            )
        new_step = node.step + K

        # Attach children to the tree. After this, node.children is non-empty,
        # so _select will descend through children on future visits rather than
        # returning this node again (each node is expanded at most once).
        for i in range(L):
            child = MCTSNode(xi[i : i + 1].clone(), new_step, parent=node)
            node.children.append(child)

        # --- Simulate: rollout all L children to completion in one GPU batch ---
        if new_step < total_steps:
            rolled, rollout_fp = self._rollout(xi, new_step, total_steps, softmax_temp, randomness)
            fp += rollout_fp
        else:
            rolled = xi  # already fully denoised, no rollout needed

        smiles_list = self._decode_smiles(rolled)
        scores = self.forward_op(smiles_list).tolist()

        # --- Backprop: propagate each child's reward up to root ---
        # This updates V (visit count) and total_R (cumulative reward) along
        # the path, which shifts future UCB scores for the whole subtree.
        results = []
        for child, smi, r in zip(node.children, smiles_list, scores):
            child.V = 1
            child.total_R = r
            anc = node
            while anc is not None:
                anc.V += 1
                anc.total_R += r
                anc = anc.parent
            if smi:
                results.append((r, smi))

        return results, fp

    def _handle_terminal(self, node: MCTSNode):
        """
        Handle a terminal node (step >= total_steps) during selection.
        Since decode is deterministic, re-rolling produces the same result.
        Instead, just backprop the cached reward to boost this path's visit count.
        Returns list of (reward, smiles) — empty if node has no valid molecule.
        """
        smi = self._decode_smiles(node.x)
        r = self.forward_op(smi).tolist()[0]
        # Backprop
        anc = node
        while anc is not None:
            anc.V += 1
            anc.total_R += r
            anc = anc.parent
        return [(r, smi[0])] if smi[0] else []

    # ------------------------------------------------------------------
    # Generation entry point
    # ------------------------------------------------------------------

    @torch.no_grad()
    def de_novo_generation(
        self,
        num_samples: int = 1,
        softmax_temp: float = 0.8,
        randomness: float = 0.5,
        min_add_len: int = 40,
        **kwargs,
    ):
        L = self.branching_factor
        K = self.steps_per_interval

        # ── Root: single masked prototype ─────────────────────────────────
        x_proto = torch.hstack([
            torch.full((1, 1), self.model.bos_index),
            torch.full((1, 1), self.model.eos_index),
        ])
        x_proto = self._insert_mask(x_proto, num_samples=1, min_add_len=min_add_len)
        x_proto = x_proto.to(self.model.device)
        total_steps = max(self.mdlm.get_num_steps_confidence(x_proto), 2)

        # Budget: how many MCTS iterations to run.
        # Each iteration expands one node into L children, so total rollouts ≈ iters * L.
        # Default formula (L * T / K) matches beam search's total candidate count
        # for a fair comparison at the same compute budget.
        if self.rollout_budget_per_sample is not None:
            num_rollouts = self.rollout_budget_per_sample * num_samples
        else:
            num_rollouts = num_samples * round(L * total_steps / K)
        iters = max(1, num_rollouts // L)

        root = MCTSNode(x_proto, step=0)
        # UCB requires parent.V > 0; seed root so first selection doesn't hit log(0).
        root.V = 1

        # ── MCTS loop ─────────────────────────────────────────────────────
        all_results = []  # list of (reward, smiles)
        total_fp = 0

        for _ in range(iters):
            node = self._select(root, total_steps)

            if node.step >= total_steps:
                # Terminal leaf — decode is deterministic, so just backprop
                # the same reward to update visit counts (no new forward passes).
                all_results.extend(self._handle_terminal(node))
            else:
                # Unexpanded node — expand, simulate, backprop
                results, fp = self._expand_simulate_backprop(
                    node, total_steps, softmax_temp, randomness
                )
                all_results.extend(results)
                total_fp += fp

        # ── Collect top-N unique SMILES ───────────────────────────────────
        # Unlike beam search which keeps a fixed-width beam, MCTS accumulates
        # all rollout results and picks the best unique ones at the end.
        all_results.sort(reverse=True, key=lambda x: x[0])
        seen = set()
        final_smiles = []
        for _, smi in all_results:
            if smi not in seen:
                seen.add(smi)
                final_smiles.append(smi)
                if len(final_smiles) >= num_samples:
                    break

        # If the tree didn't produce enough unique molecules (can happen with
        # low budget or high beam collapse), fall back to unconditional sampling.
        if len(final_smiles) < num_samples:
            extra = super().de_novo_generation(
                num_samples - len(final_smiles), softmax_temp, randomness, min_add_len
            )
            final_smiles += [s for s in extra if s]

        self.last_reward_evals = num_rollouts
        self.last_budget_per_sample = num_rollouts / max(num_samples, 1)
        self.last_forward_passes = total_fp
        self.last_fp_per_sample = total_fp / max(num_samples, 1)
        return final_smiles[:num_samples]
