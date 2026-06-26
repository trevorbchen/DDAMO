"""
Custom Oracle Example
=====================
Shows how to plug your own binding-affinity oracle (or any scoring function)
into the GenMol active loop without modifying library code.

Two patterns:
  A  per-molecule scoring  (subclass RewardFunction)
  B  batch scoring         (register any callable)

After registering, import this module before calling get_reward():

    import examples.custom_oracle           # registers "my_oracle_A/B"
    oracle = get_reward("my_oracle_A", protein_id="2VT4")

Environment variables:
    FLASHAFFINITY_ROOT   path to your FlashAffinity install (overrides bundled copy)
"""

import torch
from rdkit.Chem import Descriptors
from genmol.rewards import RewardFunction, register_reward


# ==========================================================================
# Pattern A: subclass RewardFunction for per-molecule scoring
# ==========================================================================

class MyMoleculeOracle(RewardFunction):
    """Replace _score_mol with your own model call.

    The base class handles:
    - SMILES -> RDKit Mol parsing (with SAFE fallback)
    - returning -inf for invalid molecules
    - optional score scaling via self.scale
    """

    def __init__(self, protein_id: str = "MY_TARGET", **kwargs):
        self.protein_id = protein_id
        # Load your model weights here:
        #   self.model = MyModel.from_pretrained(...)

    def _score_mol(self, mol) -> float:
        """Score one valid RDKit Mol.  Higher = better binding."""
        # ---- replace with your actual scoring logic ----
        return -Descriptors.MolWt(mol) / 500.0   # placeholder
        # ------------------------------------------------


register_reward("my_oracle_A", MyMoleculeOracle)


# ==========================================================================
# Pattern B: register any callable for batched scoring
# ==========================================================================

def my_batch_oracle(smiles_list, protein_id="MY_TARGET", **kwargs):
    """Score a batch of SMILES strings.

    Args:
        smiles_list: list[str], may include invalid SMILES.

    Returns:
        torch.Tensor of shape [N].  Use 0.0 or float('-inf') for failures.
    """
    # ---- replace with your batched inference ----
    scores = [0.5] * len(smiles_list)   # placeholder
    return torch.tensor(scores, dtype=torch.float32)
    # ---------------------------------------------


register_reward("my_oracle_B", my_batch_oracle)


if __name__ == "__main__":
    smiles = ["CCO", "c1ccccc1", "INVALID", "CC(=O)Oc1ccccc1C(=O)O"]
    oracle_a = MyMoleculeOracle(protein_id="TEST")
    print("Pattern A:", oracle_a(smiles))
    print("Pattern B:", my_batch_oracle(smiles))
