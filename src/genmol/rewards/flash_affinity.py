"""FlashAffinity ForwardOp for GenMol TTT LoRA.

Wraps FlashAffinity as a reward function: SMILES → binding probability.
Uses RDKit 3D conformers instead of FABind+ docking for speed.

Protein side (PDB + ESM3 repr) is pre-computed once at init.
Ligand side (3D conformer + torchdrug features) is computed per-call.

Usage:
    reward = FlashAffinityForwardOp(
        protein_pdb="FlashAffinity/data/protein_test/pdb/2VT4.pdb",
        protein_repr_path="FlashAffinity/data/protein_test/repr/esm3.lmdb",
        protein_id="2VT4",
        checkpoint_paths=["FlashAffinity/checkpoints/binary_1.ckpt"],
    )
    scores = reward(["CCO", "c1ccccc1"])  # → tensor([0.32, 0.67])
"""

import sys
import os
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

warnings.filterwarnings("ignore")

# Add FlashAffinity to path
FLASH_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), "FlashAffinity")
if FLASH_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(FLASH_ROOT, "src"))


# --- Ligand featurization (from FlashAffinity torchdrug.py, no torchdrug dep) ---

_ATOM_VOCAB = {"H": 0, "B": 1, "C": 2, "N": 3, "O": 4, "F": 5, "Mg": 6, "Si": 7,
               "P": 8, "S": 9, "Cl": 10, "Cu": 11, "Zn": 12, "Se": 13, "Br": 14, "Sn": 15, "I": 16}
_DEGREE_VOCAB = list(range(7))
_NUM_HS_VOCAB = list(range(7))
_FORMAL_CHARGE_VOCAB = list(range(-5, 6))
_TOTAL_VALENCE_VOCAB = list(range(8))

# Ligand atom type mapping (matches FlashAffinity parser)
_LIGAND_ATOM_MAP = {
    "H": 1, "B": 2, "C": 3, "N": 4, "O": 5, "F": 6, "Mg": 7, "Si": 8,
    "P": 9, "S": 10, "Cl": 11, "Cu": 12, "Zn": 13, "Se": 14, "Br": 15,
    "Sn": 16, "I": 17, "As": 18, "Te": 19, "At": 20,
}


def _onehot(x, vocab, allow_unknown=False):
    if isinstance(vocab, dict):
        index = vocab.get(x, -1)
        size = len(vocab) + (1 if allow_unknown else 0)
    else:
        try:
            index = list(vocab).index(x)
        except ValueError:
            index = -1
        size = len(list(vocab)) + (1 if allow_unknown else 0)
    feature = [0] * size
    if index >= 0:
        feature[index] = 1
    elif allow_unknown:
        feature[-1] = 1
    return feature


def _atom_features(atom):
    return (_onehot(atom.GetSymbol(), _ATOM_VOCAB, allow_unknown=True) +
            _onehot(atom.GetDegree(), _DEGREE_VOCAB, allow_unknown=True) +
            _onehot(atom.GetTotalNumHs(), _NUM_HS_VOCAB, allow_unknown=True) +
            _onehot(atom.GetTotalValence(), _TOTAL_VALENCE_VOCAB, allow_unknown=True) +
            _onehot(atom.GetFormalCharge(), _FORMAL_CHARGE_VOCAB, allow_unknown=True) +
            [atom.GetIsAromatic()])


def extract_ligand_repr(mol):
    """Extract torchdrug-compatible features [n_atoms, 56]."""
    return torch.tensor([_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)


def smiles_to_3d_mol(smiles: str):
    """SMILES → RDKit mol with 3D coords. Returns None on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    status = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if status != 0:
        status = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if status != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass
    mol = Chem.RemoveHs(mol)
    return mol


class FlashAffinityForwardOp:
    """FlashAffinity binding probability as a ForwardOp for GenMol."""

    def __init__(
        self,
        protein_pdb: str = None,
        protein_repr_path: str = None,
        protein_id: str = "2VT4",
        checkpoint_paths: List[str] = None,
        task: str = "binary",  # "binary" or "value"
        device: str = "cuda",
        distance_threshold: float = 20.0,
    ):
        self.device = device
        self.task = task
        self.protein_id = protein_id
        self.distance_threshold = distance_threshold

        # Default paths relative to genmol root
        genmol_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        fa_root = os.path.join(genmol_root, "FlashAffinity")

        if protein_pdb is None:
            protein_pdb = os.path.join(fa_root, "data/protein_test/pdb", f"{protein_id}.pdb")
        if protein_repr_path is None:
            protein_repr_path = os.path.join(fa_root, "data/protein_test/repr/esm3.lmdb")
        if checkpoint_paths is None:
            if task == "binary":
                checkpoint_paths = [os.path.join(fa_root, f"checkpoints/binary_{i}.ckpt") for i in [1, 2]]
            else:
                checkpoint_paths = [os.path.join(fa_root, f"checkpoints/value_{i}.ckpt") for i in [1, 2]]

        # Pre-compute protein features
        self._protein_data = self._load_protein(protein_pdb, protein_repr_path, protein_id)

        # Load model(s) for ensemble
        self._models = []
        for ckpt_path in checkpoint_paths:
            model = self._load_model(ckpt_path)
            self._models.append(model)

        print(f"FlashAffinityForwardOp: {len(self._models)} model(s), task={task}, "
              f"protein={protein_id}, device={device}")

    def _load_protein(self, pdb_path: str, repr_path: str, protein_id: str):
        """Parse protein PDB and load ESM3 representations."""
        from affinity.dataset.parser import parse_structure_file
        from affinity.utils.resource_loader import ResourceLoader
        from pathlib import Path

        # Parse protein structure
        with open(pdb_path, "r") as f:
            pdb_content = f.read()

        protein_coords, protein_atom_types, protein_residue_types, protein_residue_indices = \
            parse_structure_file(pdb_content, None, "pdb", "protein")

        # Load ESM3 repr
        loader = ResourceLoader(Path(repr_path))
        protein_repr = np.array(loader.get(protein_id))  # [n_residues, 1536]

        return {
            "coords": protein_coords,
            "atom_types": protein_atom_types,
            "residue_types": protein_residue_types,
            "residue_indices": protein_residue_indices,
            "repr": protein_repr,
        }

    def _load_model(self, ckpt_path: str):
        """Load a FlashAffinity model from checkpoint."""
        from affinity.model.model import AffinityModel

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hp = ckpt["hyper_parameters"]
        model = AffinityModel(**hp)

        # Load EMA weights if available
        if "ema" in ckpt and ckpt["ema"] is not None:
            ema_state = ckpt["ema"]
            if "shadow_params" in ema_state:
                shadow = ema_state["shadow_params"]
                state_dict = {}
                for (name, _), param in zip(model.named_parameters(), shadow):
                    state_dict[name] = param
                model.load_state_dict(state_dict, strict=False)
            else:
                model.load_state_dict(ckpt["state_dict"], strict=False)
        else:
            model.load_state_dict(ckpt["state_dict"], strict=False)

        model.to(self.device)
        model.eval()
        return model

    def _featurize_sample(self, mol, protein_data: dict) -> Optional[dict]:
        """Featurize a single protein-ligand pair."""
        from affinity.dataset.featurizer import build_edges, crop_pocket

        # Ligand coords and atom types from 3D mol
        conf = mol.GetConformer()
        ligand_coords = np.array(conf.GetPositions(), dtype=np.float32)
        ligand_atom_types = np.array(
            [_LIGAND_ATOM_MAP.get(a.GetSymbol(), 21) for a in mol.GetAtoms()],
            dtype=np.int64,
        )

        # Ligand representations
        ligand_repr = extract_ligand_repr(mol)  # [n_lig, 56]

        # Protein data (copy to allow cropping)
        protein_coords = protein_data["coords"].copy()
        protein_atom_types = protein_data["atom_types"].copy()
        protein_residue_types = protein_data["residue_types"].copy()
        protein_residue_indices = protein_data["residue_indices"].copy()
        protein_repr = protein_data["repr"].copy()

        # Crop pocket around ligand
        n_protein = len(protein_coords)
        n_ligand = len(ligand_coords)

        max_atoms = 2048
        max_residues = 512
        max_edges = 16384

        if (n_protein + n_ligand) > max_atoms:
            protein_coords, protein_atom_types, protein_residue_types, protein_residue_indices = \
                crop_pocket(
                    protein_coords, protein_atom_types, protein_residue_types,
                    protein_residue_indices, ligand_coords,
                    max_atoms, max_residues, self.distance_threshold,
                )

        n_protein = len(protein_coords)

        # Extract pocket repr from full protein repr
        unique_res = np.unique(protein_residue_indices)
        pocket_repr = protein_repr[unique_res]  # [n_pocket_res, 1536]

        # Expand to atom level
        res_to_idx = np.full(protein_repr.shape[0], -1, dtype=int)
        res_to_idx[unique_res] = np.arange(len(unique_res))
        atom_pocket_idx = res_to_idx[protein_residue_indices]
        protein_repr_atoms = pocket_repr[atom_pocket_idx]  # [n_protein, 1536]

        # Build edges
        edge_index, edge_types = build_edges(
            protein_coords, ligand_coords, 4.0, 10.0,
            mol, protein_residue_indices, protein_atom_types, protein_residue_types,
            max_edges,
        )

        # Global nodes
        p_coords = protein_coords.mean(axis=0) if n_protein > 0 else np.zeros(3)
        l_coords = ligand_coords.mean(axis=0)
        p_repr = protein_repr_atoms.mean(axis=0) if n_protein > 0 else np.zeros(1536)
        l_repr = ligand_repr.mean(dim=0, dtype=torch.float32).numpy()

        # Assemble features [P_global, protein_atoms, L_global, ligand_atoms]
        all_coords = np.concatenate([p_coords[None], protein_coords, l_coords[None], ligand_coords])
        all_atom_types = np.concatenate([[-1], protein_atom_types, [-1], ligand_atom_types])
        molecule_types = np.concatenate([[1], np.ones(n_protein, dtype=np.int64),
                                         [2], np.full(n_ligand, 2, dtype=np.int64)])
        all_residue_types = np.concatenate([[-1], protein_residue_types, [-1], ligand_atom_types])
        max_res = protein_residue_indices.max() + 1 if len(protein_residue_indices) > 0 else 0
        all_residue_indices = np.concatenate([[-1], protein_residue_indices, [-1],
                                              np.arange(n_ligand) + max_res])

        combined_protein_repr = np.concatenate([p_repr[None], protein_repr_atoms])
        combined_ligand_repr = np.concatenate([l_repr[None], ligand_repr.numpy()])

        atom_mask = np.ones(n_protein + n_ligand + 2, dtype=bool)
        edge_mask = np.ones(edge_index.shape[1], dtype=bool)

        mw = Descriptors.MolWt(mol)

        return {
            "coords": torch.tensor(all_coords, dtype=torch.float32).unsqueeze(0),
            "atom_types": torch.tensor(all_atom_types, dtype=torch.long).unsqueeze(0),
            "molecule_types": torch.tensor(molecule_types, dtype=torch.long).unsqueeze(0),
            "residue_types": torch.tensor(all_residue_types, dtype=torch.long).unsqueeze(0),
            "residue_indices": torch.tensor(all_residue_indices, dtype=torch.long).unsqueeze(0),
            "protein_repr": torch.tensor(combined_protein_repr, dtype=torch.float32).unsqueeze(0),
            "ligand_repr": torch.tensor(combined_ligand_repr, dtype=torch.float32).unsqueeze(0),
            "atom_mask": torch.tensor(atom_mask, dtype=torch.bool).unsqueeze(0),
            "edge_mask": torch.tensor(edge_mask, dtype=torch.bool).unsqueeze(0),
            "edge_index": torch.tensor(edge_index, dtype=torch.long).unsqueeze(0),
            "edge_types": torch.tensor(edge_types, dtype=torch.long).unsqueeze(0),
            "mw": torch.tensor(mw, dtype=torch.float32).unsqueeze(0),
        }

    @torch.no_grad()
    def _predict_single(self, feats: dict) -> float:
        """Run ensemble prediction on a single sample."""
        feats_gpu = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in feats.items()}

        preds = []
        for model in self._models:
            out = model(feats_gpu)
            if self.task == "binary":
                prob = torch.sigmoid(out["affinity_logits_binary"]).item()
                preds.append(prob)
            else:
                from affinity.model.model import apply_correction
                raw_val = out["affinity_pred_value"].item()
                mw = feats["mw"].item()
                corrected = apply_correction(raw_val, mw)
                preds.append(corrected)

        return sum(preds) / len(preds)

    def __call__(self, smiles_list: List[str]) -> torch.Tensor:
        """Compute binding scores for a list of SMILES.

        Returns:
            torch.Tensor of shape [len(smiles_list)] with scores in [0, 1] (binary)
            or affinity values (value task).
        """
        scores = []
        for smi in smiles_list:
            if not smi:
                scores.append(0.0)
                continue

            mol = smiles_to_3d_mol(smi)
            if mol is None or mol.GetNumAtoms() == 0:
                scores.append(0.0)
                continue

            try:
                feats = self._featurize_sample(mol, self._protein_data)
                if feats is None:
                    scores.append(0.0)
                    continue
                score = self._predict_single(feats)
                if self.task == "value":
                    score = -score  # negate so higher = stronger binding
                scores.append(score)
            except Exception as e:
                scores.append(0.0)

        return torch.tensor(scores, dtype=torch.float32)
