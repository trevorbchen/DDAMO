"""Molecular evaluation metrics computed from SMILES strings."""

import os
import importlib.util
from typing import Optional

import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import QED, Descriptors

RDLogger.logger().setLevel(RDLogger.ERROR)

# SA Score lives in rdkit's Contrib directory
_sa_path = os.path.join(os.path.dirname(rdkit.__file__), "Contrib", "SA_Score", "sascorer.py")
_spec = importlib.util.spec_from_file_location("sascorer", _sa_path)
sascorer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sascorer)


def is_valid(smiles: str) -> bool:
    """Check if RDKit can parse the SMILES into a valid molecule."""
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def qed(smiles: str) -> Optional[float]:
    """Quantitative Estimate of Drug-likeness."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return QED.qed(mol)


def sa_score(smiles: str) -> Optional[float]:
    """Synthetic Accessibility score (1 = easy, 10 = hard)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return sascorer.calculateScore(mol)


def hbond_satisfaction(smiles: str) -> Optional[float]:
    """H-bond satisfaction: ratio of donors to (donors + acceptors).

    Returns a value in [0, 1]. Returns None for invalid SMILES
    or molecules with zero donors and acceptors.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    donors = Descriptors.NumHDonors(mol)
    acceptors = Descriptors.NumHAcceptors(mol)
    total = donors + acceptors
    if total == 0:
        return None
    return donors / total
