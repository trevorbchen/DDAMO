"""Molecular evaluation metrics computed from SMILES strings.

Library usage:
    from evals.metrics import compute_metrics, qed, sa_score

CLI usage:
    python evals/metrics.py --input smiles.csv --output labeled.csv
    python evals/metrics.py --input smiles.csv --output out.csv --boltz
"""

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


def mol_weight(smiles: str) -> Optional[float]:
    """Molecular weight in Daltons."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return float(Descriptors.MolWt(mol))


def compute_metrics(smiles_list: list[str]) -> dict:
    """Compute standard generation metrics from a list of SMILES.

    Returns dict with: validity, uniqueness, qed_mean, qed_top10, qed_max,
    num_samples, mol_weights.
    """
    n = len(smiles_list)
    valid_mols = [Chem.MolFromSmiles(s) for s in smiles_list if s]
    valid_mols = [m for m in valid_mols if m is not None]

    validity = len(valid_mols) / max(n, 1)
    valid_smiles = [Chem.MolToSmiles(m) for m in valid_mols]
    uniqueness = len(set(valid_smiles)) / max(len(valid_mols), 1)

    qeds = sorted([QED.qed(m) for m in valid_mols], reverse=True)
    qed_mean = sum(qeds) / len(qeds) if qeds else 0.0
    top10_n = max(1, len(qeds) // 10)
    qed_top10 = sum(qeds[:top10_n]) / top10_n if qeds else 0.0
    qed_max = qeds[0] if qeds else 0.0

    mw = [mol_weight(s) for s in smiles_list]

    return {
        "validity": validity,
        "uniqueness": uniqueness,
        "qed_mean": qed_mean,
        "qed_top10": qed_top10,
        "qed_max": qed_max,
        "num_samples": n,
        "mol_weights": mw,
    }


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    """Evaluate molecular metrics on a CSV of SMILES strings."""
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Compute molecular eval metrics from SMILES.")
    parser.add_argument("--input", required=True, help="Input CSV with a 'smiles' or 'sequence' column.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--boltz", action="store_true", help="Also run Boltz affinity prediction (GPU, slow).")
    parser.add_argument("--boltz-input-dir", default="boltz_inputs")
    parser.add_argument("--boltz-out-dir", default="boltz_outputs")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    col = "smiles" if "smiles" in df.columns else "sequence"
    assert col in df.columns, f"Input CSV must have a 'smiles' or 'sequence' column."

    smiles = df[col]
    df["valid_rdkit"] = smiles.apply(is_valid)
    df["qed"] = smiles.apply(qed)
    df["sa_score"] = smiles.apply(sa_score)
    df["hbond_satisfaction"] = smiles.apply(hbond_satisfaction)
    df["mol_weight"] = smiles.apply(mol_weight)

    if args.boltz:
        from evals.boltz_affinity import run_boltz_affinity
        print(f"\nRunning Boltz affinity on {len(df)} molecules...")
        affinities = run_boltz_affinity(
            smiles.tolist(),
            input_dir=args.boltz_input_dir,
            out_dir=args.boltz_out_dir,
        )
        df["boltz_affinity"] = affinities

    df.to_csv(args.output, index=False)

    n_valid = df["valid_rdkit"].sum()
    print(f"\nProcessed {len(df)} molecules — {n_valid}/{len(df)} valid ({100*n_valid/len(df):.1f}%)")
    print(df.describe().to_string())


if __name__ == "__main__":
    main()
