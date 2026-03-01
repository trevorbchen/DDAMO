"""Evaluate molecular metrics on a CSV of SMILES strings.

Usage:
    conda run -n SGPO python evals/run_evals.py --input smiles.csv --output smiles_labeled.csv
    conda run -n SGPO python evals/run_evals.py --input smiles.csv --output out.csv --boltz
"""

import argparse

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import is_valid, qed, sa_score, hbond_satisfaction
from boltz_affinity import run_boltz_affinity


def main():
    parser = argparse.ArgumentParser(description="Compute molecular eval metrics from SMILES.")
    parser.add_argument("--input", required=True, help="Input CSV with a 'sequence' column.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--boltz", action="store_true", help="Run Boltz affinity prediction (requires GPU, slow).")
    parser.add_argument("--boltz-input-dir", default="boltz_inputs", help="Boltz YAML input directory.")
    parser.add_argument("--boltz-out-dir", default="boltz_outputs", help="Boltz output directory.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    assert "sequence" in df.columns, "Input CSV must have a 'sequence' column."

    smiles = df["sequence"]
    df["valid_rdkit"] = smiles.apply(is_valid)
    df["qed"] = smiles.apply(qed)
    df["sa_score"] = smiles.apply(sa_score)
    df["hbond_satisfaction"] = smiles.apply(hbond_satisfaction)

    if args.boltz:
        print(f"\nRunning Boltz affinity on {len(df)} molecules...")
        affinities = run_boltz_affinity(
            smiles.tolist(),
            input_dir=args.boltz_input_dir,
            out_dir=args.boltz_out_dir,
        )
        df["boltz_affinity"] = affinities

    df.to_csv(args.output, index=False)

    # Summary
    n = len(df)
    n_valid = df["valid_rdkit"].sum()
    print(f"\nProcessed {n} molecules — {n_valid}/{n} valid ({100*n_valid/n:.1f}%)")
    summary_cols = ["valid_rdkit", "qed", "sa_score", "hbond_satisfaction"]
    if args.boltz:
        summary_cols.append("boltz_affinity")
    print(df[summary_cols].describe().to_string())


if __name__ == "__main__":
    main()
