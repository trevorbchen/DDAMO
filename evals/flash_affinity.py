"""FlashAffinity evaluation: score SMILES against a protein target.

Library usage:
    from evals.flash_affinity import run_flash_affinity
    scores = run_flash_affinity(["CCO", "c1ccccc1"], protein_id="2VT4")

CLI usage:
    python evals/flash_affinity.py --input smiles.csv --output scored.csv --protein-id 2VT4
"""

from typing import List, Optional


def run_flash_affinity(
    smiles_list: List[str],
    protein_id: str = "2VT4",
    task: str = "binary",
    **kwargs,
) -> List[Optional[float]]:
    """Score SMILES with FlashAffinity binding prediction.

    Args:
        smiles_list: SMILES strings to evaluate.
        protein_id: Protein target ID (must have PDB + ESM3 repr in FlashAffinity/data/).
        task: "binary" (binding probability) or "value" (affinity value).

    Returns:
        List of scores (float), or None for failures.
    """
    from genmol.rewards.flash_affinity import FlashAffinityForwardOp

    model = FlashAffinityForwardOp(protein_id=protein_id, task=task, **kwargs)
    scores_tensor = model(smiles_list)
    return [s.item() if s != 0.0 else None for s in scores_tensor]


if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Score SMILES with FlashAffinity.")
    parser.add_argument("--input", required=True, help="Input CSV with 'smiles' column.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--protein-id", default="2VT4")
    parser.add_argument("--task", default="value", choices=["binary", "value"])
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    col = "smiles" if "smiles" in df.columns else "sequence"
    scores = run_flash_affinity(df[col].tolist(), protein_id=args.protein_id, task=args.task)
    df["flash_affinity"] = scores
    df.to_csv(args.output, index=False)

    valid = [s for s in scores if s is not None]
    print(f"Scored {len(valid)}/{len(scores)} molecules")
    if valid:
        print(f"  mean={sum(valid)/len(valid):.4f}  max={max(valid):.4f}  min={min(valid):.4f}")
