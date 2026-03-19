"""Boltz affinity evaluation: score SMILES against a fixed receptor.

Library usage:
    from evals.boltz_affinity import run_boltz_affinity
    affinities = run_boltz_affinity(["CCO", "c1ccccc1"])

CLI usage:
    python evals/boltz_affinity.py --input smiles.csv --output scored.csv
"""

from genmol.rewards.boltz import run_boltz_affinity, RECEPTOR_SEQUENCE  # noqa: F401


if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Score SMILES with Boltz affinity.")
    parser.add_argument("--input", required=True, help="Input CSV with 'smiles' column.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--diffusion-samples", type=int, default=16)
    parser.add_argument("--devices", type=int, default=1)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    col = "smiles" if "smiles" in df.columns else "sequence"
    affinities = run_boltz_affinity(
        df[col].tolist(),
        diffusion_samples=args.diffusion_samples,
        devices=args.devices,
    )
    df["boltz_affinity"] = affinities
    df.to_csv(args.output, index=False)

    valid = [a for a in affinities if a is not None]
    print(f"Scored {len(valid)}/{len(affinities)} molecules")
    if valid:
        print(f"  mean={sum(valid)/len(valid):.3f}  best={min(valid):.3f}")
