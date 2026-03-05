#!/usr/bin/env python3
"""Compare molecule quality across sampler output CSVs."""

import sys
import random
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED, Descriptors, DataStructs, rdMolDescriptors

CONFIGS = [
    ("outputs/none/uncond/samples.csv",    "Standard"),
    ("outputs/none/beam_buf/samples.csv",  "Beam+buf (baseline)"),
    ("outputs/none/beam_rand/samples.csv", "Fix1: randomness=2.0"),
    ("outputs/none/beam_temp/samples.csv", "Fix2: softmax_temp=0.5"),
    ("outputs/none/beam_early/samples.csv","Fix3: steps_per_interval=1"),
    ("outputs/none/beam_div/samples.csv",  "Fix4: diversity_cutoff=0.6"),
]


def fingerprint(mol):
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def mean_pairwise_diversity(mols, max_pairs=2000):
    """Average 1 - Tanimoto over random pairs (capped for speed)."""
    fps = [fingerprint(m) for m in mols]
    if len(fps) < 2:
        return 0.0
    pairs = [(i, j) for i in range(len(fps)) for j in range(i + 1, len(fps))]
    if len(pairs) > max_pairs:
        pairs = random.sample(pairs, max_pairs)
    sims = [DataStructs.TanimotoSimilarity(fps[i], fps[j]) for i, j in pairs]
    return 1.0 - sum(sims) / len(sims)


def analyze(csv_path, label):
    df = pd.read_csv(csv_path)
    smiles = df["smiles"].dropna().tolist()
    mols = [Chem.MolFromSmiles(s) for s in smiles]

    valid_pairs = [(s, m) for s, m in zip(smiles, mols) if m is not None]
    valid_smiles = [s for s, _ in valid_pairs]
    valid_mols   = [m for _, m in valid_pairs]

    qeds    = [QED.qed(m) for m in valid_mols]
    uniq_pct = len(set(valid_smiles)) / max(len(valid_smiles), 1) * 100
    div     = mean_pairwise_diversity(valid_mols)

    top10_n  = max(1, len(qeds) // 10)
    top10_qed = sum(sorted(qeds, reverse=True)[:top10_n]) / top10_n

    print(f"\n{'='*52}")
    print(f"  {label}")
    print(f"  {csv_path}")
    print(f"{'='*52}")
    print(f"  Samples:        {len(smiles)}")
    print(f"  Valid:          {len(valid_pairs)} ({len(valid_pairs)/max(len(smiles),1)*100:.1f}%)")
    print(f"  Unique:         {uniq_pct:.1f}%")
    print(f"  QED  mean:      {sum(qeds)/len(qeds):.4f}" if qeds else "  QED: n/a")
    print(f"  QED  top-10%:   {top10_qed:.4f}")
    print(f"  QED  max:       {max(qeds):.4f}" if qeds else "")
    print(f"  Diversity:      {div:.4f}   (1 - avg Tanimoto)")
    return {"label": label, "qed_mean": sum(qeds)/len(qeds) if qeds else 0,
            "qed_top10": top10_qed, "diversity": div, "uniqueness": uniq_pct}


if __name__ == "__main__":
    # If paths passed as args, generate labels from filename; else use defaults
    if len(sys.argv) > 1:
        paths  = sys.argv[1:]
        labels = [p.split("/")[-2] for p in paths]  # use parent dir as label
    else:
        paths  = [c[0] for c in CONFIGS]
        labels = [c[1] for c in CONFIGS]

    results = []
    for path, label in zip(paths, labels):
        try:
            results.append(analyze(path, label))
        except FileNotFoundError:
            print(f"\n[SKIP] {path} not found")

    if len(results) > 1:
        base = results[0]
        print(f"\n{'='*52}")
        print("  Delta vs Standard")
        print(f"{'='*52}")
        for r in results[1:]:
            dq = r["qed_mean"] - base["qed_mean"]
            dt = r["qed_top10"] - base["qed_top10"]
            dd = r["diversity"] - base["diversity"]
            print(f"  {r['label']}")
            print(f"    QED mean:    {dq:+.4f}")
            print(f"    QED top-10%: {dt:+.4f}")
            print(f"    Diversity:   {dd:+.4f}")
