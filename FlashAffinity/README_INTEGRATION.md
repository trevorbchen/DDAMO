# FlashAffinity Integration

This directory contains the FlashAffinity oracle used as the default binding affinity scorer in GenMol.

## What it is

FlashAffinity (Jiang et al., 2025) is a lightweight structure-based affinity predictor that achieves ~50x speedup over Boltz-2 by replacing expensive protein structure prediction with a fast docking model and swapping the PairFormer module for an EGNN.

In GenMol, it is used as the **binary binding classifier** — returning the probability that a small molecule binds to the target protein. Scores are in [0, 1]; higher = more likely to bind.

## Directory layout

```
FlashAffinity/
├── checkpoints/          # Model weights (.ckpt) — NOT pushed to git
├── data/
│   └── protein_test/
│       ├── pdb/          # One .pdb per target (2VT4.pdb, 5SDV.pdb, ...)
│       └── repr/         # Pre-computed ESM3 protein representations (lmdb)
├── src/affinity/         # FlashAffinity model + featurizer source code
└── scripts/              # Original FlashAffinity training/eval scripts
```

## Using your own affinity oracle

Set `FLASHAFFINITY_ROOT` to point at a different install:

```bash
export FLASHAFFINITY_ROOT=/path/to/your/FlashAffinity
python scripts/run_active_loop.py --model_path model_v2.ckpt --oracle flash_affinity ...
```

Or register a completely different oracle — see `examples/custom_oracle.py`.

## Adding a new protein target

1. Place `YOURPDB.pdb` in `data/protein_test/pdb/`.
2. Pre-compute ESM3 representations and append to the lmdb at `data/protein_test/repr/esm3.lmdb`.
3. Run with `--protein_id YOURPDB`.
