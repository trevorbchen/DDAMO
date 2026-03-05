#!/bin/bash
# Package all formal experiment results into a clean directory structure.
#
# Output: outputs/results/  (tar.gz archive + directory)
#
# Experiments included:
#   1. hp_sweep/        Full HP sweep (sweep_full_hp.sh + sweep_noelite_divpenalty.sh)
#                       beam: B∈{20,50,100,200} × N∈{5,10,20,50} × L∈{2,4,8} × temp∈{0.5,0.8}
#                       beam_noelite: no elite buffer ablation
#                       beam_diverse: diversity penalty λ∈{0.1,0.3,0.5}
#                       mcts: B∈{20,50,100,200} × L∈{2,4,8} × c_uct∈{0.5,1.0,2.0}
#                       mcts_ref: reference MCTS for task 2/3
#                       uncond_baseline: unconditional sampling baseline
#
#   2. budget_curves/   Budget vs quality curves (sweep_budget.sh)
#                       K∈{1,2,5,10,20} × {default, temp=0.5, diversity, randomness}
#                       + L∈{2,8} variants at K=5
#                       + unconditional baseline
#
#   3. boltz_eval/      Boltz structure prediction evaluation (run_boltz_eval.py)
#
# Each run directory contains:
#   - samples.csv    (smiles, mol_wt columns)
#   - metrics.json   (validity, uniqueness, qed stats — if available)
#   - config.yaml    (hydra config snapshot — if available)
#
# Usage:
#   cd scripts/exps/denovo
#   bash package_results.sh

set -e
cd "$(dirname "$0")"

DEST="outputs/results"
rm -rf "$DEST"
mkdir -p "$DEST/hp_sweep" "$DEST/budget_curves" "$DEST/boltz_eval"

# ── 1. HP sweep ──────────────────────────────────────────────────────────────
echo "Copying HP sweep results..."
SRC="outputs/sweep/none"
count=0
for d in "$SRC"/*/; do
    name=$(basename "$d")
    cp -r "$d" "$DEST/hp_sweep/$name"
    count=$((count + 1))
done
echo "  $count configurations copied"

# ── 2. Budget curves ─────────────────────────────────────────────────────────
echo "Copying budget curve results..."
SRC="outputs/none"
count=0
for d in "$SRC"/budget_*/; do
    name=$(basename "$d")
    cp -r "$d" "$DEST/budget_curves/$name"
    count=$((count + 1))
done
echo "  $count configurations copied"

# ── 3. Boltz eval ────────────────────────────────────────────────────────────
echo "Copying Boltz eval results..."
cp outputs/boltz_eval/*.csv "$DEST/boltz_eval/"
echo "  done"

# ── Summary ──────────────────────────────────────────────────────────────────
TOTAL_CSV=$(find "$DEST" -name "samples.csv" | wc -l)
TOTAL_SMILES=$(find "$DEST" -name "samples.csv" -exec tail -n +2 {} \; | wc -l)

cat > "$DEST/README.txt" <<EOF
De novo sampling experiment results — $(date +%Y-%m-%d)
======================================================

Total: $TOTAL_CSV runs, $TOTAL_SMILES molecules (SMILES)

1. hp_sweep/  ($( ls "$DEST/hp_sweep" | wc -l ) configs)
   Full hyperparameter sweep at matched rollout budgets.
   Source scripts: sweep_full_hp.sh, sweep_noelite_divpenalty.sh

   Naming convention:
     beam_B{budget}_N{beam_width}_L{branching}_t{temp}
     beam_noelite_B{budget}_N{width}_L{branching}_t{temp}
     beam_div{lambda}_B{budget}_N{width}_L{branching}_t{temp}
     mcts_B{budget}_L{branching}_c{c_uct}
     mcts_ref_B{budget}_L{branching}_c{c_uct}
     uncond_baseline

2. budget_curves/  ($( ls "$DEST/budget_curves" | wc -l ) configs)
   Quality vs compute budget at fixed L=4, N=50.
   Source script: sweep_budget.sh

   Naming convention:
     budget_K{steps}_L{branching}_{variant}
     variant ∈ {default (temp=0.8), t05 (temp=0.5), div (diversity), rand (randomness)}
     budget_standard = unconditional baseline

3. boltz_eval/
   Boltz structure prediction on selected molecules.
   Source script: run_boltz_eval.py
EOF

echo ""
echo "=== Packaged: $TOTAL_CSV runs, $TOTAL_SMILES molecules ==="
echo "=== Location: $DEST/ ==="
echo "=== README:   $DEST/README.txt ==="

# Create tar.gz
tar -czf outputs/results.tar.gz -C outputs results
echo "=== Archive:  outputs/results.tar.gz ==="
