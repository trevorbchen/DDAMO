#!/bin/bash
# Quick test: beam search + MCTS sampling → evals
set -e
cd "$(dirname "$0")"

OUT=outputs/test_evals
rm -rf "$OUT"
mkdir -p "$OUT"

echo "=== Beam Search (5 samples) ==="
python3 run_beam_mcts.py sampler=beam_search num_samples=5 output_dir="$OUT" \
    hydra.run.dir="$OUT/beam_search"

echo ""
echo "=== MCTS (5 samples) ==="
python3 run_beam_mcts.py sampler=mcts num_samples=5 output_dir="$OUT" \
    hydra.run.dir="$OUT/mcts"

echo ""
echo "=== Running evals on beam search output ==="
# Evals expect 'sequence' column, run_beam_mcts.py outputs 'smiles' — rename
BS_CSV="$OUT/qed/beam_search/samples.csv"
python3 -c "import pandas as pd; df=pd.read_csv('$BS_CSV'); df.rename(columns={'smiles':'sequence'}).to_csv('$BS_CSV', index=False)"
python3 ../../../evals/run_evals.py --input "$BS_CSV" --output "$OUT/qed/beam_search/samples_eval.csv"

echo ""
echo "=== Running evals on MCTS output ==="
MCTS_CSV="$OUT/qed/mcts/samples.csv"
python3 -c "import pandas as pd; df=pd.read_csv('$MCTS_CSV'); df.rename(columns={'smiles':'sequence'}).to_csv('$MCTS_CSV', index=False)"
python3 ../../../evals/run_evals.py --input "$MCTS_CSV" --output "$OUT/qed/mcts/samples_eval.csv"

echo ""
echo "=== Done! ==="
echo "Beam search eval: $OUT/qed/beam_search/samples_eval.csv"
echo "MCTS eval:        $OUT/qed/mcts/samples_eval.csv"
