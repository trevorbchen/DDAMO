#!/bin/bash
# Usage: bash run_collect.sh <sampler_args> <output_csv> <n_total> <batch_size>
# Example: bash run_collect.sh "sampler=beam_search reward=none" /tmp/beam.csv 100 4

SAMPLER_ARGS="$1"
OUTPUT_CSV="$2"
N_TOTAL="${3:-100}"
BATCH="${4:-4}"

ROUNDS=$(( (N_TOTAL + BATCH - 1) / BATCH ))
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="python3"
export PATH="$HOME/.local/bin:/usr/bin:$PATH"

echo "=== Collecting $N_TOTAL samples ($ROUNDS rounds x $BATCH) ==="
echo "=== Sampler args: $SAMPLER_ARGS ==="
echo "=== Output: $OUTPUT_CSV ==="

# Write header once
echo "smiles,mol_wt" > "$OUTPUT_CSV"

for i in $(seq 1 $ROUNDS); do
    echo -n "Round $i/$ROUNDS ... "
    cd "$SCRIPT_DIR"
    $PYTHON run_beam_mcts.py $SAMPLER_ARGS num_samples=$BATCH 2>/dev/null | grep -E "^(Time|Valid|Unique)" || true
    # Find the latest samples.csv for this sampler
    LATEST=$(ls -t outputs/*/*/samples.csv 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        tail -n +2 "$LATEST" >> "$OUTPUT_CSV"
    fi
done

TOTAL_LINES=$(( $(wc -l < "$OUTPUT_CSV") - 1 ))
echo ""
echo "=== Done: $TOTAL_LINES molecules saved to $OUTPUT_CSV ==="
