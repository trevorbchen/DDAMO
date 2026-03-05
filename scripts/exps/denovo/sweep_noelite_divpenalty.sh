#!/bin/bash
# Task 2: Beam search WITHOUT elite buffer → controlled 50 unique molecules
# Task 3: Diverse beam search with Tanimoto penalty λ∈{0.1, 0.3, 0.5}
#
# Uses best-performing beam HPs from the sweep at B=20 and B=100.
# Also includes MCTS baselines at same budgets for comparison.
#
# Usage:
#   cd scripts/exps/denovo
#   bash run_task23.sh

cd "$(dirname "$0")"

OUT="output_dir=outputs/sweep"
N_SAMPLES="num_samples=50"
REWARD="reward=none"
T=50

derive_K() {
    local B=$1 L=$2
    local K=$(python3 -c "k=max(1,round($L*$T/$B)); print(k)")
    echo $K
}

declare -a JOBS

# ── Task 2: Beam WITHOUT elite buffer ──────────────────────────────────────
# At B=20 and B=100, use best HPs but set elite_buffer_size=null (no buffer)
# This should produce ~N unique molecules per run instead of 2-15.

add_beam_noelite() {
    local B=$1 N=$2 L=$3 TEMP=$4
    local K=$(derive_K $B $L)
    local NAME="beam_noelite_B${B}_N${N}_L${L}_t${TEMP/./}"
    JOBS+=("python3 run_beam_mcts.py sampler=beam_search $REWARD $N_SAMPLES $OUT \
        softmax_temp=$TEMP \
        sampler.steps_per_interval=$K \
        sampler.branching_factor=$L \
        sampler.beam_width=$N \
        sampler.elite_buffer_size=null \
        sampler.diversity_cutoff=null \
        sampler.diversity_penalty=0.0 \
        name=$NAME")
}

# Best beam HPs at B=20: N=50, L=4, temp=0.8
# Best beam HPs at B=100: N=50, L=8, temp=0.8
# Also try smaller N for more diversity
for B in 20 100; do
    for N in 5 10 20 50; do
        for L in 4 8; do
            for TEMP in 0.5 0.8; do
                add_beam_noelite $B $N $L $TEMP
            done
        done
    done
done

# ── Task 3: Diverse beam search with Tanimoto penalty ──────────────────────
add_beam_diverse() {
    local B=$1 N=$2 L=$3 TEMP=$4 LAMBDA=$5
    local K=$(derive_K $B $L)
    local LNAME=$(echo $LAMBDA | tr -d '.')
    local NAME="beam_div${LNAME}_B${B}_N${N}_L${L}_t${TEMP/./}"
    JOBS+=("python3 run_beam_mcts.py sampler=beam_search $REWARD $N_SAMPLES $OUT \
        softmax_temp=$TEMP \
        sampler.steps_per_interval=$K \
        sampler.branching_factor=$L \
        sampler.beam_width=$N \
        sampler.elite_buffer_size=50 \
        sampler.diversity_cutoff=null \
        sampler.diversity_penalty=$LAMBDA \
        name=$NAME")
}

# Sweep λ∈{0.1, 0.3, 0.5} at B=20 and B=100
# Use top HPs: N∈{10,50}, L∈{4,8}, temp=0.8
for B in 20 100; do
    for LAMBDA in 0.1 0.3 0.5; do
        for N in 10 50; do
            for L in 4 8; do
                add_beam_diverse $B $N $L 0.8 $LAMBDA
            done
        done
    done
done

# ── MCTS baselines at same budgets (for reference) ────────────────────────
add_mcts() {
    local B=$1 L=$2 C=$3
    local NAME="mcts_ref_B${B}_L${L}_c${C/./}"
    JOBS+=("python3 run_beam_mcts.py sampler=mcts $REWARD $N_SAMPLES $OUT \
        sampler.branching_factor=$L \
        sampler.c_uct=$C \
        sampler.rollout_budget_per_sample=$B \
        name=$NAME")
}

# Best MCTS configs from sweep: L=4, c=1.0 performed well
for B in 20 100; do
    for L in 4 8; do
        add_mcts $B $L 1.0
    done
done

echo "=== Task 2+3 Sweep: ${#JOBS[@]} configs ==="
echo "    Task 2 (no elite):  B∈{20,100} × N∈{5,10,20,50} × L∈{4,8} × temp∈{0.5,0.8}"
echo "    Task 3 (diverse):   B∈{20,100} × λ∈{0.1,0.3,0.5} × N∈{10,50} × L∈{4,8}"
echo "    MCTS reference:     B∈{20,100} × L∈{4,8} × c=1.0"
echo ""

# Round-robin 4 GPUs
declare -a GPU0 GPU1 GPU2 GPU3
for i in "${!JOBS[@]}"; do
    case $((i % 4)) in
        0) GPU0+=("${JOBS[$i]}") ;;
        1) GPU1+=("${JOBS[$i]}") ;;
        2) GPU2+=("${JOBS[$i]}") ;;
        3) GPU3+=("${JOBS[$i]}") ;;
    esac
done

run_queue() {
    local gpu=$1; shift
    local total=$#
    local i=0
    export CUDA_VISIBLE_DEVICES=$gpu
    for cmd in "$@"; do
        i=$((i+1))
        local name=$(echo $cmd | grep -o 'name=[^ ]*' | cut -d= -f2)
        if [ -f "outputs/sweep/none/${name}/metrics.json" ]; then
            echo "[GPU$gpu] ($i/$total) SKIP (done): $name"
            continue
        fi
        echo "[GPU$gpu] ($i/$total) $name"
        eval "$cmd" 2>/dev/null
        echo "[GPU$gpu] ($i/$total) done: $name"
    done
    echo "[GPU$gpu] ALL DONE"
}

(run_queue 0 "${GPU0[@]}") &
(run_queue 1 "${GPU1[@]}") &
(run_queue 2 "${GPU2[@]}") &
(run_queue 3 "${GPU3[@]}") &

wait
echo ""
echo "=== All done. ==="
