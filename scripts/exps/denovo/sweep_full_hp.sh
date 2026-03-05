#!/bin/bash
# Sweep beam search and MCTS at matched rollout budgets.
#
# Budget B = rollouts per output molecule (both methods, same denominator).
# Beam:  N∈{5,10,20,50} × L∈{2,4,8} × temp∈{0.5,0.8}, K derived from B and L
#        → ceil(num_samples/N) independent runs, top-50 collected
# MCTS:  L∈{2,4,8} × c_uct∈{0.5,1.0,2.0}, explicit rollout_budget_per_sample=B
# Budget levels: B ∈ {20, 50, 100, 200}
# T_APPROX=50 used to derive K = max(1, round(L*T/B))
#
# Usage:
#   tmux new -s sweep
#   cd scripts/exps/denovo
#   bash run_sweep.sh
#   python3 compare_sweep.py

cd "$(dirname "$0")"

OUT="output_dir=outputs/sweep"
N_SAMPLES="num_samples=50"
REWARD="reward=none"
BUF="sampler.elite_buffer_size=50"
T=50  # approx total_steps for K derivation

derive_K() {
    local B=$1 L=$2
    local K=$(python3 -c "k=max(1,round($L*$T/$B)); print(k)")
    echo $K
}

declare -a JOBS

add_beam() {
    local B=$1 N=$2 L=$3 TEMP=$4
    local K=$(derive_K $B $L)
    local NAME="beam_B${B}_N${N}_L${L}_t${TEMP/./}"
    JOBS+=("python3 run_beam_mcts.py sampler=beam_search $REWARD $N_SAMPLES $OUT $BUF \
        softmax_temp=$TEMP \
        sampler.steps_per_interval=$K \
        sampler.branching_factor=$L \
        sampler.beam_width=$N \
        name=$NAME")
}

add_mcts() {
    local B=$1 L=$2 C=$3
    local NAME="mcts_B${B}_L${L}_c${C/./}"
    JOBS+=("python3 run_beam_mcts.py sampler=mcts $REWARD $N_SAMPLES $OUT \
        sampler.branching_factor=$L \
        sampler.c_uct=$C \
        sampler.rollout_budget_per_sample=$B \
        name=$NAME")
}

# Build job list, slow jobs first (small B, large L, small N for beam)
for B in 20 50 100 200; do
    for L in 8 4 2; do
        # Beam: sweep N at each (B, L)
        for N in 5 10 20 50; do
            for TEMP in 0.5 0.8; do
                add_beam $B $N $L $TEMP
            done
        done
        # MCTS: sweep c_uct at each (B, L)
        for C in 0.5 1.0 2.0; do
            add_mcts $B $L $C
        done
    done
done

# Round-robin assign to 4 GPUs
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

echo "=== Sweep: ${#JOBS[@]} configs across 4 GPUs ==="
echo "    Beam: B∈{20,50,100,200} × N∈{5,10,20,50} × L∈{2,4,8} × temp∈{0.5,0.8}"
echo "    MCTS: B∈{20,50,100,200} × L∈{2,4,8} × c_uct∈{0.5,1.0,2.0}"
echo ""

# GPU 0 gets baseline first, then its share
(
    export CUDA_VISIBLE_DEVICES=0
    if [ ! -f "outputs/sweep/none/uncond_baseline/metrics.json" ]; then
        echo "[GPU0] baseline: uncond"
        python3 run_beam_mcts.py sampler=uncond $REWARD $N_SAMPLES $OUT name=uncond_baseline 2>/dev/null
    else
        echo "[GPU0] SKIP baseline: uncond (done)"
    fi
    run_queue 0 "${GPU0[@]}"
) &

(run_queue 1 "${GPU1[@]}") &
(run_queue 2 "${GPU2[@]}") &
(run_queue 3 "${GPU3[@]}") &

wait
echo ""
echo "=== All done. Run: python3 compare_sweep.py ==="
