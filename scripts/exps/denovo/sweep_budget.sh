#!/bin/bash
# Sweep beam search quality vs compute budget across different HP axes.
#
# Each curve fixes one HP variant and sweeps K (steps_per_interval) with L=4.
# N=beam_width=num_samples=50, reward=none (QED).
#
# Usage:
#   bash sweep_budget.sh                    # run ALL curves
#   bash sweep_budget.sh default temp       # run only default + temp curves
#   bash sweep_budget.sh diversity           # run only diversity curve
#
# Available curves:
#   default    - softmax_temp=0.8 (baseline HP)
#   temp       - softmax_temp=0.5
#   diversity  - diversity_cutoff=0.6
#   randomness - randomness=2.0
#
# Analysis:  python3 analyze_budget_curves.py

set -e
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:/usr/bin:$PATH"

N=50
REWARD="reward=none"
SAMPLES="num_samples=$N sampler.beam_width=$N sampler.branching_factor=4"
K_VALUES=(1 2 5 10 20)

# ── Curve launchers ────────────────────────────────────────────────────────

run_baseline() {
    echo "--- baseline (unconditional) ---"
    CUDA_VISIBLE_DEVICES=0 python3 run_beam_mcts.py sampler=uncond $REWARD num_samples=$N \
        name=budget_standard 2>/dev/null &
}

run_default() {
    echo "--- default (softmax_temp=0.8) ---"
    local gpu=0
    for K in "${K_VALUES[@]}"; do
        CUDA_VISIBLE_DEVICES=$gpu python3 run_beam_mcts.py sampler=beam_search $REWARD $SAMPLES \
            softmax_temp=0.8 sampler.steps_per_interval=$K \
            name=budget_K${K}_L4_default 2>/dev/null &
        gpu=$(( (gpu + 1) % 2 ))
    done
    # Extra: L=2 and L=8 at K=5
    CUDA_VISIBLE_DEVICES=0 python3 run_beam_mcts.py sampler=beam_search $REWARD \
        num_samples=$N sampler.beam_width=$N sampler.branching_factor=2 \
        sampler.steps_per_interval=5 softmax_temp=0.8 \
        name=budget_K5_L2_default 2>/dev/null &
    CUDA_VISIBLE_DEVICES=1 python3 run_beam_mcts.py sampler=beam_search $REWARD \
        num_samples=$N sampler.beam_width=$N sampler.branching_factor=8 \
        sampler.steps_per_interval=5 softmax_temp=0.8 \
        name=budget_K5_L8_default 2>/dev/null &
}

run_temp() {
    echo "--- temp (softmax_temp=0.5) ---"
    local gpu=0
    for K in "${K_VALUES[@]}"; do
        CUDA_VISIBLE_DEVICES=$gpu python3 run_beam_mcts.py sampler=beam_search $REWARD $SAMPLES \
            softmax_temp=0.5 sampler.steps_per_interval=$K \
            name=budget_K${K}_L4_t05 2>/dev/null &
        gpu=$(( (gpu + 1) % 2 ))
    done
    # Extra: L=2 and L=8 at K=5
    CUDA_VISIBLE_DEVICES=0 python3 run_beam_mcts.py sampler=beam_search $REWARD \
        num_samples=$N sampler.beam_width=$N sampler.branching_factor=2 \
        sampler.steps_per_interval=5 softmax_temp=0.5 \
        name=budget_K5_L2_t05 2>/dev/null &
    CUDA_VISIBLE_DEVICES=1 python3 run_beam_mcts.py sampler=beam_search $REWARD \
        num_samples=$N sampler.beam_width=$N sampler.branching_factor=8 \
        sampler.steps_per_interval=5 softmax_temp=0.5 \
        name=budget_K5_L8_t05 2>/dev/null &
}

run_diversity() {
    echo "--- diversity (diversity_cutoff=0.6) ---"
    local gpu=0
    for K in "${K_VALUES[@]}"; do
        CUDA_VISIBLE_DEVICES=$gpu python3 run_beam_mcts.py sampler=beam_search $REWARD $SAMPLES \
            sampler.diversity_cutoff=0.6 sampler.steps_per_interval=$K \
            name=budget_K${K}_L4_div 2>/dev/null &
        gpu=$(( (gpu + 1) % 2 ))
    done
}

run_randomness() {
    echo "--- randomness (randomness=2.0) ---"
    local gpu=0
    for K in "${K_VALUES[@]}"; do
        CUDA_VISIBLE_DEVICES=$gpu python3 run_beam_mcts.py sampler=beam_search $REWARD $SAMPLES \
            randomness=2.0 sampler.steps_per_interval=$K \
            name=budget_K${K}_L4_rand 2>/dev/null &
        gpu=$(( (gpu + 1) % 2 ))
    done
}

# ── Main ───────────────────────────────────────────────────────────────────

CURVES=("$@")
if [ ${#CURVES[@]} -eq 0 ]; then
    CURVES=(default temp diversity randomness)
fi

echo "=== Budget sweep: ${CURVES[*]} (2 GPUs) ==="

run_baseline

for curve in "${CURVES[@]}"; do
    case "$curve" in
        default)    run_default ;;
        temp)       run_temp ;;
        diversity)  run_diversity ;;
        randomness) run_randomness ;;
        *) echo "Unknown curve: $curve (options: default, temp, diversity, randomness)" ;;
    esac
done

wait
echo "=== All done. Run: python3 analyze_budget_curves.py ==="
