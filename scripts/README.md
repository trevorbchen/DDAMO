# scripts/

Entry-point scripts for running experiments. Each script is self-contained and accepts all options via CLI flags (`--help` for full list).

## Run scripts

| Script | What it does |
|---|---|
| `run_active_loop.py` | **Main method.** Adaptive active-learning loop: generate candidates → Thompson-sample top-K → score with oracle → CVaR fine-tune model. Supports FA and Boltz oracles, beam/MCTS/Thompson selection strategies, and custom oracles via `--oracle_module`. |
| `run_vidd.py` | **VIDD baseline.** Iterative distillation reward-guided fine-tuning (Wang et al. 2025). No surrogate; directly optimizes oracle reward via KL-regularized policy updates. |
| `run_ddpp_online.py` | **Online DDPP baseline.** Generate a batch → score all with oracle → CVaR fine-tune. No surrogate or Thompson sampling; pure online fine-tuning. |
| `run_dfkc_fa.py` | **DFKC baseline.** Discrete Feynman-Kac Corrector inference-time guidance. No model fine-tuning; steers generation at inference time via sequential Monte Carlo. |
| `run_beam_surrogate.py` | **Surrogate-only beam search ablation.** Beam search guided by an online-retrained surrogate, without any DDPP model fine-tuning. Isolates the value of surrogate-guided search. |

## Subfolders

| Folder | What it does |
|---|---|
| `benchmarks/` | PMO, fragment-constrained, and lead-optimization benchmark runners. Each subfolder has its own `run.py` and `eval.py`. |
| `training/` | Standalone training scripts (`train.py` unconditional pretraining, `train_ddpp.py` DDPP supervised warm-start). |

## Using a custom oracle

All run scripts accept `--oracle_module <import.path>` to load custom oracles at startup:

```bash
python scripts/run_active_loop.py \
    --model_path model_v2.ckpt \
    --oracle my_oracle \
    --oracle_module examples.custom_oracle \
    --protein_id MY_TARGET \
    --output_dir outputs/my_run
```

See `examples/custom_oracle.py` for a working template.
