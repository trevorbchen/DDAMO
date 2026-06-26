# genmol/utils/

Internal utilities used across the library.

| Module | What it contains |
|---|---|
| `utils_chem.py` | SMILES/SAFE conversion, validity checking, deduplication, RDKit helpers. |
| `utils_data.py` | Dataset loading, tokenization, batching helpers for the GenMol vocab. |
| `utils_save.py` | Checkpoint saving/loading, `oracle_timeline.jsonl` logging format used by all run scripts. |
| `utils_moco.py` | Momentum-contrast utilities (used internally by the ensemble scorer). |
| `bracket_safe_converter.py` | Converts bracket-SAFE notation (from aggressive fine-tuning) back to standard SAFE for parsing. |
| `ema.py` | Exponential moving average wrapper for model parameters. |

## Output format

All run scripts log oracle calls to `outputs/<run_name>/oracle_timeline.jsonl`, one JSON object per line:

```json
{"step": 42, "smiles": "CCO...", "score": 0.81, "elapsed_sec": 312.4}
```

This is the canonical format used for all downstream analysis and plotting.
