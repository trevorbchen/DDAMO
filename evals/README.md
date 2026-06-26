# evals/

Evaluation utilities for scoring generated molecules after a run.

| Module | What it does |
|---|---|
| `flash_affinity.py` | Batch-score a list of SMILES with FlashAffinity. Used to re-evaluate saved candidate sets. |
| `boltz_affinity.py` | Batch-score with Boltz-2 affinity head. |
| `metrics.py` | Diversity (Tanimoto), validity, uniqueness, SA score, QED, and top-K aggregation metrics over a candidate set. |

## Typical usage

After a run, the `oracle_timeline.jsonl` already contains oracle scores for every molecule evaluated during the active loop. The evals scripts are for:
1. Re-scoring the final top-K candidate set with a slower/different oracle.
2. Computing diversity and drug-likeness metrics on the saved candidates.

```python
from evals.metrics import compute_metrics
results = compute_metrics(smiles_list, reference_smiles=anchor_smiles)
```
