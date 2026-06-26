# examples/

Working code templates for extending GenMol with your own models and oracles.

## custom_oracle.py

Shows two patterns for plugging in a proprietary binding-affinity oracle (or any scoring function) without modifying library code.

**Pattern A — per-molecule scoring** (subclass `RewardFunction`):
- Override `_score_mol(mol) -> float` where `mol` is an RDKit Mol.
- Invalid SMILES, parsing errors, and exceptions are handled automatically.
- Best for models that score one molecule at a time.

**Pattern B — batch callable** (register any function):
- Register `fn(smiles_list) -> torch.Tensor` directly.
- You control batching and error handling.
- Best for GPU neural nets that benefit from batched inference.

```python
from genmol.rewards import register_reward, get_reward

register_reward("my_oracle", MyOracleClass)
oracle = get_reward("my_oracle", protein_id="2VT4")
scores = oracle(["CCO", "c1ccccc1"])   # → tensor([0.42, 0.71])
```

Then in any run script:
```bash
python scripts/run_active_loop.py \
    --model_path model_v2.ckpt \
    --oracle my_oracle \
    --oracle_module examples.custom_oracle
```
