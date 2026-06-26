# genmol/rewards/

Oracle / reward functions for guided generation. Each reward is a callable:

```python
scores: torch.Tensor = reward(smiles_list: List[str])
```

## Built-in rewards

| Module | Class / function | Oracle cost | Notes |
|---|---|---|---|
| `properties.py` | `QEDReward`, `LogPReward`, `MolecularWeightReward`, `TPSAReward` | ~free (RDKit) | Base class `RewardFunction` — subclass for custom per-mol oracles |
| `flash_affinity.py` | `FlashAffinityForwardOp` | ~100-500 ms/mol | Binding probability from neural docking surrogate. Set `FLASHAFFINITY_ROOT` env var to point at your FA install. |
| `boltz.py` | `BoltzAffinityReward` | ~3-10 s/mol | Structure-based affinity via Boltz-2. |
| `kl_penalty.py` | `KLPenalizedReward` | free | Wraps another reward and subtracts a KL penalty term. |
| `threshold.py` | `ThresholdReward` | free | Clips reward to 0 below a threshold. |

## Registry and custom oracles

```python
from genmol.rewards import get_reward, register_reward

# Use a built-in
oracle = get_reward("flash_affinity", protein_id="2VT4")

# Register and use your own
register_reward("my_dock", MyDockingClass)
oracle = get_reward("my_dock", protein_pdb="target.pdb")
```

See `examples/custom_oracle.py` for a full working template.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `FLASHAFFINITY_ROOT` | `<repo>/FlashAffinity` | Path to FlashAffinity install (override if using your own copy) |
