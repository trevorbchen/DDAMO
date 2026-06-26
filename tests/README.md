# tests/

Unit and integration tests for the core library.

```bash
cd ~/genmol-app
.venv/bin/pytest tests/ -v
```

| Test file | What it covers |
|---|---|
| `test_vidd.py` | VIDD trainer: forward pass, buffer updates, KL loss correctness. |
| `test_ablation_configs.py` | Validates that all ablation config combinations (no-CVaR, no-buffer, no-Thompson, no-expA) instantiate without error and produce sensible oracle timelines on a mock oracle. |
| `conftest.py` | Shared fixtures: mock GenMol model, mock oracle, temp output dirs. |

## Running a quick smoke test

```bash
.venv/bin/pytest tests/ -v -k "not slow"
```
