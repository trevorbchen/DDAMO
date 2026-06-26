# configs/

Hydra configuration files for the GenMol framework. The main entry point `configs/experiment.yaml` composes defaults from the subfolders.

## Structure

```
configs/
├── experiment.yaml       # top-level experiment config (composes defaults below)
├── model.yaml            # GenMol model hyperparameters
├── reward/               # oracle / reward configs
│   ├── none.yaml         # no reward (unconditional generation)
│   ├── qed.yaml          # drug-likeness (RDKit QED)
│   ├── logp.yaml         # lipophilicity (RDKit LogP)
│   ├── mw.yaml           # molecular weight
│   ├── tpsa.yaml         # topological polar surface area
│   ├── flash_affinity.yaml  # FlashAffinity binding probability
│   └── boltz.yaml        # Boltz-2 binding affinity
├── sampler/              # inference-time sampler configs
│   ├── uncond.yaml       # unconditional (no guidance)
│   ├── beam_search.yaml
│   ├── mcts.yaml
│   ├── smc.yaml
│   ├── daps.yaml
│   └── dfkc.yaml
└── finetune/
    └── ddpp.yaml         # DDPP-LB fine-tuning hyperparameters
```

## Adding a custom reward config

Create `configs/reward/my_oracle.yaml`:
```yaml
name: my_oracle
protein_id: MY_TARGET
# any kwargs passed to get_reward("my_oracle", ...)
```

Then run with `reward=my_oracle` via Hydra overrides.
