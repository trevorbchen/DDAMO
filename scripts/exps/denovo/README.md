# De Novo Generation Experiments

Run molecule generation with different samplers and reward functions using `scripts/exps/denovo/run_generation.py` (Hydra-based).

All commands assume you are in the project root.

## Common parameters

These apply to all samplers:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_samples` | int | 100 | Number of molecules to generate |
| `softmax_temp` | float | 1.0 | Softmax temperature for token sampling |
| `randomness` | float | 0.3 | Fraction of tokens unmasked randomly vs. by confidence |
| `min_add_len` | int | 60 | Minimum sequence length |
| `model_path` | str | `model_v2.ckpt` | Path to model checkpoint |
| `output_dir` | str | `outputs` | Base output directory |
| `seed` | int | 0 | Random seed |

## Available rewards

`none`, `qed`, `mw`, `logp`, `tpsa` — set via `reward=<name>`.

## Output

Each run writes to `{output_dir}/{reward}/{name}/`:

- `samples.csv` — generated SMILES with molecular weights
- `metrics.json` — validity, uniqueness, QED stats, timing, budget
- `config.yaml` — full Hydra config snapshot

## Examples (QED reward)

### Standard (unconditional)

```bash
python scripts/exps/denovo/run_generation.py sampler=uncond reward=qed \
    num_samples=100 softmax_temp=0.8 randomness=0.5
```

No sampler-specific hyperparameters.

### Beam Search

```bash
python scripts/exps/denovo/run_generation.py sampler=beam_search reward=qed \
    num_samples=100 \
    sampler.beam_width=8 sampler.branching_factor=4 \
    sampler.diversity_penalty=0.1
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sampler.beam_width` | int | 4 | Number of beams (N) |
| `sampler.branching_factor` | int | 3 | Candidates per beam per expansion (L) |
| `sampler.steps_per_interval` | int | null | Denoising steps between expansions (K); null = total_steps // 4 |
| `sampler.elite_buffer_size` | int | null | Size of elite buffer; null = disabled |
| `sampler.diversity_cutoff` | float | null | Tanimoto threshold for elite buffer dedup; null = disabled |
| `sampler.diversity_penalty` | float | 0.0 | Tanimoto diversity penalty weight (lambda) |

### MCTS

```bash
python scripts/exps/denovo/run_generation.py sampler=mcts reward=qed \
    num_samples=100 \
    sampler.branching_factor=4 sampler.steps_per_interval=5 \
    sampler.c_uct=1.0
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sampler.branching_factor` | int | 4 | Children per node expansion |
| `sampler.steps_per_interval` | int | 5 | Denoising steps per tree level |
| `sampler.c_uct` | float | 1.0 | UCT exploration constant |
| `sampler.rollout_budget_per_sample` | int | null | Max rollouts per sample; null = unlimited |

### DAPS

```bash
python scripts/exps/denovo/run_generation.py sampler=daps reward=qed \
    num_samples=50 \
    sampler.num_steps=50 sampler.beta=100 \
    sampler.mh_steps=2 sampler.ode_steps=100
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sampler.num_steps` | int | 50 | Number of annealing steps |
| `sampler.beta` | float | 100.0 | Inverse temperature on reward (higher = stronger guidance) |
| `sampler.mh_steps` | int | 2 | Metropolis-Hastings refinement steps per annealing step |
| `sampler.max_mutations` | int | 1 | Max fragment mutations per MH proposal |
| `sampler.remask_max` | float | 0.6 | Max remask fraction (start of schedule) |
| `sampler.remask_min` | float | 0.05 | Min remask fraction (end of schedule) |
| `sampler.remask_schedule` | str | `linear` | Remask annealing schedule: `linear` |
| `sampler.ode_steps` | int | 100 | ODE steps for reverse diffusion (20 = poor quality) |
| `sampler.mutate_strategy` | str | `fragment` | Mutation strategy: `infill` (mask+model) or `fragment` (SAFE fragment swap) |
| `sampler.proposal_mask_frac` | float | 0.1 | Token fraction to remask for MH proposals (only used with `infill`) |
| `sampler.seed` | int | null | Random seed; null = use global `seed` |
| `sampler.verbose` | bool | false | Print per-step diagnostics |

### DFKC

Discrete Feynman-Kac Correctors ([Hasan et al., arXiv 2601.10403](https://arxiv.org/abs/2601.10403)).
SMC with Feynman-Kac weight correction and ESS-based resampling. Two modes:

- **reward** (default) — samples from the reward-tilted distribution `p(x) * exp(beta * r(x))`.
  Here `beta` is the **inverse temperature on the reward**, consistent with how beta is used
  elsewhere in the codebase (DAPS, beam search). Higher beta = stronger reward guidance.
  Weight increments use `d_beta_t * r(x)` (approximation of Corollary 3.6; exact per-token
  formulation is O(V*d) per step). Requires a reward function.
- **annealing** — samples from the sharpened base distribution `p(x)^beta` via scaled logits
  (Eq 15), with weight correction from Corollary 3.2. Here `beta` plays a different role:
  it is a **sharpness exponent on the model's own distribution**, not a reward temperature.
  `beta > 1` concentrates samples on high-likelihood modes; `beta < 1` increases diversity.
  No reward function needed — this only uses the pretrained model.

```bash
# Reward-tilted mode (default) — beta is inverse temperature on reward
python scripts/exps/denovo/run_generation.py sampler=dfkc reward=qed \
    num_samples=100 \
    sampler.num_particles=8 sampler.beta=2.0 \
    sampler.mode=reward

# Annealing mode — beta is sharpness exponent on p(x), no reward needed
python scripts/exps/denovo/run_generation.py sampler=dfkc reward=none \
    num_samples=100 \
    sampler.num_particles=8 sampler.beta=2.0 \
    sampler.mode=annealing sampler.beta_schedule=linear
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sampler.num_particles` | int | 8 | Number of particles (min 2) |
| `sampler.mode` | str | `reward` | Weighting mode: `annealing` or `reward` |
| `sampler.beta` | float | 2.0 | In reward mode: inverse temperature on reward (higher = stronger guidance). In annealing mode: sharpness exponent on p(x) (>1 sharpens, <1 flattens). Ramps from 1 to beta over denoising. |
| `sampler.beta_schedule` | str | `constant` | Beta annealing schedule: `linear`, `constant`, or `cosine` |
| `sampler.ess_threshold` | float | 0.5 | Effective sample size threshold for resampling (0-1) |
| `sampler.resample_strategy` | str | `systematic` | Resampling method: `systematic` or `multinomial` |
| `sampler.seed` | int | null | Random seed; null = use global `seed` |
| `sampler.verbose` | bool | false | Print per-step ESS and resampling events |

**Benchmark** (N=100, 20 particles, model_v2.ckpt):

| Method | Validity | QED mean | QED top10% | QED max |
|--------|----------|----------|------------|---------|
| Standard (uncond) | 1.000 | 0.8432 | 0.9402 | 0.9456 |
| DFKC reward (beta=1.0) | 1.000 | 0.8624 | 0.9469 | 0.9484 |

### SMC

```bash
python scripts/exps/denovo/run_generation.py sampler=smc reward=qed \
    num_samples=100 \
    sampler.num_particles=8 sampler.resample_interval=5 \
    sampler.beta=10.0
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sampler.num_particles` | int | 8 | Number of particles (min 2) |
| `sampler.resample_interval` | int | 5 | Resample every N denoising steps |
| `sampler.resample_start` | float | 0.5 | Fraction of denoising to complete before resampling begins (0-1) |
| `sampler.beta` | float | 10.0 | Inverse temperature on reward (higher = stronger guidance) |
| `sampler.ess_threshold` | float | 0.5 | Effective sample size threshold for resampling (0-1) |
| `sampler.resample_strategy` | str | `systematic` | Resampling method: `systematic` or `multinomial` |
