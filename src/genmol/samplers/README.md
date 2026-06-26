# genmol/samplers/

Inference-time sampling strategies for guided generation from the GenMol masked diffusion model. All samplers take a loaded model and a forward operator (reward/oracle) and return SMILES strings.

## Samplers

| Module | Class | Strategy |
|---|---|---|
| `base.py` | `BaseSampler` | Abstract base; unconditional generation (no guidance). |
| `beam_search.py` | `BeamSearchSampler` | Left-to-right beam search guided by a forward operator score at each step. |
| `mcts.py` | `MCTSSampler` | Monte Carlo Tree Search over the SAFE token space, guided by oracle scores at leaf nodes. |
| `smc.py` | `SMCSampler` | Sequential Monte Carlo (particle filter) with resampling at each denoising step. |
| `daps.py` | `DAPSSampler` | Diffusion Posterior Sampling — gradient-based guidance via Tweedie-denoised score estimates. |

## Usage

Samplers are configured via Hydra configs in `configs/sampler/` and instantiated by the run scripts. To use one directly:

```python
from genmol.samplers import BeamSearchSampler
sampler = BeamSearchSampler(model, forward_op=my_oracle, beam_width=8)
smiles = sampler.sample(n=100)
```
