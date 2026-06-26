# genmol/finetune/

Fine-tuning algorithms that update the GenMol backbone weights during the active loop.

## Trainers

| Module | Class | Method |
|---|---|---|
| `ddpp.py` | `DDPPLBTrainer` | **DDPP-LB** (Discrete Denoising Posterior Prediction, lower-bound objective). Trains the model to sample from a reward-tilted distribution via a GFlowNet-style trajectory balance objective. Used in the main active loop. |
| `vidd.py` | `VIDDTrainer` | **VIDD** (Iterative Distillation). Off-policy KL-regularized fine-tuning. Collects oracle-scored rollouts, simulates a soft-optimal policy, and distills it into the model. |

## How fine-tuning fits in

In the active loop (`src/genmol/active_loop.py`):
1. Generate candidates from the current model.
2. Score top-K with the oracle.
3. Call the trainer's `step()` on the scored buffer.
4. Repeat.

The trainer holds its own optimizer and internal buffer; the active loop just feeds it (smiles, score) pairs each epoch.
