"""General-purpose generation runner (no Hydra dependency).

Accepts all sampler/reward/generation parameters via argparse CLI.
Used by the sweep runner and can also be invoked standalone.

Usage:
    python scripts/exps/denovo/run_generation.py \
        --sampler beam_search --reward qed \
        --model_path model_v2.ckpt --output_dir outputs/sweep \
        --name beam_N8_L4 --num_samples 100 \
        --softmax_temp 0.8 --randomness 0.5 \
        --beam_width 8 --branching_factor 4
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.realpath("."))

from time import time

def mol_weight(smiles):
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return float(Descriptors.MolWt(mol))


# ── Sampler construction ──────────────────────────────────────────────

_SAMPLER_CLASSES = {
    "uncond": ("genmol.sampler", "Sampler"),
    "beam_search": ("genmol.beam_search_sampler", "BeamSearchSampler"),
    "mcts": ("genmol.mcts_sampler", "MCTSSampler"),
    "daps": ("genmol.DAPS_sampler", "DAPSSampler"),
    "dfkc": ("genmol.smc_sampler", "DFKCSampler"),
    "smc": ("genmol.smc_sampler", "SMCSampler"),
}

# Parameters each sampler constructor accepts (beyond path + forward_op)
_SAMPLER_PARAMS = {
    "beam_search": [
        "beam_width", "branching_factor", "steps_per_interval",
        "elite_buffer_size", "diversity_cutoff", "diversity_penalty",
    ],
    "mcts": [
        "branching_factor", "steps_per_interval", "c_uct",
        "rollout_budget_per_sample",
    ],
    "daps": [
        "num_steps", "alpha", "mh_steps", "max_mutations",
        "remask_max", "remask_min", "remask_schedule", "ode_steps",
        "mutate_strategy", "proposal_mask_frac", "seed", "verbose",
    ],
    "uncond": [],
    "dfkc": [
        "num_particles", "mode", "beta", "beta_schedule",
        "ess_threshold", "resample_strategy", "seed", "verbose",
    ],
    "smc": [
        "num_particles", "resample_interval", "resample_start",
        "alpha", "ess_threshold", "resample_strategy", "seed", "verbose",
    ],
}


def _build_sampler(args):
    """Instantiate the sampler from CLI args."""
    from genmol.rewards import get_reward
    import importlib

    module_name, class_name = _SAMPLER_CLASSES[args.sampler]
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)

    # Resolve reward / forward_op
    forward_op = get_reward(args.reward) if args.reward != "none" else None

    # For beam/mcts with no explicit reward, they default to QED internally
    reward_name = args.reward
    if forward_op is None and args.sampler not in ("uncond",):
        reward_name = "qed"

    # Collect sampler-specific kwargs
    param_names = _SAMPLER_PARAMS.get(args.sampler, [])
    kwargs = {}
    for p in param_names:
        val = getattr(args, p, None)
        if val is not None:
            kwargs[p] = val

    sampler = cls(path=args.model_path, forward_op=forward_op, **kwargs)
    return sampler, reward_name


# ── Post-processing ───────────────────────────────────────────────────

def _postprocess_daps(samples):
    """DAPS returns bracket-SAFE strings — convert to SMILES."""
    from genmol.utils.utils_chem import safe_to_smiles
    from genmol.utils.bracket_safe_converter import bracketsafe2safe

    converted = []
    for s in samples:
        if not s:
            converted.append(None)
            continue
        smi = safe_to_smiles(s, fix=True)
        if not smi:
            try:
                smi = safe_to_smiles(bracketsafe2safe(s), fix=True)
            except Exception:
                smi = None
        if smi:
            smi = sorted(smi.split("."), key=len)[-1]
        converted.append(smi)
    return converted


# ── Main ──────────────────────────────────────────────────────────────

def run(args):
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import QED

    sampler, reward_name = _build_sampler(args)

    exp_folder = os.path.join(args.output_dir, reward_name, args.name)
    os.makedirs(exp_folder, exist_ok=True)

    t_start = time()
    samples = sampler.de_novo_generation(
        args.num_samples,
        softmax_temp=args.softmax_temp,
        randomness=args.randomness,
        min_add_len=args.min_add_len,
    )
    elapsed = time() - t_start

    if args.sampler == "daps":
        samples = _postprocess_daps(samples)

    # Build dataframe
    mw = [mol_weight(smi) for smi in samples]
    df = pd.DataFrame({"smiles": samples, "mol_wt": mw})
    out_csv = os.path.join(exp_folder, "samples.csv")
    df.to_csv(out_csv, index=False)

    # Save config
    config = vars(args).copy()
    config_path = os.path.join(exp_folder, "config.yaml")
    with open(config_path, "w") as f:
        for k, v in sorted(config.items()):
            f.write(f"{k}: {v}\n")

    # Compute metrics
    valid = df["smiles"].notna().sum() / max(args.num_samples, 1)
    uniq = df.drop_duplicates("smiles")["smiles"].count() / max(len(samples), 1)

    budget_per_sample = getattr(sampler, "last_budget_per_sample", 0)
    total_reward_evals = getattr(sampler, "last_reward_evals", 0)
    forward_passes = getattr(sampler, "last_forward_passes", 0)
    fp_per_sample = getattr(sampler, "last_fp_per_sample", 0)

    valid_mols = [Chem.MolFromSmiles(s) for s in samples if s]
    valid_mols = [m for m in valid_mols if m is not None]
    qeds = sorted([QED.qed(m) for m in valid_mols], reverse=True)
    qed_mean = sum(qeds) / len(qeds) if qeds else 0.0
    top10_n = max(1, len(qeds) // 10)
    qed_top10 = sum(qeds[:top10_n]) / top10_n if qeds else 0.0
    qed_max = qeds[0] if qeds else 0.0

    metrics = {
        "elapsed_sec": elapsed,
        "budget_per_sample": budget_per_sample,
        "total_reward_evals": total_reward_evals,
        "forward_passes": forward_passes,
        "fp_per_sample": fp_per_sample,
        "validity": float(valid),
        "uniqueness": float(uniq),
        "qed_mean": qed_mean,
        "qed_top10": qed_top10,
        "qed_max": qed_max,
        "num_samples": args.num_samples,
        "name": args.name,
        "reward": reward_name,
        "sampler": args.sampler,
        "softmax_temp": args.softmax_temp,
        "randomness": args.randomness,
    }
    # Add sampler-specific params
    for k in _SAMPLER_PARAMS.get(args.sampler, []):
        val = getattr(args, k, None)
        if val is not None:
            metrics[k] = val

    metrics_path = os.path.join(exp_folder, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Sampler:     {args.sampler}")
    print(f"Reward:      {reward_name}")
    print(f"Time:        {elapsed:.2f} sec")
    print(f"Output:      {out_csv}")
    if budget_per_sample:
        print(f"Budget/sample: {budget_per_sample:.1f} reward evals")
    print(f"Validity:    {valid:.4f}")
    print(f"Uniqueness:  {uniq:.4f}")
    print(f"QED mean:    {qed_mean:.4f}  top10: {qed_top10:.4f}  max: {qed_max:.4f}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="GenMol generation runner")

    # Core
    p.add_argument("--sampler", default="uncond",
                   choices=["uncond", "beam_search", "mcts", "daps",
                            "dfkc", "smc"])
    p.add_argument("--reward", default="none",
                   choices=["none", "qed", "mw", "logp", "tpsa"])
    p.add_argument("--model_path", default="model_v2.ckpt")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--name", default="run")

    # Generation
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--softmax_temp", type=float, default=0.8)
    p.add_argument("--randomness", type=float, default=0.5)
    p.add_argument("--min_add_len", type=int, default=40)

    # Beam Search
    p.add_argument("--beam_width", type=int, default=None)
    p.add_argument("--branching_factor", type=int, default=None)
    p.add_argument("--steps_per_interval", type=int, default=None)
    p.add_argument("--elite_buffer_size", type=int, default=None)
    p.add_argument("--diversity_cutoff", type=float, default=None)
    p.add_argument("--diversity_penalty", type=float, default=None)

    # MCTS
    p.add_argument("--c_uct", type=float, default=None)
    p.add_argument("--rollout_budget_per_sample", type=int, default=None)

    # DAPS
    p.add_argument("--num_steps", type=int, default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--mh_steps", type=int, default=None)
    p.add_argument("--max_mutations", type=int, default=None)
    p.add_argument("--remask_max", type=float, default=None)
    p.add_argument("--remask_min", type=float, default=None)
    p.add_argument("--remask_schedule", default=None)
    p.add_argument("--ode_steps", type=int, default=None)
    p.add_argument("--mutate_strategy", default=None)
    p.add_argument("--proposal_mask_frac", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--verbose", action="store_true", default=None)

    # DFKC / SMC
    p.add_argument("--num_particles", type=int, default=None)
    p.add_argument("--mode", default=None, choices=["annealing", "reward"])
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--beta_schedule", default=None,
                   choices=["linear", "constant", "cosine"])
    p.add_argument("--ess_threshold", type=float, default=None)
    p.add_argument("--resample_strategy", default=None,
                   choices=["systematic", "multinomial"])
    p.add_argument("--resample_interval", type=int, default=None)
    p.add_argument("--resample_start", type=float, default=None)

    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
