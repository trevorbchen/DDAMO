"""Hydra-based generation runner for all samplers and rewards.

Usage:
    cd scripts/exps/denovo
    python run_generation.py sampler=beam_search reward=qed \
        model_path=model_v2.ckpt output_dir=outputs \
        name=beam_N8_L4 num_samples=100 \
        sampler.beam_width=8 sampler.branching_factor=4
"""

import json
import os
import sys

_project_root = os.path.realpath(".")
sys.path.append(_project_root)
sys.path.append(os.path.join(_project_root, "src"))

from time import time

import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, QED


def mol_weight(smiles):
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return float(Descriptors.MolWt(mol))


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


@hydra.main(version_base="1.3", config_path="config", config_name="generation")
def main(cfg: DictConfig):
    model_path = hydra.utils.to_absolute_path(cfg.model_path)
    output_base = hydra.utils.to_absolute_path(cfg.output_dir)

    sampler_name = cfg.get("name", "standard")
    reward_name = cfg.reward.get("type", "none")

    # Instantiate reward / forward operator
    forward_op = None
    if cfg.reward.get("target") is not None:
        target = cfg.reward.get("target")
        params = cfg.reward.get("params", {})
        forward_op_class = hydra.utils.get_class(target)
        forward_op = forward_op_class(**params)
    elif sampler_name not in ("standard", "uncond"):
        reward_name = "qed"

    exp_folder = os.path.join(output_base, reward_name, sampler_name)
    os.makedirs(exp_folder, exist_ok=True)

    # Instantiate sampler via Hydra
    sampler = hydra.utils.instantiate(
        cfg.sampler, path=model_path, forward_op=forward_op,
    )

    t_start = time()
    samples = sampler.de_novo_generation(
        cfg.num_samples,
        softmax_temp=cfg.softmax_temp,
        randomness=cfg.randomness,
        min_add_len=cfg.min_add_len,
    )
    elapsed = time() - t_start

    # DAPS returns bracket-SAFE strings that need conversion
    if sampler_name == "daps":
        samples = _postprocess_daps(samples)

    # Build dataframe
    mw = [mol_weight(smi) for smi in samples]
    df = pd.DataFrame({"smiles": samples, "mol_wt": mw})
    out_csv = os.path.join(exp_folder, "samples.csv")
    df.to_csv(out_csv, index=False)

    # Save config
    config_path = os.path.join(exp_folder, "config.yaml")
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # Compute metrics
    valid = df["smiles"].notna().sum() / max(cfg.num_samples, 1)
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
        "num_samples": cfg.num_samples,
        "name": sampler_name,
        "reward": reward_name,
        "sampler": sampler_name,
        "softmax_temp": cfg.softmax_temp,
        "randomness": cfg.randomness,
    }

    # Include sampler-specific hyperparams
    sampler_cfg = OmegaConf.to_container(cfg.sampler, resolve=True)
    for k, v in sampler_cfg.items():
        if k.startswith("_") or k in ("path", "forward_op"):
            continue
        metrics[k] = v

    # Save trajectory if available
    trajectory = getattr(sampler, "trajectory", [])
    if trajectory:
        metrics["has_trajectory"] = True
        traj_path = os.path.join(exp_folder, "trajectory.json")
        with open(traj_path, "w") as f:
            json.dump(trajectory, f)

    metrics_path = os.path.join(exp_folder, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(OmegaConf.to_yaml(cfg))
    print(f"Sampler:     {sampler_name}")
    print(f"Reward:      {reward_name}")
    print(f"Time:        {elapsed:.2f} sec")
    print(f"Output:      {out_csv}")
    if budget_per_sample:
        print(f"Budget/sample: {budget_per_sample:.1f} reward evals")
    print(f"Validity:    {valid:.4f}")
    print(f"Uniqueness:  {uniq:.4f}")
    print(f"QED mean:    {qed_mean:.4f}  top10: {qed_top10:.4f}  max: {qed_max:.4f}")


if __name__ == "__main__":
    main()
