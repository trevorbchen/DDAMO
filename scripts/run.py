"""Unified experiment runner for GenMol.

Composes: sampler × reward (× optional DDPP-finetuned model).

Usage (from project root):
    python scripts/run.py sampler=beam_search reward=qed num_samples=100

    python scripts/run.py sampler=smc reward=flash_affinity \
        ddpp_checkpoint=outputs/ddpp/ddpp_checkpoint.pt

    python scripts/run.py sampler=mcts reward=boltz num_samples=20
"""

import json
import os
import sys

sys.path.insert(0, os.path.realpath("."))
sys.path.insert(0, os.path.join(os.path.realpath("."), "src"))

from time import time

import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd

from evals.metrics import compute_metrics


@hydra.main(version_base="1.3", config_path="../configs", config_name="experiment")
def main(cfg: DictConfig):
    model_path = hydra.utils.to_absolute_path(cfg.model_path)
    output_base = hydra.utils.to_absolute_path(cfg.output_dir)

    reward_name = cfg.reward.get("type", "none")
    sampler_name = cfg.get("name") or "default"
    finetune = cfg.get("finetune", "none")

    # ── Finetune: apply to GenMol before inference ────────────────────
    if finetune == "ddpp":
        ddpp_ckpt = cfg.get("ddpp_checkpoint")
        if ddpp_ckpt is None:
            raise ValueError("finetune=ddpp requires ddpp_checkpoint=<path>")
        from genmol.model_loader import merge_ddpp_checkpoint
        model_path = merge_ddpp_checkpoint(model_path, hydra.utils.to_absolute_path(ddpp_ckpt))

    # ── Reward ───────────────────────────────────────────────────────
    forward_op = None
    if cfg.reward.get("target") is not None:
        target = cfg.reward.get("target")
        params = OmegaConf.to_container(cfg.reward.get("params", {}), resolve=True)
        forward_op_class = hydra.utils.get_class(target)
        forward_op = forward_op_class(**params)
    elif reward_name not in ("none", ""):
        from genmol.rewards import get_reward
        forward_op = get_reward(reward_name)

    # ── Output path ──────────────────────────────────────────────────
    exp_folder = os.path.join(output_base, finetune, reward_name, sampler_name)
    os.makedirs(exp_folder, exist_ok=True)

    # ── Sampler ──────────────────────────────────────────────────────
    sampler = hydra.utils.instantiate(
        cfg.sampler, path=model_path, forward_op=forward_op,
    )

    # ── Generation ───────────────────────────────────────────────────
    t_start = time()
    samples = sampler.de_novo_generation(
        cfg.num_samples,
        softmax_temp=cfg.softmax_temp,
        randomness=cfg.randomness,
        min_add_len=cfg.min_add_len,
    )
    elapsed = time() - t_start

    # ── Metrics ──────────────────────────────────────────────────────
    m = compute_metrics(samples)

    df = pd.DataFrame({"smiles": samples, "mol_wt": m["mol_weights"]})
    out_csv = os.path.join(exp_folder, "samples.csv")
    df.to_csv(out_csv, index=False)

    with open(os.path.join(exp_folder, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    metrics = {
        "finetune": finetune,
        "sampler": sampler_name,
        "reward": reward_name,
        "elapsed_sec": elapsed,
        "budget_per_sample": getattr(sampler, "last_budget_per_sample", 0),
        "total_reward_evals": getattr(sampler, "last_reward_evals", 0),
        "forward_passes": getattr(sampler, "last_forward_passes", 0),
        "fp_per_sample": getattr(sampler, "last_fp_per_sample", 0),
        **{k: m[k] for k in ("validity", "uniqueness", "qed_mean", "qed_top10", "qed_max")},
        "num_samples": cfg.num_samples,
        "softmax_temp": cfg.softmax_temp,
        "randomness": cfg.randomness,
    }

    sampler_cfg = OmegaConf.to_container(cfg.sampler, resolve=True)
    for k, v in sampler_cfg.items():
        if not k.startswith("_") and k not in ("path", "forward_op"):
            metrics[f"sampler_{k}"] = v

    trajectory = getattr(sampler, "trajectory", [])
    if trajectory:
        metrics["has_trajectory"] = True
        with open(os.path.join(exp_folder, "trajectory.json"), "w") as f:
            json.dump(trajectory, f)

    with open(os.path.join(exp_folder, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Finetune: {finetune}  Sampler: {sampler_name}  Reward: {reward_name}")
    print(f"Time: {elapsed:.2f}s  Validity: {m['validity']:.4f}  Uniqueness: {m['uniqueness']:.4f}")
    print(f"QED: mean={m['qed_mean']:.4f}  top10={m['qed_top10']:.4f}  max={m['qed_max']:.4f}")
    print(f"Output: {out_csv}")


if __name__ == "__main__":
    main()
