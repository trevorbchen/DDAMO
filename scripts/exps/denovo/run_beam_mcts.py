# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import sys
sys.path.append(os.path.realpath('.'))

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


@hydra.main(version_base="1.3", config_path="config", config_name="run_beam_mcts")
def main(cfg: DictConfig):
    model_path = hydra.utils.to_absolute_path(cfg.model_path)
    output_base = hydra.utils.to_absolute_path(cfg.output_dir)
    
    # "name" is set per-run via CLI (e.g. name=budget_K5_L4_default) to tag the
    # output folder; it defaults to "standard" for plain unconditional sampling.
    sampler_name = cfg.get("name", "standard")
    reward_name = cfg.reward.get("type", "none")

    # Instantiate forward operator (reward function) if one is configured.
    # reward.target is a fully-qualified class path (e.g. "genmol.rewards.VinaScore")
    # resolved by Hydra; reward.params are passed as kwargs to its constructor.
    #
    # When no explicit reward is configured (target=None), beam search and MCTS
    # default to QEDForwardOp internally. We fix reward_name here so that the
    # output folder reads "qed/" instead of the misleading "none/".
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
    
    sampler = hydra.utils.instantiate(cfg.sampler, path=model_path, forward_op=forward_op)

    t_start = time()
    samples = sampler.de_novo_generation(
        cfg.num_samples,
        softmax_temp=cfg.softmax_temp,
        randomness=cfg.randomness,
        min_add_len=cfg.min_add_len,
    )

    elapsed = time() - t_start

    mw = [mol_weight(smi) for smi in samples]
    df = pd.DataFrame({"smiles": samples, "mol_wt": mw})
    
    # Generate dynamic CSV filename
    csv_name = f"samples.csv"
    out_csv = os.path.join(exp_folder, csv_name)
    df.to_csv(out_csv, index=False)
    
    # Save config summary to the same folder
    config_summary_path = os.path.join(exp_folder, "config.yaml")
    with open(config_summary_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # Validity = fraction of non-None SMILES (invalid molecules decoded as None).
    # Uniqueness = fraction of distinct SMILES among all returned samples.
    valid = df["smiles"].notna().sum() / max(cfg.num_samples, 1)
    uniq = df.drop_duplicates("smiles")["smiles"].count() / max(len(samples), 1)

    # Log budget if sampler tracked it.
    # "budget" = reward evals per molecule, "fp" = BERT forward passes per molecule.
    # These diverge because one rollout = multiple forward passes (depends on
    # remaining steps). fp_per_sample is the fair unit for Pareto comparisons.
    budget_per_sample = getattr(sampler, "last_budget_per_sample", 0)
    total_reward_evals = getattr(sampler, "last_reward_evals", 0)
    forward_passes = getattr(sampler, "last_forward_passes", 0)
    fp_per_sample = getattr(sampler, "last_fp_per_sample", 0)

    # Compute QED stats (always, regardless of reward function — QED is our
    # universal quality sanity check even when optimizing a different objective).
    # top10 = mean QED of the best 10% of samples, a more robust metric than max.
    valid_mols = [Chem.MolFromSmiles(s) for s in samples if s]
    valid_mols = [m for m in valid_mols if m is not None]
    qeds = sorted([QED.qed(m) for m in valid_mols], reverse=True)
    qed_mean  = sum(qeds) / len(qeds) if qeds else 0.0
    top10_n   = max(1, len(qeds) // 10)
    qed_top10 = sum(qeds[:top10_n]) / top10_n if qeds else 0.0
    qed_max   = qeds[0] if qeds else 0.0

    # Save metrics.json
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
    }
    metrics_path = os.path.join(exp_folder, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(OmegaConf.to_yaml(cfg))
    print(f"Time:\t\t{elapsed:.2f} sec")
    print(f"Output:\t{out_csv}")
    print(f"Config:\t{config_summary_path}")
    if budget_per_sample:
        print(f"Budget/sample:\t{budget_per_sample:.1f} reward evals")
        print(f"Total evals:\t{total_reward_evals}")
    print(f"Validity:\t{valid}")
    print(f"Uniqueness:\t{uniq}")
    print(f"QED mean:\t{qed_mean:.4f}  top10: {qed_top10:.4f}  max: {qed_max:.4f}")


if __name__ == "__main__":
    main()
