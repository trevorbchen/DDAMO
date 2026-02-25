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

import os
import sys
sys.path.append(os.path.realpath('.'))

from time import time
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors


def mol_weight(smiles):
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return float(Descriptors.MolWt(mol))


@hydra.main(version_base="1.3", config_path="config", config_name="run")
def main(cfg: DictConfig):
    model_path = hydra.utils.to_absolute_path(cfg.model_path)
    output_dir = hydra.utils.to_absolute_path(cfg.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    forward_op = hydra.utils.instantiate(cfg.reward) if cfg.reward.get("_target_") else None
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
    out_csv = os.path.join(output_dir, cfg.output_csv)
    df.to_csv(out_csv, index=False)

    valid = df["smiles"].notna().sum() / max(cfg.num_samples, 1)
    uniq = df.drop_duplicates("smiles")["smiles"].count() / max(len(samples), 1)

    print(OmegaConf.to_yaml(cfg))
    print(f"Time:\t\t{elapsed:.2f} sec")
    print(f"Output:\t{out_csv}")
    print(f"Validity:\t{valid}")
    print(f"Uniqueness:\t{uniq}")


if __name__ == "__main__":
    main()
