"""Boltz structure-based affinity prediction.

Contains the raw CLI wrapper (``run_boltz_affinity``) and the reward
callable (``BoltzAffinityReward``) for use as ``forward_op`` in samplers.

~3-10s per molecule.

Usage:
    from genmol.rewards.boltz import BoltzAffinityReward, run_boltz_affinity
"""

import json
import hashlib
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional

import torch


RECEPTOR_SEQUENCE = (
    "MGSSHHHHHHSSGNNFNNEIKLILQQYLEKFEAHYERVLQDDQYIEALETLMDDYSEFILNPIYEQQFNAWRDVEEKAQLIKSLQYITAQCVKQVEVIRARRLLDGQASTTGYFDNIEHCIDEEFGQCSITSNDKLLLVGSGAYPMTLIQVAKETGASVIGIDIDPQAVDLGRRIVNVLAPNEDITITDQKVSELKDIKDVTHIIFSSTIPLKYSILEELYDLTNENVVVAMRFGDGIKAIFNYPSQETAEDKWQCVNKHMRPQQIFDIALYKKAAIKVGITD"
)


# ── Raw Boltz CLI wrapper ────────────────────────────────────────────

def _smiles_hash(smiles: str, receptor_seq: str = "") -> str:
    return hashlib.md5(f"{smiles}_{receptor_seq}".encode("utf-8")).hexdigest()[:10]


def _write_yaml(path: Path, receptor_seq: str, ligand_smiles: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        "      id: A\n"
        f"      sequence: {receptor_seq}\n"
        "      msa: empty\n"
        "  - ligand:\n"
        "      id: L\n"
        f"      smiles: '{ligand_smiles}'\n"
        "properties:\n"
        "  - affinity:\n"
        "      binder: L\n"
    )


def run_boltz_affinity(
    smiles_list: list[str],
    receptor_seq: str = RECEPTOR_SEQUENCE,
    input_dir: str = "boltz_inputs",
    out_dir: str = "boltz_outputs",
    diffusion_samples: int = 16,
    sampling_steps: int = 150,
    recycling_steps: int = 5,
    devices: int = 1,
    num_workers: int = 8,
    use_msa_server: bool = True,
    cleanup: bool = False,
) -> list[Optional[float]]:
    """Run Boltz affinity prediction on a list of SMILES.

    Writes YAML inputs, calls ``boltz predict`` once (batched), parses results.

    Returns:
        List of affinity_pred_value floats (log10 IC50 in uM), or None for failures.
    """
    input_path = Path(input_dir)
    out_path = Path(out_dir)
    input_path.mkdir(parents=True, exist_ok=True)
    out_path.mkdir(parents=True, exist_ok=True)

    stems = []
    for smi in smiles_list:
        stem = f"lig_{_smiles_hash(smi, receptor_seq)}"
        stems.append(stem)
        yml = input_path / f"{stem}.yaml"
        if not yml.exists():
            _write_yaml(yml, receptor_seq, smi)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    cmd = [
        "boltz", "predict", str(input_path),
        "--out_dir", str(out_path),
        "--accelerator", accelerator,
        "--devices", str(devices),
        "--diffusion_samples", str(diffusion_samples),
        "--recycling_steps", str(recycling_steps),
        "--num_workers", str(num_workers),
        "--sampling_steps", str(sampling_steps),
        "--no_kernels",
    ]
    if use_msa_server:
        cmd.append("--use_msa_server")

    print(f"Running Boltz: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Boltz failed:\n{result.stderr}")

    result_folder = f"boltz_results_{input_path.name}"
    pred_root = out_path / result_folder / "predictions"

    affinities = []
    for stem in stems:
        f = pred_root / stem / f"affinity_{stem}.json"
        if f.exists():
            try:
                data = json.loads(f.read_text())
                affinities.append(float(data["affinity_pred_value"]))
            except Exception as e:
                print(f"  Warning: failed to parse {f.name}: {e}")
                affinities.append(None)
        else:
            print(f"  Warning: missing affinity for {stem}")
            affinities.append(None)

    if cleanup:
        result_root = out_path / result_folder
        for subdir in ["structures", "processed"]:
            d = result_root / subdir
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
        for stem in stems:
            yml = input_path / f"{stem}.yaml"
            yml.unlink(missing_ok=True)
            stem_pred = pred_root / stem
            if stem_pred.exists():
                for f in stem_pred.iterdir():
                    if f.name != f"affinity_{stem}.json":
                        f.unlink(missing_ok=True)

    return affinities


# ── Reward callable ──────────────────────────────────────────────────

class BoltzAffinityReward:
    """Boltz affinity as a reward callable for samplers.

    Scores are negated log10 IC50 so that higher = stronger binding.
    Failed predictions receive ``-inf``.
    """

    def __init__(
        self,
        receptor_seq: Optional[str] = None,
        diffusion_samples: int = 4,
        input_dir: str = "boltz_inputs",
        out_dir: str = "boltz_outputs",
        **kwargs,
    ):
        self._receptor_seq = receptor_seq or RECEPTOR_SEQUENCE
        self._kwargs = {
            "diffusion_samples": diffusion_samples,
            "input_dir": input_dir,
            "out_dir": out_dir,
            **kwargs,
        }

    def __call__(self, smiles_list: List[str]) -> torch.Tensor:
        affinities = run_boltz_affinity(
            smiles_list,
            receptor_seq=self._receptor_seq,
            **self._kwargs,
        )
        scores = [
            (-a if a is not None else float("-inf"))
            for a in affinities
        ]
        return torch.tensor(scores, dtype=torch.float32)
