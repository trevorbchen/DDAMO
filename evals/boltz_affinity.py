"""Boltz affinity prediction for SMILES ligands against a fixed receptor.

Boltz is invoked via CLI. All YAML inputs are written to a temp directory,
then `boltz predict` is called once (batched). Results are parsed from the
output JSON files.
"""

import json
import hashlib
import subprocess
import shutil
from pathlib import Path
from typing import Optional

import torch


RECEPTOR_SEQUENCE = (
    "MGSSHHHHHHSSGNNFNNEIKLILQQYLEKFEAHYERVLQDDQYIEALETLMDDYSEFILNPIYEQQFNAWRDVEEKAQLIKSLQYITAQCVKQVEVIRARRLLDGQASTTGYFDNIEHCIDEEFGQCSITSNDKLLLVGSGAYPMTLIQVAKETGASVIGIDIDPQAVDLGRRIVNVLAPNEDITITDQKVSELKDIKDVTHIIFSSTIPLKYSILEELYDLTNENVVVAMRFGDGIKAIFNYPSQETAEDKWQCVNKHMRPQQIFDIALYKKAAIKVGITD"
)


def _smiles_hash(smiles: str, receptor_seq: str = "") -> str:
    hash_input = f"{smiles}_{receptor_seq}"
    return hashlib.md5(hash_input.encode("utf-8")).hexdigest()[:10]


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

    All inputs are written as YAML files, then `boltz predict` is called once
    on the whole directory (batched). Returns a list of affinity_pred_value
    floats (log10 IC50 in μM), or None for failed predictions.

    Args:
        smiles_list: SMILES strings to evaluate.
        receptor_seq: Fixed receptor amino acid sequence.
        input_dir: Directory to write YAML inputs.
        out_dir: Directory for Boltz outputs.
        diffusion_samples: Number of diffusion samples per ligand.
        sampling_steps: Number of sampling steps.
        recycling_steps: Number of recycling steps.
        devices: Number of GPU devices.
        num_workers: Dataloader workers.
        use_msa_server: Use MMSeqs2 MSA server.
        cleanup: Remove input YAMLs after prediction.

    Returns:
        List of affinity values (or None for failures).
    """
    input_path = Path(input_dir)
    out_path = Path(out_dir)
    input_path.mkdir(parents=True, exist_ok=True)
    out_path.mkdir(parents=True, exist_ok=True)

    # Write YAML inputs
    stems = []
    for smi in smiles_list:
        stem = f"lig_{_smiles_hash(smi, receptor_seq)}"
        stems.append(stem)
        yml = input_path / f"{stem}.yaml"
        if not yml.exists():
            _write_yaml(yml, receptor_seq, smi)

    # Build command
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

    # Parse results
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
        for stem in stems:
            yml = input_path / f"{stem}.yaml"
            yml.unlink(missing_ok=True)

    return affinities
