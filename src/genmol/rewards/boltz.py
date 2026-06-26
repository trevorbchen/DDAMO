"""Boltz structure-based affinity prediction.

Contains:
- ``run_boltz_affinity``: CLI wrapper (for standalone/eval use)
- ``BoltzAffinityReward``: Persistent-model reward callable for samplers.
  Loads structure + affinity models ONCE at init, reuses across calls.
  Includes score cache to skip duplicate molecules.

Usage:
    from genmol.rewards.boltz import BoltzAffinityReward, run_boltz_affinity
"""

import json
import hashlib
import os
import subprocess
import shutil
import sys
import tempfile
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import torch


RECEPTOR_SEQUENCE = (
    "MGSSHHHHHHSSGNNFNNEIKLILQQYLEKFEAHYERVLQDDQYIEALETLMDDYSEFILNPIYEQQFNAWRDVEEKAQLIKSLQYITAQCVKQVEVIRARRLLDGQASTTGYFDNIEHCIDEEFGQCSITSNDKLLLVGSGAYPMTLIQVAKETGASVIGIDIDPQAVDLGRRIVNVLAPNEDITITDQKVSELKDIKDVTHIIFSSTIPLKYSILEELYDLTNENVVVAMRFGDGIKAIFNYPSQETAEDKWQCVNKHMRPQQIFDIALYKKAAIKVGITD"
)


# ── Helpers ──────────────────────────────────────────────────────────

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


def _parse_affinity_results(pred_root: Path, stems: list[str]) -> list[Optional[float]]:
    """Parse affinity JSON results from Boltz predictions directory."""
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
    return affinities


# ── Raw Boltz CLI wrapper (for standalone/eval use) ──────────────────

def run_boltz_affinity(
    smiles_list: list[str],
    receptor_seq: str = RECEPTOR_SEQUENCE,
    input_dir: str = None,
    out_dir: str = None,
    diffusion_samples: int = 16,
    sampling_steps: int = 150,
    recycling_steps: int = 5,
    devices: int = 1,
    num_workers: int = 8,
    use_msa_server: bool = True,
    cleanup: bool = True,
    gpu_id: int = None,
) -> list[Optional[float]]:
    """Run Boltz affinity prediction via CLI subprocess.

    Each call spawns a new process and loads models from scratch.
    Use ``BoltzAffinityReward`` for persistent-model inference.
    """
    if input_dir is None:
        input_path = Path(tempfile.mkdtemp(prefix="boltz_in_"))
    else:
        input_path = Path(input_dir)
        input_path.mkdir(parents=True, exist_ok=True)

    if out_dir is None:
        out_path = Path(tempfile.mkdtemp(prefix="boltz_out_"))
    else:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

    stems = []
    for smi in smiles_list:
        stem = f"lig_{_smiles_hash(smi, receptor_seq)}"
        stems.append(stem)
        yml = input_path / f"{stem}.yaml"
        if not yml.exists():
            _write_yaml(yml, receptor_seq, smi)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    boltz_bin = shutil.which("boltz") or os.path.join(
        os.path.dirname(os.path.realpath(sys.executable)), "boltz")
    cmd = [
        boltz_bin, "predict", str(input_path),
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

    env = None
    if gpu_id is not None:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"Running Boltz CLI ({len(smiles_list)} mols)")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Boltz failed:\n{result.stderr}")

    result_folder = f"boltz_results_{input_path.name}"
    pred_root = out_path / result_folder / "predictions"
    affinities = _parse_affinity_results(pred_root, stems)

    if cleanup:
        shutil.rmtree(input_path, ignore_errors=True)
        shutil.rmtree(out_path, ignore_errors=True)

    return affinities


# ── Persistent-model reward callable ─────────────────────────────────

class BoltzAffinityReward:
    """Boltz affinity as a reward callable for samplers.

    Loads structure + affinity models ONCE at init, keeps them on GPU.
    Subsequent calls only do: write YAML -> process inputs -> predict -> parse.

    Scores are negated log10 IC50 so that higher = stronger binding.
    Failed predictions receive ``-inf``.
    Includes score cache to skip duplicate molecules.
    """

    def __init__(
        self,
        receptor_seq: Optional[str] = None,
        diffusion_samples: int = 16,
        sampling_steps: int = 200,
        recycling_steps: int = 3,
        diffusion_samples_affinity: int = 3,
        sampling_steps_affinity: int = 200,
        gpu_id: int = None,
        cache_dir: str = "~/.boltz",
        num_workers: int = 2,
        **kwargs,
    ):
        self._receptor_seq = receptor_seq or RECEPTOR_SEQUENCE
        self._diffusion_samples = diffusion_samples
        self._sampling_steps = sampling_steps
        self._recycling_steps = recycling_steps
        self._diffusion_samples_affinity = diffusion_samples_affinity
        self._sampling_steps_affinity = sampling_steps_affinity
        self._num_workers = num_workers
        self._gpu_id = gpu_id
        self._score_cache = {}
        self._call_counter = 0

        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Load models once (same GPU as GenMol)
        self._load_models(cache_dir)
        print(f"BoltzAffinityReward: models loaded, "
              f"diffusion_samples={diffusion_samples}")

    def _load_models(self, cache_dir: str):
        """Load Boltz2 structure and affinity models."""
        from boltz.main import (
            download_boltz2, Boltz2DiffusionParams, PairformerArgsV2,
            MSAModuleArgs, BoltzSteeringParams, load_canonicals,
        )
        from boltz.model.models.boltz2 import Boltz2

        warnings.filterwarnings("ignore", ".*Tensor Cores.*")
        # Do NOT set torch.set_grad_enabled(False) globally — breaks DDPP training.
        torch.set_float32_matmul_precision("highest")

        from rdkit import Chem
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

        for key in ["CUEQ_DEFAULT_CONFIG", "CUEQ_DISABLE_AOT_TUNING"]:
            os.environ[key] = os.environ.get(key, "1")

        cache = Path(cache_dir).expanduser()
        cache.mkdir(parents=True, exist_ok=True)
        download_boltz2(cache)

        self._cache = cache
        self._ccd_path = cache / "ccd.pkl"
        self._mol_dir = cache / "mols"

        # Diffusion params
        diffusion_params = Boltz2DiffusionParams()
        diffusion_params.step_scale = 1.5
        pairformer_args = PairformerArgsV2()
        msa_args = MSAModuleArgs(
            subsample_msa=True, num_subsampled_msa=1024, use_paired_feature=True,
        )

        # Structure model
        struct_ckpt = cache / "boltz2_conf.ckpt"
        steering_args = BoltzSteeringParams()
        predict_args = {
            "recycling_steps": self._recycling_steps,
            "sampling_steps": self._sampling_steps,
            "diffusion_samples": self._diffusion_samples,
            "max_parallel_samples": None,
            "write_confidence_summary": True,
            "write_full_pae": False,
            "write_full_pde": False,
        }
        self._struct_model = Boltz2.load_from_checkpoint(
            struct_ckpt, strict=True, predict_args=predict_args,
            map_location="cpu",
            diffusion_process_args=asdict(diffusion_params),
            ema=False, use_kernels=False,
            pairformer_args=asdict(pairformer_args),
            msa_args=asdict(msa_args),
            steering_args=asdict(steering_args),
        )
        self._struct_model.eval()

        # Affinity model
        aff_ckpt = cache / "boltz2_aff.ckpt"
        steering_args_aff = BoltzSteeringParams()
        steering_args_aff.fk_steering = False
        steering_args_aff.physical_guidance_update = False
        steering_args_aff.contact_guidance_update = False
        predict_aff_args = {
            "recycling_steps": 5,
            "sampling_steps": self._sampling_steps_affinity,
            "diffusion_samples": self._diffusion_samples_affinity,
            "max_parallel_samples": 1,
            "write_confidence_summary": False,
            "write_full_pae": False,
            "write_full_pde": False,
        }
        self._aff_model = Boltz2.load_from_checkpoint(
            aff_ckpt, strict=True, predict_args=predict_aff_args,
            map_location="cpu",
            diffusion_process_args=asdict(diffusion_params),
            ema=False,
            pairformer_args=asdict(pairformer_args),
            msa_args=asdict(msa_args),
            steering_args=asdict(steering_args_aff),
            affinity_mw_correction=False,
        )
        self._aff_model.eval()

        self._diffusion_params = diffusion_params
        self._pairformer_args = pairformer_args
        self._msa_args = msa_args

    def _run_prediction(self, smiles_list: list[str]) -> list[Optional[float]]:
        """Run structure + affinity prediction using persistent models."""
        from boltz.main import process_inputs, check_inputs, filter_inputs_structure, filter_inputs_affinity
        from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
        from boltz.data.types import Manifest
        from boltz.data.write.writer import BoltzWriter, BoltzAffinityWriter
        from pytorch_lightning import Trainer

        input_path = Path(tempfile.mkdtemp(prefix="boltz_in_"))
        out_path = Path(tempfile.mkdtemp(prefix="boltz_out_"))

        stems = []
        for smi in smiles_list:
            stem = f"lig_{_smiles_hash(smi, self._receptor_seq)}"
            stems.append(stem)
            _write_yaml(input_path / f"{stem}.yaml", self._receptor_seq, smi)

        # Process inputs
        data = check_inputs(input_path)
        out_dir = out_path / f"boltz_results_{input_path.name}"
        out_dir.mkdir(parents=True, exist_ok=True)

        process_inputs(
            data=data, out_dir=out_dir,
            ccd_path=self._ccd_path, mol_dir=self._mol_dir,
            use_msa_server=False, msa_server_url="",
            msa_pairing_strategy="greedy", boltz2=True,
        )

        manifest = Manifest.load(out_dir / "processed" / "manifest.json")
        processed_dir = out_dir / "processed"

        extra_mols_dir = (processed_dir / "mols") if (processed_dir / "mols").exists() else None
        constraints_dir = (processed_dir / "constraints") if (processed_dir / "constraints").exists() else None
        template_dir = (processed_dir / "templates") if (processed_dir / "templates").exists() else None

        # --- Structure prediction ---
        filtered = filter_inputs_structure(manifest=manifest, outdir=out_dir, override=True)
        if filtered.records:
            struct_writer = BoltzWriter(
                data_dir=processed_dir / "structures",
                output_dir=out_dir / "predictions",
                output_format="mmcif", boltz2=True, write_embeddings=False,
            )
            trainer = Trainer(
                default_root_dir=str(out_dir), callbacks=[struct_writer],
                accelerator="gpu", devices=1, precision="bf16-mixed",
                enable_progress_bar=False, logger=False,
            )
            data_module = Boltz2InferenceDataModule(
                manifest=filtered, target_dir=processed_dir / "structures",
                msa_dir=processed_dir / "msa", mol_dir=self._mol_dir,
                num_workers=self._num_workers,
                extra_mols_dir=extra_mols_dir,
                constraints_dir=constraints_dir,
                template_dir=template_dir,
            )
            trainer.predict(self._struct_model, datamodule=data_module,
                            return_predictions=False)

        # --- Affinity prediction ---
        if any(r.affinity for r in manifest.records):
            aff_filtered = filter_inputs_affinity(
                manifest=manifest, outdir=out_dir, override=True)
            if aff_filtered.records:
                aff_writer = BoltzAffinityWriter(
                    data_dir=processed_dir / "structures",
                    output_dir=out_dir / "predictions",
                )
                trainer = Trainer(
                    default_root_dir=str(out_dir), callbacks=[aff_writer],
                    accelerator="gpu", devices=1, precision="bf16-mixed",
                    enable_progress_bar=False, logger=False,
                )
                data_module = Boltz2InferenceDataModule(
                    manifest=aff_filtered,
                    target_dir=out_dir / "predictions",
                    msa_dir=processed_dir / "msa", mol_dir=self._mol_dir,
                    num_workers=self._num_workers,
                    extra_mols_dir=extra_mols_dir,
                    constraints_dir=constraints_dir,
                    template_dir=template_dir,
                    override_method="other", affinity=True,
                )
                trainer.predict(self._aff_model, datamodule=data_module,
                                return_predictions=False)

        pred_root = out_dir / "predictions"
        affinities = _parse_affinity_results(pred_root, stems)

        shutil.rmtree(input_path, ignore_errors=True)
        shutil.rmtree(out_path, ignore_errors=True)

        return affinities

    def __call__(self, smiles_list: List[str]) -> torch.Tensor:
        self._call_counter += 1

        scores = [None] * len(smiles_list)
        new_indices = []
        new_smiles = []
        for i, smi in enumerate(smiles_list):
            if smi in self._score_cache:
                scores[i] = self._score_cache[smi]
            else:
                new_indices.append(i)
                new_smiles.append(smi)

        if new_smiles:
            affinities = self._run_prediction(new_smiles)
            for idx, smi, aff in zip(new_indices, new_smiles, affinities):
                score = -aff if aff is not None else float("-inf")
                self._score_cache[smi] = score
                scores[idx] = score

        return torch.tensor(scores, dtype=torch.float32)

    @property
    def cache_size(self):
        return len(self._score_cache)
