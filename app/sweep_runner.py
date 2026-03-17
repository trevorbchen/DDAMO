"""Multi-GPU hyperparameter sweep runner.

Launches ``run_beam_mcts.py`` subprocesses with round-robin GPU assignment,
following the same pattern as ``sweep_full_hp.sh``.  No Streamlit dependency.
"""

import itertools
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Grid / random search helpers
# ---------------------------------------------------------------------------

_SAMPLER_MAP = {
    "Beam Search": "beam_search",
    "MCTS": "mcts",
    "DAPS": "daps",
    "DFKC": "dfkc",
    "SMC": "smc",
    "Standard": "uncond",
}

_SAMPLER_PARAMS = {
    "Beam Search": [
        "sampler.beam_width", "sampler.branching_factor",
        "sampler.steps_per_interval", "sampler.diversity_penalty",
    ],
    "MCTS": [
        "sampler.branching_factor", "sampler.steps_per_interval",
        "sampler.c_uct",
    ],
    "DAPS": [
        "sampler.num_steps", "sampler.alpha",
        "sampler.mh_steps", "sampler.ode_steps",
    ],
    "DFKC": [
        "sampler.num_particles", "sampler.beta",
        "sampler.mode", "sampler.beta_schedule",
    ],
    "SMC": [
        "sampler.num_particles", "sampler.resample_interval",
        "sampler.alpha",
    ],
    "Standard": [],
}

_SHORT = {
    "sampler.beam_width": "N",
    "sampler.branching_factor": "L",
    "sampler.steps_per_interval": "K",
    "sampler.diversity_penalty": "dp",
    "sampler.c_uct": "c",
    "sampler.num_steps": "s",
    "sampler.alpha": "a",
    "sampler.mh_steps": "mh",
    "sampler.ode_steps": "ode",
    "sampler.num_particles": "K",
    "sampler.beta": "b",
    "sampler.mode": "m",
    "sampler.beta_schedule": "bs",
    "sampler.resample_interval": "ri",
    "softmax_temp": "t",
    "num_samples": "n",
}


def _fmt_val(v):
    """Format a value for inclusion in a run name."""
    if isinstance(v, float):
        if v == int(v):
            return str(int(v))
        return f"{v:g}".replace(".", "")
    return str(v)


def _make_name(sampler_type: str, overrides: dict) -> str:
    prefix = _SAMPLER_MAP.get(sampler_type, "run")
    parts = [prefix]
    for k, v in sorted(overrides.items()):
        short = _SHORT.get(k, k.split(".")[-1])
        parts.append(f"{short}{_fmt_val(v)}")
    return "_".join(parts)


def build_grid(sampler_type: str, param_ranges: dict[str, list]) -> list[dict]:
    """Cartesian product of parameter ranges.

    Returns list of config dicts, each with keys:
        name, sampler_type, overrides (dict of Hydra CLI overrides)
    """
    if not param_ranges:
        return [{"name": _SAMPLER_MAP.get(sampler_type, "run"),
                 "sampler_type": sampler_type, "overrides": {}}]

    keys = list(param_ranges.keys())
    values = [param_ranges[k] for k in keys]
    configs = []
    for combo in itertools.product(*values):
        overrides = dict(zip(keys, combo))
        name = _make_name(sampler_type, overrides)
        configs.append({
            "name": name,
            "sampler_type": sampler_type,
            "overrides": overrides,
        })
    return configs


def build_random(sampler_type: str, param_ranges: dict[str, list],
                 n: int, seed: int = 42) -> list[dict]:
    """Random sample from the cartesian product."""
    full = build_grid(sampler_type, param_ranges)
    rng = random.Random(seed)
    if n >= len(full):
        return full
    return rng.sample(full, n)


def detect_gpus() -> int:
    """Return number of available CUDA GPUs."""
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-L"], stderr=subprocess.DEVNULL, text=True,
        )
        return len([l for l in out.strip().splitlines() if l.startswith("GPU")])
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

class SweepRunner:
    """Manage multi-GPU sweep via subprocesses."""

    def __init__(self, sweep_dir: str, n_gpus: int, configs: list[dict],
                 model_path: str = "model_v2.ckpt", reward: str = "none"):
        self.sweep_dir = sweep_dir
        self.n_gpus = max(1, n_gpus)
        self.model_path = model_path
        self.reward = reward
        self.all_configs = list(configs)

        # State
        self.pending: list[dict] = []
        self.running: dict[str, tuple[subprocess.Popen, int]] = {}  # name -> (proc, gpu)
        self.completed: list[str] = []
        self.failed: list[str] = []
        self.errors: dict[str, str] = {}  # name -> stderr snippet
        self.gpu_free: list[int] = list(range(self.n_gpus))

        os.makedirs(sweep_dir, exist_ok=True)

        # Skip already-completed runs
        for cfg in self.all_configs:
            if self._find_metrics(cfg["name"]):
                self.completed.append(cfg["name"])
            else:
                self.pending.append(cfg)

        # Persist manifest
        self._write_manifest()

    def _find_metrics(self, name: str) -> str | None:
        """Find metrics.json for a run, checking both direct and reward-subdirectory paths.

        run_beam_mcts.py saves to: output_dir/reward_name/run_name/metrics.json
        """
        # Direct path: sweep_dir/name/metrics.json
        direct = os.path.join(self.sweep_dir, name, "metrics.json")
        if os.path.exists(direct):
            return direct
        # With reward subdirectory: sweep_dir/*/name/metrics.json
        for entry in os.scandir(self.sweep_dir):
            if entry.is_dir():
                sub = os.path.join(entry.path, name, "metrics.json")
                if os.path.exists(sub):
                    return sub
        return None

    # ── Command builder ────────────────────────────────────────────

    def _build_command(self, config: dict) -> list[str]:
        sampler_cli = _SAMPLER_MAP.get(config["sampler_type"], "uncond")
        cmd = [
            sys.executable,
            "scripts/exps/denovo/run_generation.py",
            "--sampler", sampler_cli,
            "--reward", self.reward,
            "--model_path", self.model_path,
            "--output_dir", self.sweep_dir,
            "--name", config["name"],
        ]
        for k, v in config["overrides"].items():
            # Convert dotted Hydra keys (sampler.beam_width) to flat CLI
            # flags (--beam_width)
            flag = k.split(".")[-1]
            cmd.extend([f"--{flag}", str(v)])
        return cmd

    # ── Lifecycle ──────────────────────────────────────────────────

    def launch(self):
        """Start initial batch — one job per free GPU."""
        self._fill_gpus()

    def poll(self) -> dict:
        """Check running processes, reap finished, launch next. Returns status."""
        finished = []
        for name, (proc, gpu) in list(self.running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            finished.append((name, gpu, rc, proc))

        for name, gpu, rc, proc in finished:
            del self.running[name]
            self.gpu_free.append(gpu)
            if rc == 0 and self._find_metrics(name):
                self.completed.append(name)
            else:
                self.failed.append(name)
                stderr = ""
                if proc.stderr:
                    try:
                        stderr = proc.stderr.read()
                        if isinstance(stderr, bytes):
                            stderr = stderr.decode(errors="replace")
                        stderr = stderr[-500:]  # last 500 chars
                    except Exception:
                        pass
                self.errors[name] = stderr or f"exit code {rc}"

        self._fill_gpus()
        self._write_manifest()
        return self.status()

    def stop(self):
        """Kill all running subprocesses."""
        for name, (proc, gpu) in list(self.running.items()):
            try:
                proc.kill()
            except Exception:
                pass
            self.gpu_free.append(gpu)
        self.running.clear()

    def status(self) -> dict:
        return {
            "total": len(self.all_configs),
            "completed": len(self.completed),
            "running": len(self.running),
            "pending": len(self.pending),
            "failed": len(self.failed),
            "completed_names": list(self.completed),
            "running_names": {n: gpu for n, (_, gpu) in self.running.items()},
            "failed_names": list(self.failed),
        }

    @property
    def is_done(self) -> bool:
        return len(self.pending) == 0 and len(self.running) == 0

    # ── Internal ───────────────────────────────────────────────────

    def _fill_gpus(self):
        """Launch pending jobs on free GPUs."""
        while self.pending and self.gpu_free:
            cfg = self.pending.pop(0)
            gpu = self.gpu_free.pop(0)
            cmd = self._build_command(cfg)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            )
            self.running[cfg["name"]] = (proc, gpu)

    def _write_manifest(self):
        manifest = {
            "sweep_dir": self.sweep_dir,
            "created": datetime.now().isoformat(),
            "n_gpus": self.n_gpus,
            "total_configs": len(self.all_configs),
            "status": self.status(),
        }
        try:
            with open(os.path.join(self.sweep_dir, "sweep_manifest.json"), "w") as f:
                json.dump(manifest, f, indent=2)
        except Exception:
            pass
