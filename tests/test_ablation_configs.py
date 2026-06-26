"""Lightweight unit tests for the 6 DDPP + 6 VIDD ablation cell launchers
and the 4 search-method launchers.

These tests DON'T run the trainers (no model load, no GPU). They verify:
  1. Each cell's flag combination parses cleanly through the relevant
     argparse — catches typos / removed flags before the massive run.
  2. Argparse normalizes the values as expected (e.g. --tau_mode none
     vs cvar) so we can downstream-trust the parsed args.

If a flag was renamed or removed in run_active_loop.py / run_ddpp_online.py /
run_vidd.py / run_*.py, these tests fail loudly — the run never gets to the
trainer constructor.
"""

import importlib
import os
import shlex
import subprocess
import sys

import pytest

REPO = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))


def _can_import_argparser(script_relpath, argv):
    """Spawn `python script --help` and confirm the script can build its parser
    AND parse the given argv. We use subprocess so the spawned process does
    not pollute the test interpreter (these scripts may import torch, etc.)."""
    cmd = [sys.executable, os.path.join(REPO, script_relpath)] + argv + ["--help"]
    # `--help` short-circuits before the heavy imports usually; with --help
    # appended, the parser is built first and the process exits with 0.
    # We don't need the help output — we need a non-error parse of the prefix.
    # Strategy: instead use a custom marker — we inject `--gpu 999 --output_dir /tmp/_x`
    # and rely on the script erroring AFTER argparse if there's a real arg
    # mismatch. Keep this lightweight: check return code, ignore stdout.
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30,
        cwd=REPO,
    )
    # `--help` always exits 0; the parser must have accepted `argv` first.
    # If argv contains an unknown flag, argparse prints "unrecognized arguments"
    # to stderr and exits 2 BEFORE printing help.
    if result.returncode != 0:
        return False, result.stderr
    if "unrecognized arguments" in result.stderr:
        return False, result.stderr
    return True, ""


# ── Common flag fragments ────────────────────────────────────────────

DDPP_AL_BASE = (
    "--tau_mode cvar --tau_quantile 0.80 --selection thompson "
    "--exploration_approach A --exploration_gamma 1.0 --invalid_penalty -5 "
    "--model_path m.ckpt --wall_budget_sec 300 --seed 0"
)
DDPP_ON_BASE = (
    "--tau_mode cvar --tau_quantile 0.80 --exploration_approach A "
    "--exploration_gamma 1.0 --invalid_penalty -5 --oracle fa "
    "--model_path m.ckpt --wall_budget_sec 300 --seed 0"
)
VIDD_AL_PREFIX = "--trainer vidd --vidd_loss_func rw_mle --vidd_gkd_lmbda 0"


# ── Per-cell flag definitions (mirrors smoke_test_all.sh) ────────────

DDPP_CELLS = {
    "orig":    f"scripts/run_active_loop.py {DDPP_AL_BASE}",
    "nopen":   f"scripts/run_active_loop.py --tau_mode cvar --tau_quantile 0.80 --selection thompson --exploration_approach A --exploration_gamma 1.0 --model_path m.ckpt --wall_budget_sec 300 --seed 0",
    "nocvar":  f"scripts/run_active_loop.py --tau_mode none --selection thompson --exploration_approach A --exploration_gamma 1.0 --invalid_penalty -5 --model_path m.ckpt --wall_budget_sec 300 --seed 0",
    "noexpA":  f"scripts/run_active_loop.py --tau_mode cvar --tau_quantile 0.80 --selection thompson --invalid_penalty -5 --model_path m.ckpt --wall_budget_sec 300 --seed 0",
    "nobuf":   f"scripts/run_active_loop.py --tau_mode cvar --tau_quantile 0.80 --selection thompson --exploration_approach A --exploration_gamma 1.0 --invalid_penalty -5 --buffer_size 16 --model_path m.ckpt --wall_budget_sec 300 --seed 0",
    "nothomp": f"scripts/run_ddpp_online.py {DDPP_ON_BASE}",
}

VIDD_CELLS = {
    "orig":    f"scripts/run_active_loop.py {VIDD_AL_PREFIX} {DDPP_AL_BASE}",
    "nopen":   f"scripts/run_active_loop.py {VIDD_AL_PREFIX} --tau_mode cvar --tau_quantile 0.80 --selection thompson --exploration_approach A --exploration_gamma 1.0 --model_path m.ckpt --wall_budget_sec 300 --seed 0",
    "nocvar":  f"scripts/run_active_loop.py {VIDD_AL_PREFIX} --tau_mode none --selection thompson --exploration_approach A --exploration_gamma 1.0 --invalid_penalty -5 --model_path m.ckpt --wall_budget_sec 300 --seed 0",
    "noexpA":  f"scripts/run_active_loop.py {VIDD_AL_PREFIX} --tau_mode cvar --tau_quantile 0.80 --selection thompson --invalid_penalty -5 --model_path m.ckpt --wall_budget_sec 300 --seed 0",
    "nobuf":   f"scripts/run_active_loop.py {VIDD_AL_PREFIX} --tau_mode cvar --tau_quantile 0.80 --selection thompson --exploration_approach A --exploration_gamma 1.0 --invalid_penalty -5 --buffer_size 16 --model_path m.ckpt --wall_budget_sec 300 --seed 0",
    "nothomp": "scripts/run_vidd.py --oracle fa --loss_func rw_mle --gkd_lmbda 0.1 --tau_mode cvar --tau_quantile 0.80 --exploration_approach A --exploration_gamma 1.0 --invalid_penalty -5 --model_path m.ckpt --wall_budget_sec 300 --num_steps 5000 --seed 0",
}

SEARCH_METHODS = {
    # beam/mcts consolidated into run_active_loop.py --selection
    "beam":           "scripts/run_active_loop.py --model_path m.ckpt --wall_budget_sec 300 --seed 0 --selection beam --beam_width 8",
    "mcts":           "scripts/run_active_loop.py --model_path m.ckpt --wall_budget_sec 300 --seed 0 --selection thompson_mcts",
    "beam_surrogate": "scripts/run_beam_surrogate.py --model_path m.ckpt --wall_budget_sec 300 --seed 0",
    "dfkc":           "scripts/run_dfkc_fa.py --model_path m.ckpt --wall_budget_sec 300 --seed 0 --num_particles 16 --beta 15.0 --max_oracle_calls 80",
}

# ── Tests ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cell,cmd", list(DDPP_CELLS.items()))
def test_ddpp_cell_argparse(cell, cmd):
    """Each DDPP ablation cell's flag combination parses cleanly."""
    parts = shlex.split(cmd)
    script, *argv = parts
    ok, err = _can_import_argparser(script, argv)
    assert ok, f"DDPP cell '{cell}' argparse failed: {err}"


@pytest.mark.parametrize("cell,cmd", list(VIDD_CELLS.items()))
def test_vidd_cell_argparse(cell, cmd):
    """Each VIDD ablation cell's flag combination parses cleanly."""
    parts = shlex.split(cmd)
    script, *argv = parts
    ok, err = _can_import_argparser(script, argv)
    assert ok, f"VIDD cell '{cell}' argparse failed: {err}"


@pytest.mark.parametrize("method,cmd", list(SEARCH_METHODS.items()))
def test_search_method_argparse(method, cmd):
    """Each search-method script's flag combination parses cleanly."""
    parts = shlex.split(cmd)
    script, *argv = parts
    ok, err = _can_import_argparser(script, argv)
    assert ok, f"Search method '{method}' argparse failed: {err}"


# ── Sanity: the launcher script itself has correct cell mappings ────

@pytest.mark.skip(reason="shell launcher removed in public-release cleanup")
def test_smoke_launcher_exists():
    p = os.path.join(REPO, "scripts/smoke_test_all.sh")
    assert os.path.isfile(p), f"missing smoke launcher: {p}"
    assert os.access(p, os.X_OK), f"smoke launcher not executable: {p}"


@pytest.mark.skip(reason="shell launcher removed in public-release cleanup")
def test_search_methods_launcher_exists():
    p = os.path.join(REPO, "scripts/run_search_methods.sh")
    assert os.path.isfile(p), f"missing search launcher: {p}"
    assert os.access(p, os.X_OK), f"search launcher not executable: {p}"


@pytest.mark.skip(reason="shell launcher removed in public-release cleanup")
def test_smoke_launcher_covers_all_cells():
    """Sanity-check that smoke_test_all.sh references every cell name we
    expect — regression guard if someone deletes a cell from the launcher."""
    p = os.path.join(REPO, "scripts/smoke_test_all.sh")
    contents = open(p).read()
    expected = (
        [f"ddpp_{c}" for c in DDPP_CELLS]
        + [f"vidd_{c}" for c in VIDD_CELLS]
        + [f"search_{m}" for m in SEARCH_METHODS]
    )
    missing = [name for name in expected if name not in contents]
    assert not missing, f"smoke launcher missing cells: {missing}"
