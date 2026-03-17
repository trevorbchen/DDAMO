"""
GenMol Studio — Interactive molecular generation, visualization & evaluation.

Launch:  streamlit run app/app.py
"""

import os, sys

# Ensure repo root is on the path so `genmol` package resolves
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import warnings
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

from rdkit import Chem, RDLogger
from rdkit.Chem import (
    AllChem, Descriptors, Draw, QED, rdMolDescriptors, DataStructs,
)
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

RDLogger.DisableLog("rdApp.*")

# ─── Colour palette ──────────────────────────────────────────────────
ACCENT   = "#7C3AED"
BG_CARD  = "#1E1E2E"
BG_DARK  = "#11111B"
SUCCESS  = "#22C55E"
WARNING  = "#F59E0B"
DANGER   = "#EF4444"

# ─── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="GenMol Studio",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
/* Global overrides */
[data-testid="stSidebar"] {background: #1a1a2e;}
.stTabs [data-baseweb="tab-list"] {gap: 8px;}
.stTabs [data-baseweb="tab"] {
    background: #1E1E2E; border-radius: 8px; padding: 8px 20px;
    color: #ccc; border: 1px solid #333;
}
.stTabs [aria-selected="true"] {
    background: #7C3AED !important; color: white !important;
    border-color: #7C3AED !important;
}
div.mol-card {
    background: #1E1E2E; border-radius: 12px; padding: 12px;
    border: 1px solid #2a2a3e; margin-bottom: 8px;
    transition: border-color 0.2s;
}
div.mol-card:hover {border-color: #7C3AED;}
.metric-box {
    background: #1E1E2E; border-radius: 10px; padding: 16px 20px;
    border: 1px solid #2a2a3e; text-align: center;
}
.metric-box .value {font-size: 2rem; font-weight: 700; color: #7C3AED;}
.metric-box .label {font-size: 0.85rem; color: #888; margin-top: 4px;}
.hero-text {
    font-size: 0.95rem; color: #aaa; margin-bottom: 24px; line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# Chemistry helpers
# ═══════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def compute_properties(smiles_list: List[str]) -> pd.DataFrame:
    """Compute a rich set of molecular properties."""
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            rows.append({"smiles": smi, "valid": False})
            continue
        try:
            row = {
                "smiles": smi,
                "valid": True,
                "MW": Descriptors.MolWt(mol),
                "LogP": Descriptors.MolLogP(mol),
                "QED": QED.qed(mol),
                "TPSA": Descriptors.TPSA(mol),
                "HBA": rdMolDescriptors.CalcNumHBA(mol),
                "HBD": rdMolDescriptors.CalcNumHBD(mol),
                "RotBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
                "Rings": rdMolDescriptors.CalcNumRings(mol),
                "AromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
                "HeavyAtoms": mol.GetNumHeavyAtoms(),
                "SA": _sa_score(mol),
            }
            rows.append(row)
        except Exception:
            rows.append({"smiles": smi, "valid": False})
    return pd.DataFrame(rows)


def _sa_score(mol):
    """Synthetic accessibility score (1=easy, 10=hard)."""
    try:
        from rdkit.Contrib.SA_Score import sascorer
        return sascorer.calculateScore(mol)
    except Exception:
        return np.nan


def mol_to_svg(smi: str, size=(300, 250), highlight_smi: str = None) -> str:
    """Render a molecule to SVG string."""
    mol = Chem.MolFromSmiles(smi) if smi else None
    if mol is None:
        return _placeholder_svg(size)
    AllChem.Compute2DCoords(mol)

    highlight_atoms = []
    if highlight_smi:
        pat = Chem.MolFromSmarts(highlight_smi) or Chem.MolFromSmiles(highlight_smi)
        if pat:
            matches = mol.GetSubstructMatches(pat)
            if matches:
                highlight_atoms = list(matches[0])

    drawer = Draw.rdMolDraw2D.MolDraw2DSVG(*size)
    opts = drawer.drawOptions()
    opts.setBackgroundColour((0.118, 0.118, 0.173, 1.0))  # #1E1E2E
    opts.bondLineWidth = 2.0
    if highlight_atoms:
        colours = {a: (0.486, 0.227, 0.929, 0.35) for a in highlight_atoms}
        drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms,
                            highlightAtomColors=colours)
    else:
        drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def _placeholder_svg(size):
    w, h = size
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">
    <rect width="{w}" height="{h}" fill="#1E1E2E" rx="8"/>
    <text x="50%" y="50%" fill="#555" font-size="14"
          text-anchor="middle" dominant-baseline="middle">Invalid molecule</text></svg>"""


def svg_to_html(svg: str, caption: str = "", width: str = "100%") -> str:
    return f"""<div class="mol-card">
        <div style="display:flex;justify-content:center;">{svg}</div>
        <div style="text-align:center;color:#888;font-size:0.78rem;
                    margin-top:6px;word-break:break-all;">{caption}</div>
    </div>"""


@st.cache_data(show_spinner=False)
def compute_fingerprints(smiles_list):
    fps = []
    valid = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
            arr = np.zeros(1024, dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
            valid.append(smi)
    return np.array(fps) if fps else np.empty((0, 1024)), valid


@st.cache_data(show_spinner=False)
def internal_diversity(smiles_list):
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024))
    if len(fps) < 2:
        return 0.0
    sims = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    return 1.0 - np.mean(sims)


def scaffold_counts(smiles_list):
    scaffolds = {}
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            continue
        try:
            sc = Chem.MolToSmiles(GetScaffoldForMol(mol))
            scaffolds[sc] = scaffolds.get(sc, 0) + 1
        except Exception:
            pass
    return scaffolds


# ═══════════════════════════════════════════════════════════════════════
# Model loading  (cached across sessions)
# ═══════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading GenMol model …")
def load_sampler(path: str):
    from genmol.sampler import Sampler
    return Sampler(path)


@st.cache_resource(show_spinner="Loading DAPS sampler …")
def load_daps_sampler(path: str, reward_name: str, **kwargs):
    from genmol.DAPS_sampler import DAPSSampler
    from genmol.rewards import get_reward
    forward_op = get_reward(reward_name)
    return DAPSSampler(path, forward_op=forward_op, **kwargs)


@st.cache_resource(show_spinner="Loading Beam Search sampler …")
def load_beam_sampler(path: str, reward_name: str = "QED", **kwargs):
    from genmol.beam_search_sampler import BeamSearchSampler
    from genmol.rewards import get_reward
    forward_op = get_reward(reward_name)
    return BeamSearchSampler(path, forward_op=forward_op, **kwargs)


@st.cache_resource(show_spinner="Loading MCTS sampler …")
def load_mcts_sampler(path: str, reward_name: str = "QED", **kwargs):
    from genmol.mcts_sampler import MCTSSampler
    from genmol.rewards import get_reward
    forward_op = get_reward(reward_name)
    return MCTSSampler(path, forward_op=forward_op, **kwargs)


@st.cache_resource(show_spinner="Loading DFKC sampler …")
def load_dfkc_sampler(path: str, reward_name: str = "none", **kwargs):
    from genmol.smc_sampler import DFKCSampler
    from genmol.rewards import get_reward
    forward_op = get_reward(reward_name)
    return DFKCSampler(path, forward_op=forward_op, **kwargs)


@st.cache_resource(show_spinner="Loading SMC sampler …")
def load_smc_sampler(path: str, reward_name: str = "QED", **kwargs):
    from genmol.smc_sampler import SMCSampler
    from genmol.rewards import get_reward
    forward_op = get_reward(reward_name)
    return SMCSampler(path, forward_op=forward_op, **kwargs)


# ═══════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════

def _default_slider_ranges():
    """Default min/max/step for each slider parameter."""
    return {
        "samples":      {"min": 1,    "max": 2000,  "step": 5,    "default": 50},
        "softmax_temp": {"min": 0.1,  "max": 3.0,   "step": 0.1,  "default": 1.2},
        "randomness":   {"min": 0.0,  "max": 5.0,   "step": 0.1,  "default": 2.0},
        "gamma":        {"min": 0.0,  "max": 1.0,   "step": 0.05, "default": 0.0},
        "num_steps":    {"min": 5,    "max": 100,   "step": 1,    "default": 50},
        "mh_steps":     {"min": 0,    "max": 10,    "step": 1,    "default": 2},
        "alpha":        {"min": 0.0,  "max": 500.0, "step": 10.0, "default": 100.0},
        "ode_steps":    {"min": 5,    "max": 200,   "step": 5,    "default": 20},
        "beam_width":   {"min": 2,    "max": 50,    "step": 1,    "default": 8},
        "branching":    {"min": 2,    "max": 16,    "step": 1,    "default": 4},
        "steps_per_interval": {"min": 1, "max": 50, "step": 1,   "default": 5},
        "c_uct":        {"min": 0.1,  "max": 5.0,   "step": 0.1,  "default": 1.0},
        "diversity_penalty": {"min": 0.0, "max": 1.0, "step": 0.05, "default": 0.0},
    }


def _get_slider_ranges():
    """Return slider ranges, merging user overrides from session state."""
    ranges = _default_slider_ranges()
    overrides = st.session_state.get("slider_ranges", {})
    for key in ranges:
        if key in overrides:
            ranges[key].update(overrides[key])
    return ranges


def render_sidebar():
    with st.sidebar:
        st.markdown("## 🧬 GenMol Studio")
        st.markdown(
            '<p class="hero-text">Interactive discrete diffusion for '
            "molecular generation, visualisation & evaluation.</p>",
            unsafe_allow_html=True,
        )
        st.divider()

        # Auto-discover checkpoint files in the workspace
        import glob as _glob
        _ckpt_files = sorted(_glob.glob("**/*.ckpt", recursive=True))
        if _ckpt_files:
            model_path = st.selectbox(
                "Model checkpoint",
                _ckpt_files,
                help="Discovered .ckpt files in workspace",
            )
        else:
            st.warning("No .ckpt files found in workspace. Enter path manually.")
            model_path = st.text_input(
                "Model checkpoint",
                value="model.ckpt",
                help="Path to GenMol .ckpt file",
            )

        # Load suggested hparams for the selected checkpoint
        _HPARAMS = {
            "model.ckpt":    "scripts/exps/denovo/hparams.yaml",
            "model_v2.ckpt": "scripts/exps/denovo/hparams_v2.yaml",
        }
        hp = {}
        _hp_file = _HPARAMS.get(os.path.basename(model_path))
        if _hp_file and os.path.exists(_hp_file):
            try:
                import yaml as _yaml
                with open(_hp_file) as _f:
                    hp = _yaml.safe_load(_f) or {}
            except Exception:
                pass

        sampler_type = st.selectbox(
            "Sampler", ["Standard", "DAPS", "Beam Search", "MCTS", "DFKC", "SMC"],
            help="Standard = confidence denoising · DAPS = annealing + MH · "
                 "Beam Search = branch & prune · MCTS = tree search · "
                 "DFKC = Feynman-Kac corrector (SMC) · SMC = vanilla particle filter",
        )

        st.divider()
        st.markdown("### Generation parameters")
        if hp:
            st.caption(f"Defaults from `{os.path.basename(_hp_file)}`")

        sr = _get_slider_ranges()

        task = st.selectbox(
            "Task",
            ["De novo", "Fragment linking", "Fragment linking (1‑step)",
             "Motif extension", "Scaffold decoration",
             "Superstructure", "Mask modification"],
        )

        def _sync(src, dst):
            st.session_state[dst] = st.session_state[src]

        def _synced_param(label, lo, hi, default, step, name, fmt=None):
            """Render a slider + number_input that stay in sync."""
            sk, nk = f"{name}_slider", f"{name}_num"
            # Initialise from default on first render
            if sk not in st.session_state:
                st.session_state[sk] = default
            if nk not in st.session_state:
                st.session_state[nk] = default
            _s, _n = st.columns([3, 1])
            kw = dict(format=fmt) if fmt else {}
            with _s:
                st.slider(label, lo, hi, step=step,
                          key=sk, on_change=_sync, args=(sk, nk), **kw)
            with _n:
                st.number_input(label, min_value=lo, max_value=hi, step=step,
                                key=nk, on_change=_sync, args=(nk, sk),
                                label_visibility="collapsed", **kw)
            return st.session_state[sk]

        num_samples = _synced_param("Samples",
            sr["samples"]["min"], sr["samples"]["max"],
            hp.get("num_samples", sr["samples"]["default"]),
            sr["samples"]["step"], "samples")
        softmax_temp = _synced_param("Softmax temperature",
            sr["softmax_temp"]["min"], sr["softmax_temp"]["max"],
            hp.get("softmax_temp", sr["softmax_temp"]["default"]),
            sr["softmax_temp"]["step"], "temp", fmt="%.2f")
        randomness = _synced_param("Randomness",
            sr["randomness"]["min"], sr["randomness"]["max"],
            hp.get("randomness", sr["randomness"]["default"]),
            sr["randomness"]["step"], "rand", fmt="%.2f")
        min_add_len = _synced_param("Min sequence length",
            10, 120, hp.get("min_add_len", 40), 5, "minlen")
        gamma = _synced_param("MCG γ (context guidance)",
            sr["gamma"]["min"], sr["gamma"]["max"],
            sr["gamma"]["default"],
            sr["gamma"]["step"], "gamma", fmt="%.2f")

        fragment_input = ""
        if task not in ("De novo",):
            fragment_input = st.text_area(
                "Fragment / SMILES input",
                placeholder="e.g. c1ccc(*)cc1 or CC(=O)O.c1ccccc1",
                height=80,
            )

        daps_kwargs = {}
        beam_kwargs = {}
        mcts_kwargs = {}
        dfkc_kwargs = {}
        smc_kwargs = {}

        if sampler_type == "DAPS":
            st.divider()
            st.markdown("### DAPS parameters")
            daps_kwargs["reward"] = st.selectbox(
                "Reward function",
                ["None", "MW", "QED", "LogP", "TPSA"],
                help="Property to optimise during MH refinement",
            )
            daps_kwargs["num_steps"] = _synced_param("Annealing steps",
                sr["num_steps"]["min"], sr["num_steps"]["max"],
                sr["num_steps"]["default"], sr["num_steps"]["step"],
                "daps_nsteps")
            daps_kwargs["mh_steps"] = _synced_param("MH steps / anneal step",
                sr["mh_steps"]["min"], sr["mh_steps"]["max"],
                sr["mh_steps"]["default"], sr["mh_steps"]["step"],
                "daps_mh")
            daps_kwargs["alpha"] = _synced_param("α (reward weight)",
                sr["alpha"]["min"], sr["alpha"]["max"],
                sr["alpha"]["default"], sr["alpha"]["step"],
                "daps_alpha", fmt="%.1f")
            daps_kwargs["ode_steps"] = _synced_param("ODE sub-steps",
                sr["ode_steps"]["min"], sr["ode_steps"]["max"],
                sr["ode_steps"]["default"], sr["ode_steps"]["step"],
                "daps_ode")
            daps_kwargs["mutate_strategy"] = st.selectbox(
                "Proposal strategy",
                ["infill", "fragment"],
                help="infill = mask & model re-fill (recommended) · "
                     "fragment = SAFE fragment swap (legacy)",
            )
            if daps_kwargs["mutate_strategy"] == "infill":
                daps_kwargs["proposal_mask_frac"] = _synced_param(
                    "Proposal mask fraction", 0.05, 0.5, 0.1, 0.05,
                    "daps_maskfrac", fmt="%.2f")

        elif sampler_type == "Beam Search":
            st.divider()
            st.markdown("### Beam Search parameters")
            beam_kwargs["reward"] = st.selectbox(
                "Reward function", ["QED", "MW", "LogP", "TPSA"],
                help="Objective for beam scoring",
            )
            beam_kwargs["beam_width"] = _synced_param("Beam width (N)",
                sr["beam_width"]["min"], sr["beam_width"]["max"],
                sr["beam_width"]["default"], sr["beam_width"]["step"],
                "beam_width")
            beam_kwargs["branching_factor"] = _synced_param("Branching factor (L)",
                sr["branching"]["min"], sr["branching"]["max"],
                sr["branching"]["default"], sr["branching"]["step"],
                "beam_branch")
            beam_kwargs["steps_per_interval"] = _synced_param("Steps per interval (K)",
                sr["steps_per_interval"]["min"], sr["steps_per_interval"]["max"],
                sr["steps_per_interval"]["default"], sr["steps_per_interval"]["step"],
                "beam_spi")
            beam_kwargs["diversity_penalty"] = _synced_param("Diversity penalty (λ)",
                sr["diversity_penalty"]["min"], sr["diversity_penalty"]["max"],
                sr["diversity_penalty"]["default"], sr["diversity_penalty"]["step"],
                "beam_dp", fmt="%.2f")

        elif sampler_type == "MCTS":
            st.divider()
            st.markdown("### MCTS parameters")
            mcts_kwargs["reward"] = st.selectbox(
                "Reward function", ["QED", "MW", "LogP", "TPSA"],
                help="Objective for rollout scoring",
            )
            mcts_kwargs["branching_factor"] = _synced_param("Branching factor (L)",
                sr["branching"]["min"], sr["branching"]["max"],
                sr["branching"]["default"], sr["branching"]["step"],
                "mcts_branch")
            mcts_kwargs["steps_per_interval"] = _synced_param("Steps per interval (K)",
                sr["steps_per_interval"]["min"], sr["steps_per_interval"]["max"],
                sr["steps_per_interval"]["default"], sr["steps_per_interval"]["step"],
                "mcts_spi")
            mcts_kwargs["c_uct"] = _synced_param("UCB exploration (c)",
                sr["c_uct"]["min"], sr["c_uct"]["max"],
                sr["c_uct"]["default"], sr["c_uct"]["step"],
                "mcts_cuct", fmt="%.1f")

        elif sampler_type == "DFKC":
            st.divider()
            st.markdown("### DFKC parameters")
            dfkc_kwargs["reward"] = st.selectbox(
                "Reward function", ["none", "QED", "MW", "LogP", "TPSA"],
                help="Reward for weight computation (none = annealing only)",
            )
            dfkc_kwargs["mode"] = st.selectbox(
                "Mode", ["annealing", "reward"],
                help="annealing = sharpen model distribution · "
                     "reward = tilt toward high-reward molecules",
            )
            dfkc_kwargs["num_particles"] = _synced_param(
                "Particles (K)", 2, 32, 8, 1, "dfkc_particles")
            dfkc_kwargs["beta"] = _synced_param(
                "Beta (sharpening)", 1.0, 10.0, 2.0, 0.1, "dfkc_beta", fmt="%.1f")
            dfkc_kwargs["beta_schedule"] = st.selectbox(
                "Beta schedule", ["linear", "constant", "cosine"],
                help="How beta ramps from 1 to target over denoising steps",
            )
            dfkc_kwargs["ess_threshold"] = _synced_param(
                "ESS threshold", 0.1, 1.0, 0.5, 0.1, "dfkc_ess", fmt="%.1f")

        elif sampler_type == "SMC":
            st.divider()
            st.markdown("### SMC parameters")
            smc_kwargs["reward"] = st.selectbox(
                "Reward function", ["QED", "MW", "LogP", "TPSA"],
                help="Objective for resampling weights",
            )
            smc_kwargs["num_particles"] = _synced_param(
                "Particles (K)", 2, 32, 8, 1, "smc_particles")
            smc_kwargs["alpha"] = _synced_param(
                "Alpha (reward weight)", 1.0, 100.0, 10.0, 1.0,
                "smc_alpha", fmt="%.0f")
            smc_kwargs["resample_interval"] = _synced_param(
                "Resample interval (steps)", 1, 20, 5, 1, "smc_ri")
            smc_kwargs["resample_start"] = _synced_param(
                "Resample start (fraction)", 0.0, 1.0, 0.5, 0.1,
                "smc_start", fmt="%.1f")
            smc_kwargs["ess_threshold"] = _synced_param(
                "ESS threshold", 0.1, 1.0, 0.5, 0.1, "smc_ess", fmt="%.1f")

        st.divider()
        st.markdown("### Visualisation")
        highlight_smi = st.text_input(
            "Highlight substructure (SMARTS/SMILES)", value="",
            help="Atoms matching this pattern will be highlighted in molecule cards",
        )

        # ─── Backend settings (slider ranges) ────────────────────
        st.divider()
        with st.expander("⚙️ Backend settings"):
            st.caption("Set min / max limits for each parameter slider.")
            defaults = _default_slider_ranges()
            overrides = st.session_state.get("slider_ranges", {})

            new_overrides = {}
            for key, dflt in defaults.items():
                label = key.replace("_", " ").title()
                c1, c2 = st.columns(2)
                cur_min = overrides.get(key, {}).get("min", dflt["min"])
                cur_max = overrides.get(key, {}).get("max", dflt["max"])
                new_min = c1.number_input(f"{label} min", value=float(cur_min),
                                          step=float(dflt["step"]), key=f"be_{key}_min")
                new_max = c2.number_input(f"{label} max", value=float(cur_max),
                                          step=float(dflt["step"]), key=f"be_{key}_max")
                if new_min != dflt["min"] or new_max != dflt["max"]:
                    new_overrides[key] = {"min": type(dflt["min"])(new_min),
                                          "max": type(dflt["max"])(new_max)}

            if st.button("Apply ranges", key="apply_ranges"):
                st.session_state["slider_ranges"] = new_overrides
                st.rerun()

        return {
            "model_path": model_path,
            "sampler_type": sampler_type,
            "task": task,
            "num_samples": num_samples,
            "softmax_temp": softmax_temp,
            "randomness": randomness,
            "min_add_len": min_add_len,
            "gamma": gamma,
            "fragment": fragment_input.strip(),
            "daps_kwargs": daps_kwargs,
            "beam_kwargs": beam_kwargs,
            "mcts_kwargs": mcts_kwargs,
            "dfkc_kwargs": dfkc_kwargs,
            "smc_kwargs": smc_kwargs,
            "highlight": highlight_smi.strip(),
        }


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _auto_run_name(gen_cfg: dict, n_total: int) -> str:
    """Build a descriptive run name from sampler type + key params."""
    stype = gen_cfg.get("sampler_type", "Standard")
    parts = []

    if stype == "Standard":
        parts.append("uncond")
    elif stype == "DAPS":
        dk = gen_cfg.get("daps_kwargs", {})
        parts.append("daps")
        reward = dk.get("reward", "None")
        if reward and reward != "None":
            parts.append(reward.lower())
        parts.append(f"s{dk.get('num_steps', 50)}")
        parts.append(f"mh{dk.get('mh_steps', 2)}")
        parts.append(f"a{dk.get('alpha', 100)}")
    elif stype == "Beam Search":
        bk = gen_cfg.get("beam_kwargs", {})
        parts.append("beam")
        parts.append(f"N{bk.get('beam_width', 4)}")
        parts.append(f"L{bk.get('branching_factor', 3)}")
        parts.append(f"K{bk.get('steps_per_interval', 5)}")
        dp = bk.get("diversity_penalty", 0)
        if dp:
            parts.append(f"dp{dp}")
    elif stype == "MCTS":
        mk = gen_cfg.get("mcts_kwargs", {})
        parts.append("mcts")
        parts.append(f"L{mk.get('branching_factor', 4)}")
        parts.append(f"K{mk.get('steps_per_interval', 5)}")
        parts.append(f"c{mk.get('c_uct', 1.0)}")

    temp = gen_cfg.get("softmax_temp", 1.0)
    parts.append(f"t{temp}")
    parts.append(f"n{n_total}")
    return "_".join(str(p) for p in parts)


def _save_run_to_disk(run_name, smiles, gen_cfg, elapsed, props_df):
    """Persist a run to the results directory (samples.csv + metrics.json + config.yaml)."""
    import json
    try:
        import yaml
    except ImportError:
        yaml = None

    run_dir = os.path.join(RESULTS_DIR, "app_runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    # samples.csv
    smi_series = pd.Series(smiles, name="smiles")
    smi_series.to_frame().to_csv(os.path.join(run_dir, "samples.csv"), index=False)

    # Compute QED stats from props_df
    valid_qed = props_df.loc[props_df["valid"] == True, "QED"].dropna().sort_values(ascending=False)
    qed_mean = float(valid_qed.mean()) if len(valid_qed) else 0.0
    top10_n = max(1, len(valid_qed) // 10)
    qed_top10 = float(valid_qed.iloc[:top10_n].mean()) if len(valid_qed) else 0.0
    qed_max = float(valid_qed.iloc[0]) if len(valid_qed) else 0.0

    n_valid = int((props_df["valid"] == True).sum())
    n_unique = int(props_df["smiles"].nunique())
    n_total = len(smiles)

    # Determine method name
    stype = gen_cfg.get("sampler_type", "Standard")
    method_map = {"Standard": "uncond", "DAPS": "daps", "Beam Search": "beam", "MCTS": "mcts"}

    metrics = {
        "name": run_name,
        "reward": "none",
        "elapsed_sec": elapsed,
        "validity": n_valid / max(n_total, 1),
        "uniqueness": n_unique / max(n_total, 1),
        "qed_mean": qed_mean,
        "qed_top10": qed_top10,
        "qed_max": qed_max,
        "num_samples": n_total,
        "budget_per_sample": 0,
        "total_reward_evals": 0,
        "forward_passes": 0,
        "fp_per_sample": 0,
        "softmax_temp": gen_cfg.get("softmax_temp"),
        "randomness": gen_cfg.get("randomness"),
    }
    # Add sampler-specific params
    extra_keys = {
        "DAPS": ("daps_kwargs", ["num_steps", "alpha", "mh_steps", "ode_steps",
                                  "mutate_strategy", "proposal_mask_frac"]),
        "Beam Search": ("beam_kwargs", ["beam_width", "branching_factor",
                                         "steps_per_interval", "diversity_penalty"]),
        "MCTS": ("mcts_kwargs", ["branching_factor", "steps_per_interval", "c_uct"]),
        "DFKC": ("dfkc_kwargs", ["num_particles", "mode", "beta", "beta_schedule",
                                  "ess_threshold"]),
        "SMC": ("smc_kwargs", ["num_particles", "resample_interval", "resample_start",
                                "alpha", "ess_threshold"]),
    }
    if stype in extra_keys:
        kw_key, param_names = extra_keys[stype]
        kw = gen_cfg.get(kw_key, {})
        for k in param_names:
            if k in kw:
                metrics[k] = kw[k]
        reward = kw.get("reward", "None")
        if reward and reward != "None":
            metrics["reward"] = reward.lower()

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # config.yaml
    if yaml:
        cfg_out = {
            "name": run_name,
            "sampler": {"_target_": f"genmol.{method_map.get(stype, 'sampler')}_sampler"},
            "softmax_temp": gen_cfg.get("softmax_temp"),
            "randomness": gen_cfg.get("randomness"),
            "min_add_len": gen_cfg.get("min_add_len"),
            "num_samples": n_total,
        }
        if stype in extra_keys:
            kw_key, param_names = extra_keys[stype]
            kw = gen_cfg.get(kw_key, {})
            for k in param_names:
                if k in kw:
                    cfg_out["sampler"][k] = kw[k]
        with open(os.path.join(run_dir, "config.yaml"), "w") as f:
            yaml.dump(cfg_out, f, default_flow_style=False)

    st.success(f"Saved **{run_name}** ({n_total} molecules)")


# ═══════════════════════════════════════════════════════════════════════
# Generation
# ═══════════════════════════════════════════════════════════════════════

def _get_sampler(cfg: dict):
    """Return the appropriate sampler instance."""
    path = cfg["model_path"]
    stype = cfg["sampler_type"]

    if stype == "DAPS":
        dk = cfg["daps_kwargs"]
        return load_daps_sampler(
            path,
            reward_name=dk.get("reward", "None"),
            num_steps=dk.get("num_steps", 50),
            mh_steps=dk.get("mh_steps", 2),
            alpha=dk.get("alpha", 100.0),
            ode_steps=dk.get("ode_steps", 20),
            mutate_strategy=dk.get("mutate_strategy", "infill"),
            proposal_mask_frac=dk.get("proposal_mask_frac", 0.1),
        )
    elif stype == "Beam Search":
        bk = cfg["beam_kwargs"]
        return load_beam_sampler(
            path,
            reward_name=bk.get("reward", "QED"),
            beam_width=bk.get("beam_width", 8),
            branching_factor=bk.get("branching_factor", 4),
            steps_per_interval=bk.get("steps_per_interval", 5),
            diversity_penalty=bk.get("diversity_penalty", 0.0),
        )
    elif stype == "MCTS":
        mk = cfg["mcts_kwargs"]
        return load_mcts_sampler(
            path,
            reward_name=mk.get("reward", "QED"),
            branching_factor=mk.get("branching_factor", 4),
            steps_per_interval=mk.get("steps_per_interval", 5),
            c_uct=mk.get("c_uct", 1.0),
        )
    elif stype == "DFKC":
        fk = cfg.get("dfkc_kwargs", {})
        return load_dfkc_sampler(
            path,
            reward_name=fk.get("reward", "none"),
            num_particles=fk.get("num_particles", 8),
            mode=fk.get("mode", "annealing"),
            beta=fk.get("beta", 2.0),
            beta_schedule=fk.get("beta_schedule", "linear"),
            ess_threshold=fk.get("ess_threshold", 0.5),
        )
    elif stype == "SMC":
        sk = cfg.get("smc_kwargs", {})
        return load_smc_sampler(
            path,
            reward_name=sk.get("reward", "QED"),
            num_particles=sk.get("num_particles", 8),
            resample_interval=sk.get("resample_interval", 5),
            resample_start=sk.get("resample_start", 0.5),
            alpha=sk.get("alpha", 10.0),
            ess_threshold=sk.get("ess_threshold", 0.5),
        )
    return load_sampler(path)


def _convert_safe_to_smiles(safe_strings: List[str]) -> List[str]:
    """Convert SAFE strings to SMILES (for DAPS output)."""
    from genmol.utils.utils_chem import safe_to_smiles
    from genmol.utils.bracket_safe_converter import bracketsafe2safe

    smiles = []
    for s in safe_strings:
        if not s:
            continue
        smi = safe_to_smiles(s, fix=True)
        if not smi:
            try:
                smi = safe_to_smiles(bracketsafe2safe(s), fix=True)
            except Exception:
                smi = None
        if smi:
            # Keep the largest fragment (same logic as DAPSSampler._decode)
            smiles.append(sorted(smi.split("."), key=len)[-1])
    return smiles


def run_generation(cfg: dict, progress_cb=None) -> List[str]:
    """Dispatch to the appropriate sampler method and return SMILES list.

    Args:
        progress_cb: Optional callable(fraction, status_text) for progress updates.
    """
    task = cfg["task"]
    n = cfg["num_samples"]
    frag = cfg["fragment"]
    kwargs = dict(
        softmax_temp=cfg["softmax_temp"],
        randomness=cfg["randomness"],
        min_add_len=cfg.get("min_add_len", 40),
        gamma=cfg["gamma"],
    )
    sampler = _get_sampler(cfg)
    stype = cfg["sampler_type"]

    if progress_cb:
        progress_cb(0.0, "Loading sampler …")

    if task == "Mask modification":
        # Per-molecule loop — report progress for each
        results = []
        for i in range(n):
            if progress_cb:
                progress_cb((i / n), f"Generating molecule {i+1}/{n} …")
            r = sampler.mask_modification(frag, **kwargs)
            if r:
                results.append(r)
    else:
        if progress_cb:
            progress_cb(0.05, f"Running {task} ({n} samples) …")

        if task == "De novo":
            results = sampler.de_novo_generation(n, **kwargs)
        elif task == "Fragment linking":
            results = sampler.fragment_linking(frag, n, **kwargs)
        elif task == "Fragment linking (1\u2011step)":
            results = sampler.fragment_linking_onestep(frag, n, **kwargs)
        elif task in ("Motif extension", "Scaffold decoration"):
            results = sampler.fragment_completion(frag, n, **kwargs)
        elif task == "Superstructure":
            results = sampler.fragment_completion(frag, n, apply_filter=False, **kwargs)
        else:
            results = []

    # DAPS returns SAFE strings — convert to SMILES in the outer loop
    if stype == "DAPS" and results:
        if progress_cb:
            progress_cb(0.9, "Converting SAFE → SMILES …")
        results = _convert_safe_to_smiles(results)

    if progress_cb:
        progress_cb(1.0, "Done!")

    return results


# ═══════════════════════════════════════════════════════════════════════
# Tab: Generate
# ═══════════════════════════════════════════════════════════════════════

def tab_generate(cfg):
    st.markdown("### 🚀 Generate molecules")
    info_parts = [
        f"**{cfg['sampler_type']}** sampler",
        f"**{cfg['task']}**",
        f"**{cfg['num_samples']}** samples",
        f"temp={cfg['softmax_temp']}",
        f"rand={cfg['randomness']}",
    ]
    if cfg["sampler_type"] == "Beam Search":
        bk = cfg["beam_kwargs"]
        info_parts.append(f"N={bk.get('beam_width',8)} L={bk.get('branching_factor',4)}")
    elif cfg["sampler_type"] == "MCTS":
        mk = cfg["mcts_kwargs"]
        info_parts.append(f"L={mk.get('branching_factor',4)} c={mk.get('c_uct',1.0)}")
    st.markdown(" · ".join(info_parts))

    if cfg["task"] not in ("De novo",) and not cfg["fragment"]:
        st.warning("Enter a fragment / SMILES in the sidebar to continue.")
        return

    col_btn, col_status = st.columns([1, 3])
    with col_btn:
        go_btn = st.button("⚡ Generate", type="primary", use_container_width=True)

    if go_btn:
        progress_bar = st.progress(0, text="Starting …")
        def _progress(frac, text):
            progress_bar.progress(min(frac, 1.0), text=text)

        t0 = time.time()
        try:
            smiles = run_generation(cfg, progress_cb=_progress)
        except FileNotFoundError as e:
            progress_bar.empty()
            st.error(f"File not found — make sure `model.ckpt` and `data/len.pk` exist.\n\n`{e}`")
            return
        except Exception as e:
            progress_bar.empty()
            st.error(f"Generation error: {e}")
            return
        elapsed = time.time() - t0
        progress_bar.empty()

        st.session_state["generated_smiles"] = smiles
        st.session_state["gen_time"] = elapsed
        st.session_state["gen_cfg"] = cfg.copy()

    # Show results if available
    smiles = st.session_state.get("generated_smiles")
    if not smiles:
        st.info("Configure parameters in the sidebar and click **Generate**.")
        return

    elapsed = st.session_state.get("gen_time", 0)
    df = compute_properties(smiles)
    valid_df = df[df["valid"] == True]

    # ─── Metric cards ─────────────────────────────────────────────
    n_total = len(smiles)
    n_valid = len(valid_df)
    n_unique = valid_df["smiles"].nunique() if len(valid_df) else 0
    diversity = internal_diversity(valid_df["smiles"].tolist()) if len(valid_df) > 1 else 0

    cols = st.columns(5)
    metrics = [
        ("Generated", n_total, ""),
        ("Valid", f"{n_valid}/{n_total}", f"{100*n_valid/max(n_total,1):.0f}%"),
        ("Unique", n_unique, f"{100*n_unique/max(n_valid,1):.0f}%"),
        ("Diversity", f"{diversity:.3f}", ""),
        ("Time", f"{elapsed:.1f}s", f"{n_total/max(elapsed,0.01):.0f}/s"),
    ]
    for col, (label, value, sub) in zip(cols, metrics):
        col.markdown(
            f'<div class="metric-box">'
            f'<div class="value">{value}</div>'
            f'<div class="label">{label} {sub}</div></div>',
            unsafe_allow_html=True,
        )

    # ─── Save run ─────────────────────────────────────────────────
    st.divider()
    save_col1, save_col2 = st.columns([3, 1])
    with save_col1:
        gen_cfg = st.session_state.get("gen_cfg", cfg)
        default_name = _auto_run_name(gen_cfg, n_total)
        run_name = st.text_input("Run name", value=default_name,
                                 key="run_name_input",
                                 help="Name this run to save it for comparison")
    with save_col2:
        st.markdown("<br/>", unsafe_allow_html=True)  # vertical alignment
        if st.button("💾 Save run", use_container_width=True):
            if "saved_runs" not in st.session_state:
                st.session_state["saved_runs"] = {}
            st.session_state["saved_runs"][run_name] = {
                "smiles": list(smiles),
                "cfg": {k: v for k, v in gen_cfg.items()
                        if k != "daps_kwargs" or isinstance(v, (str, int, float, bool, dict))},
                "time": elapsed,
            }
            # Persist to results directory so it appears in Results Explorer
            _save_run_to_disk(run_name, smiles, gen_cfg, elapsed, df)
            _load_all_metrics.clear()

    # Show saved runs count
    saved = st.session_state.get("saved_runs", {})
    if saved:
        st.caption(f"{len(saved)} saved run(s): {', '.join(saved.keys())}")

    st.divider()

    # ─── Molecule grid ────────────────────────────────────────────
    st.markdown("#### Molecule gallery")
    page_size = st.select_slider("Molecules per page", [12, 24, 48, 96], value=24)
    total_pages = max(1, (len(valid_df) + page_size - 1) // page_size)
    page = st.number_input("Page", 1, total_pages, 1) - 1
    page_df = valid_df.iloc[page * page_size : (page + 1) * page_size]

    grid_cols = st.columns(4)
    for i, (_, row) in enumerate(page_df.iterrows()):
        svg = mol_to_svg(row["smiles"], (280, 220), cfg.get("highlight"))
        caption = (
            f"{row['smiles'][:45]}{'…' if len(str(row['smiles']))>45 else ''}"
            f"<br/><span style='color:#7C3AED'>MW {row.get('MW',0):.0f}</span>"
            f" · QED {row.get('QED',0):.2f}"
            f" · LogP {row.get('LogP',0):.1f}"
        )
        grid_cols[i % 4].markdown(svg_to_html(svg, caption), unsafe_allow_html=True)

    # ─── Download ─────────────────────────────────────────────────
    st.divider()
    csv = valid_df.to_csv(index=False)
    st.download_button(
        "📥 Download results CSV",
        data=csv,
        file_name="genmol_generated.csv",
        mime="text/csv",
    )


# ═══════════════════════════════════════════════════════════════════════
# Tab: Visualize
# ═══════════════════════════════════════════════════════════════════════

def tab_visualize(cfg):
    st.markdown("### 🔬 Molecular Visualisation")

    smiles = st.session_state.get("generated_smiles")
    if not smiles:
        st.info("Generate molecules first, or paste SMILES below.")

    custom = st.text_area(
        "Paste SMILES (one per line) to visualise",
        height=100,
        placeholder="CCO\nc1ccccc1\n...",
    )
    if custom.strip():
        smiles = [s.strip() for s in custom.strip().splitlines() if s.strip()]

    if not smiles:
        return

    df = compute_properties(smiles)
    valid_df = df[df["valid"] == True].copy()

    if valid_df.empty:
        st.warning("No valid molecules to visualise.")
        return

    # ─── Featured molecule ────────────────────────────────────────
    st.markdown("#### Molecule Inspector")
    selected_idx = st.selectbox(
        "Select molecule",
        range(len(valid_df)),
        format_func=lambda i: f"#{i+1}  {valid_df.iloc[i]['smiles'][:60]}",
    )
    row = valid_df.iloc[selected_idx]

    col_mol, col_props = st.columns([1, 1])
    with col_mol:
        svg = mol_to_svg(row["smiles"], (450, 380), cfg.get("highlight"))
        st.markdown(
            f'<div class="mol-card" style="padding:20px">{svg}'
            f'<div style="text-align:center;color:#aaa;font-size:0.82rem;'
            f'margin-top:10px;word-break:break-all;">{row["smiles"]}</div></div>',
            unsafe_allow_html=True,
        )

    with col_props:
        # Radar chart of normalised properties
        props = {
            "QED": row.get("QED", 0),
            "SA (norm)": 1 - (row.get("SA", 5) - 1) / 9,  # invert: 1=easy
            "LogP (norm)": np.clip(row.get("LogP", 0) / 5, 0, 1),
            "MW (norm)": np.clip(row.get("MW", 0) / 500, 0, 1),
            "TPSA (norm)": np.clip(row.get("TPSA", 0) / 140, 0, 1),
            "HBD/5": np.clip(row.get("HBD", 0) / 5, 0, 1),
        }
        cats = list(props.keys())
        vals = list(props.values())
        fig_radar = go.Figure(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=cats + [cats[0]],
            fill="toself",
            fillcolor="rgba(124,58,237,0.25)",
            line=dict(color=ACCENT, width=2),
            marker=dict(size=6, color=ACCENT),
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="#1E1E2E",
                radialaxis=dict(visible=True, range=[0, 1], showticklabels=False,
                                gridcolor="#333"),
                angularaxis=dict(gridcolor="#333", color="#aaa"),
            ),
            paper_bgcolor="#11111B",
            font=dict(color="#ccc"),
            margin=dict(l=60, r=60, t=30, b=30),
            height=380,
            showlegend=False,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Property table
        prop_cols = ["MW", "LogP", "QED", "TPSA", "HBA", "HBD",
                     "RotBonds", "Rings", "AromaticRings", "HeavyAtoms", "SA"]
        prop_vals = {k: [row.get(k, "")] for k in prop_cols}
        st.dataframe(
            pd.DataFrame(prop_vals),
            hide_index=True,
            use_container_width=True,
        )

    # ─── Lipinski / Drug-likeness check ───────────────────────────
    st.markdown("#### Drug-likeness Rules")
    rules = {
        "Lipinski MW ≤ 500": row.get("MW", 0) <= 500,
        "Lipinski LogP ≤ 5": row.get("LogP", 0) <= 5,
        "Lipinski HBD ≤ 5": row.get("HBD", 0) <= 5,
        "Lipinski HBA ≤ 10": row.get("HBA", 0) <= 10,
        "Veber RotBonds ≤ 10": row.get("RotBonds", 0) <= 10,
        "Veber TPSA ≤ 140": row.get("TPSA", 0) <= 140,
    }
    rule_cols = st.columns(len(rules))
    for col, (name, passed) in zip(rule_cols, rules.items()):
        icon = "✅" if passed else "❌"
        col.markdown(f"**{icon} {name}**")


# ═══════════════════════════════════════════════════════════════════════
# Tab: Evaluate
# ═══════════════════════════════════════════════════════════════════════

def tab_evaluate(cfg):
    st.markdown("### 📊 Evaluation Dashboard")

    smiles = st.session_state.get("generated_smiles")
    if not smiles:
        st.info("Generate molecules first to see evaluation metrics.")
        return

    df = compute_properties(smiles)
    valid_df = df[df["valid"] == True].copy()
    if valid_df.empty:
        st.warning("No valid molecules to evaluate.")
        return

    # ─── Summary metrics ──────────────────────────────────────────
    n_total = len(smiles)
    n_valid = len(valid_df)
    n_unique = valid_df["smiles"].nunique()
    diversity = internal_diversity(valid_df["smiles"].tolist())
    mean_qed = valid_df["QED"].mean()
    mean_sa = valid_df["SA"].mean()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    for col, (lbl, val) in zip(
        [c1, c2, c3, c4, c5, c6],
        [
            ("Validity", f"{100*n_valid/max(n_total,1):.1f}%"),
            ("Uniqueness", f"{100*n_unique/max(n_valid,1):.1f}%"),
            ("Diversity", f"{diversity:.3f}"),
            ("Mean QED", f"{mean_qed:.3f}"),
            ("Mean SA", f"{mean_sa:.2f}"),
            ("Drug-like", f"{100*len(valid_df[valid_df['QED']>=0.6])/max(n_valid,1):.0f}%"),
        ],
    ):
        col.markdown(
            f'<div class="metric-box"><div class="value">{val}</div>'
            f'<div class="label">{lbl}</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ─── Property distributions ───────────────────────────────────
    st.markdown("#### Property Distributions")
    dist_props = ["MW", "LogP", "QED", "TPSA", "SA", "HeavyAtoms"]
    fig = make_subplots(rows=2, cols=3, subplot_titles=dist_props)
    for i, prop in enumerate(dist_props):
        r, c = i // 3 + 1, i % 3 + 1
        vals = valid_df[prop].dropna()
        fig.add_trace(
            go.Histogram(
                x=vals, nbinsx=30,
                marker_color=ACCENT, opacity=0.85,
                name=prop, showlegend=False,
            ),
            row=r, col=c,
        )
    fig.update_layout(
        height=500,
        paper_bgcolor="#11111B",
        plot_bgcolor="#1E1E2E",
        font=dict(color="#ccc"),
        margin=dict(l=40, r=20, t=40, b=30),
    )
    fig.update_xaxes(gridcolor="#333")
    fig.update_yaxes(gridcolor="#333")
    st.plotly_chart(fig, use_container_width=True)

    # ─── Property correlations ────────────────────────────────────
    st.markdown("#### Property Correlations")
    cor_x = st.selectbox("X axis", dist_props, index=0, key="cor_x")
    cor_y = st.selectbox("Y axis", dist_props, index=2, key="cor_y")
    fig_scatter = px.scatter(
        valid_df, x=cor_x, y=cor_y,
        color="QED", color_continuous_scale="Viridis",
        hover_data=["smiles"],
        opacity=0.7,
    )
    fig_scatter.update_layout(
        height=450,
        paper_bgcolor="#11111B",
        plot_bgcolor="#1E1E2E",
        font=dict(color="#ccc"),
    )
    fig_scatter.update_xaxes(gridcolor="#333")
    fig_scatter.update_yaxes(gridcolor="#333")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ─── Chemical space (PCA) ─────────────────────────────────────
    st.markdown("#### Chemical Space (PCA of Morgan fingerprints)")
    fp_mat, fp_smiles = compute_fingerprints(valid_df["smiles"].tolist())
    if len(fp_mat) >= 3:
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2).fit_transform(fp_mat.astype(float))
        cs_df = pd.DataFrame({
            "PC1": coords[:, 0], "PC2": coords[:, 1],
            "smiles": fp_smiles,
        })
        # merge QED for colouring
        cs_df = cs_df.merge(valid_df[["smiles", "QED"]], on="smiles", how="left")
        fig_cs = px.scatter(
            cs_df, x="PC1", y="PC2",
            color="QED", color_continuous_scale="Plasma",
            hover_data=["smiles"],
            opacity=0.75,
        )
        fig_cs.update_layout(
            height=500,
            paper_bgcolor="#11111B",
            plot_bgcolor="#1E1E2E",
            font=dict(color="#ccc"),
        )
        fig_cs.update_xaxes(gridcolor="#333")
        fig_cs.update_yaxes(gridcolor="#333")
        st.plotly_chart(fig_cs, use_container_width=True)
    else:
        st.info("Need ≥ 3 valid molecules for PCA.")

    # ─── Scaffold analysis ────────────────────────────────────────
    st.markdown("#### Top Scaffolds")
    sc = scaffold_counts(valid_df["smiles"].tolist())
    if sc:
        sc_df = pd.DataFrame(
            sorted(sc.items(), key=lambda x: -x[1])[:12],
            columns=["scaffold", "count"],
        )
        fig_sc = px.bar(
            sc_df, x="scaffold", y="count",
            color_discrete_sequence=[ACCENT],
        )
        fig_sc.update_layout(
            height=350,
            paper_bgcolor="#11111B",
            plot_bgcolor="#1E1E2E",
            font=dict(color="#ccc"),
            xaxis_tickangle=-45,
        )
        fig_sc.update_xaxes(gridcolor="#333")
        fig_sc.update_yaxes(gridcolor="#333")
        st.plotly_chart(fig_sc, use_container_width=True)

        # Render top scaffolds as molecules
        top_scaffolds = sc_df["scaffold"].tolist()[:8]
        scaf_cols = st.columns(4)
        for i, sc_smi in enumerate(top_scaffolds):
            svg = mol_to_svg(sc_smi, (200, 160))
            scaf_cols[i % 4].markdown(
                svg_to_html(svg, f"<b>{sc.get(sc_smi, 0)}×</b> {sc_smi[:30]}"),
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════
# Tab: Compare
# ═══════════════════════════════════════════════════════════════════════

def tab_compare(cfg):
    st.markdown("### ⚖️ Compare Runs")
    st.markdown(
        "Compare saved runs, uploaded CSVs, or the current session. "
        "Save runs from the **Generate** tab using the 💾 button."
    )

    uploaded = st.file_uploader(
        "Upload CSV files (must have a `smiles` column)",
        type=["csv"],
        accept_multiple_files=True,
    )

    # Gather datasets: saved runs + disk results + current session + uploaded files
    datasets = {}

    # Saved runs from session
    saved_runs = st.session_state.get("saved_runs", {})
    if saved_runs:
        st.markdown(f"**Session runs** ({len(saved_runs)}):")
        selected_runs = st.multiselect(
            "Select session runs to compare",
            list(saved_runs.keys()),
            default=list(saved_runs.keys()),
        )
        for name in selected_runs:
            datasets[name] = saved_runs[name]["smiles"]

        if st.button("🗑️ Clear all saved runs", key="clear_saved"):
            st.session_state["saved_runs"] = {}
            st.rerun()

    # On-disk results from Results Explorer directory
    disk_runs = {}
    for root, _dirs, files in os.walk(RESULTS_DIR):
        if "samples.csv" in files:
            name = os.path.basename(root)
            disk_runs[name] = os.path.join(root, "samples.csv")
    if disk_runs:
        all_names = sorted(disk_runs.keys())
        search = st.text_input(
            "Search experiments", placeholder="e.g. beam, mcts, L4, t08...",
            key="cmp_search",
        )
        if search:
            terms = search.lower().split()
            filtered_names = [
                n for n in all_names
                if all(t in n.lower() for t in terms)
            ]
        else:
            filtered_names = all_names
        st.caption(f"{len(filtered_names)} / {len(all_names)} experiments match")
        selected_disk = st.multiselect(
            "Select experiments to compare",
            filtered_names,
            key="cmp_disk_runs",
        )
        for name in selected_disk:
            try:
                ddf = pd.read_csv(disk_runs[name])
                col = "smiles" if "smiles" in ddf.columns else ddf.columns[0]
                datasets[name] = ddf[col].dropna().tolist()
            except Exception:
                pass

    # Current session (only if not already saved)
    current = st.session_state.get("generated_smiles")
    if current and "Current session" not in datasets:
        datasets["Current session"] = current

    for f in (uploaded or []):
        try:
            udf = pd.read_csv(f)
            col = "smiles" if "smiles" in udf.columns else udf.columns[0]
            datasets[f.name] = udf[col].dropna().tolist()
        except Exception as e:
            st.warning(f"Could not read {f.name}: {e}")

    if len(datasets) < 1:
        st.info("Save a run from the Generate tab, upload a CSV, or generate molecules to compare.")
        return

    # Build comparison table
    rows = []
    all_props = {}
    for name, smi_list in datasets.items():
        df = compute_properties(smi_list)
        vdf = df[df["valid"] == True]
        n_total = len(smi_list)
        n_valid = len(vdf)
        n_unique = vdf["smiles"].nunique() if len(vdf) else 0
        div = internal_diversity(vdf["smiles"].tolist()) if len(vdf) > 1 else 0
        rows.append({
            "Dataset": name,
            "Total": n_total,
            "Valid (%)": f"{100*n_valid/max(n_total,1):.1f}",
            "Unique (%)": f"{100*n_unique/max(n_valid,1):.1f}",
            "Diversity": f"{div:.3f}",
            "Mean MW": f"{vdf['MW'].mean():.1f}" if len(vdf) and "MW" in vdf else "—",
            "Mean QED": f"{vdf['QED'].mean():.3f}" if len(vdf) and "QED" in vdf else "—",
            "Mean LogP": f"{vdf['LogP'].mean():.2f}" if len(vdf) and "LogP" in vdf else "—",
            "Mean SA": f"{vdf['SA'].mean():.2f}" if len(vdf) and "SA" in vdf else "—",
        })
        all_props[name] = vdf

    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # ─── Overlaid distributions ───────────────────────────────────
    if len(all_props) >= 1:
        st.markdown("#### Distribution Overlay")
        prop = st.selectbox("Property", ["MW", "LogP", "QED", "TPSA", "SA"], key="cmp_prop")
        fig = go.Figure()
        colours = px.colors.qualitative.Set2
        for i, (name, vdf) in enumerate(all_props.items()):
            if prop not in vdf.columns or vdf[prop].dropna().empty:
                continue
            fig.add_trace(go.Histogram(
                x=vdf[prop].dropna(), nbinsx=30,
                name=name, opacity=0.6,
                marker_color=colours[i % len(colours)],
            ))
        fig.update_layout(
            barmode="overlay",
            height=400,
            paper_bgcolor="#11111B",
            plot_bgcolor="#1E1E2E",
            font=dict(color="#ccc"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        fig.update_xaxes(gridcolor="#333")
        fig.update_yaxes(gridcolor="#333")
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# Tab: Results Explorer
# ═══════════════════════════════════════════════════════════════════════

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts", "exps", "denovo", "outputs", "results",
)


@st.cache_data(show_spinner=False)
def _load_all_metrics():
    """Walk the results directory, load metrics.json + config.yaml for each run."""
    import json
    try:
        import yaml
    except ImportError:
        yaml = None

    rows = []
    for root, _dirs, files in os.walk(RESULTS_DIR):
        if "metrics.json" not in files:
            continue
        try:
            with open(os.path.join(root, "metrics.json")) as f:
                m = json.load(f)
        except Exception:
            continue

        m["_dir"] = root

        # Infer method from name
        name = m.get("name", "")
        if name.startswith("beam"):
            m["method"] = "Beam Search"
        elif name.startswith("mcts"):
            m["method"] = "MCTS"
        elif name.startswith("daps"):
            m["method"] = "DAPS"
        else:
            m["method"] = "Standard"

        # Beam Search / MCTS default to QED internally when no reward is set
        if m.get("reward") in ("none", None) and m["method"] in ("Beam Search", "MCTS"):
            m["reward"] = "qed"

        # Extract structured params from config.yaml
        cfg_path = os.path.join(root, "config.yaml")
        if yaml and os.path.exists(cfg_path):
            try:
                with open(cfg_path) as f:
                    cfg = yaml.safe_load(f)
                sampler_cfg = cfg.get("sampler", {})
                # Common params
                for k in ("softmax_temp", "randomness", "min_add_len", "seed"):
                    if k in cfg and k not in m:
                        m[k] = cfg[k]
                # Sampler-specific params
                for k in (
                    "beam_width", "branching_factor", "steps_per_interval",
                    "diversity_penalty", "diversity_cutoff", "elite_buffer_size",
                    "c_uct", "rollout_budget_per_sample",
                    "num_steps", "alpha", "mh_steps", "ode_steps",
                    "mutate_strategy", "proposal_mask_frac",
                ):
                    if k in sampler_cfg and k not in m:
                        m[k] = sampler_cfg[k]
            except Exception:
                pass

        rows.append(m)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def tab_results(cfg):
    st.markdown("### 🏆 Experiment Results Explorer")
    st.markdown(
        "Browse pre-computed results from the hyperparameter sweep "
        "(219 runs, 8153 molecules)."
    )

    df = _load_all_metrics()
    if df.empty:
        st.warning(
            f"No results found. Expected metrics at `{RESULTS_DIR}`."
        )
        return

    # ─── Method filter (pill buttons) ─────────────────────────────
    methods = sorted(df["method"].unique())
    sel_methods = st.pills(
        "Method", methods, selection_mode="multi", default=methods,
        key="results_methods",
    )
    if not sel_methods:
        sel_methods = methods

    # ─── Numeric filters ──────────────────────────────────────────
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        budgets = sorted(df["budget_per_sample"].dropna().unique())
        if budgets:
            budget_range = st.slider(
                "Budget / sample",
                float(min(budgets)), float(max(budgets)),
                (float(min(budgets)), float(max(budgets))),
            )
        else:
            budget_range = (0, 1e9)
    with col_f2:
        min_qed = st.slider("Min mean QED", 0.0, 1.0, 0.0, 0.01)

    filtered = df[
        (df["method"].isin(sel_methods))
        & (df["budget_per_sample"].between(*budget_range))
        & (df["qed_mean"].ge(min_qed))
    ].copy()

    st.caption(f"Showing {len(filtered)} / {len(df)} runs")

    # ─── Column toggles (grouped checkboxes) ──────────────────────
    _COL_GROUPS = {
        "Quality": ["qed_mean", "qed_top10", "qed_max", "validity", "uniqueness"],
        "Cost": ["budget_per_sample", "elapsed_sec", "forward_passes",
                 "fp_per_sample", "total_reward_evals"],
        "Beam": ["beam_width", "branching_factor", "steps_per_interval",
                 "diversity_penalty", "diversity_cutoff", "elite_buffer_size"],
        "MCTS": ["c_uct", "rollout_budget_per_sample",
                 "branching_factor", "steps_per_interval"],
        "DAPS": ["num_steps", "alpha", "mh_steps", "ode_steps",
                 "mutate_strategy", "proposal_mask_frac"],
        "General": ["softmax_temp", "randomness", "num_samples"],
    }
    # Always-on columns
    always_cols = [c for c in ["name", "method", "reward"] if c in filtered.columns]

    with st.expander("Table columns", expanded=False):
        group_cols = st.columns(len(_COL_GROUPS))
        enabled_cols = list(always_cols)
        for (group_name, cols), gcol in zip(_COL_GROUPS.items(), group_cols):
            with gcol:
                st.caption(group_name)
                for c in cols:
                    if c not in filtered.columns:
                        continue
                    default_on = c in (
                        "qed_mean", "qed_top10", "qed_max",
                        "validity", "uniqueness",
                        "budget_per_sample", "elapsed_sec",
                    )
                    if st.checkbox(c, value=default_on, key=f"col_{group_name}_{c}"):
                        if c not in enabled_cols:
                            enabled_cols.append(c)
    show_cols = enabled_cols if len(enabled_cols) > len(always_cols) else (
        always_cols + ["budget_per_sample", "elapsed_sec",
                       "validity", "uniqueness", "qed_mean", "qed_top10", "qed_max"]
    )

    # Pick sort column
    sortable = [c for c in show_cols if c not in ("name", "method", "reward", "mutate_strategy")]
    sort_col = st.selectbox(
        "Sort by", sortable,
        index=sortable.index("qed_mean") if "qed_mean" in sortable else 0,
        key="results_sort",
    )

    display_df = (
        filtered[show_cols]
        .sort_values(sort_col, ascending=False)
        .reset_index(drop=True)
    )

    # Select all / unselect all
    sa_col1, sa_col2, _ = st.columns([1, 1, 6])
    with sa_col1:
        if st.button("Select all", key="res_sel_all", use_container_width=True):
            st.session_state["_res_select_all"] = True
            st.rerun()
    with sa_col2:
        if st.button("Clear all", key="res_clr_all", use_container_width=True):
            st.session_state["_res_select_all"] = False
            st.rerun()

    select_all = st.session_state.pop("_res_select_all", False)
    display_df.insert(0, "✓", select_all)

    edited = st.data_editor(
        display_df,
        use_container_width=True,
        height=400,
        key="results_editor",
        column_config={"✓": st.column_config.CheckboxColumn(
            "✓", width=40, help="Select for Compare",
        )},
        disabled=[c for c in display_df.columns if c != "✓"],
    )

    checked_names = edited.loc[edited["✓"], "name"].tolist() if "✓" in edited.columns else []
    if checked_names:
        if st.button(f"⚖️ Send {len(checked_names)} selected to Compare", key="send_to_compare"):
            # Load SMILES for checked runs and store in session saved_runs
            if "saved_runs" not in st.session_state:
                st.session_state["saved_runs"] = {}
            for rname in checked_names:
                rrow = filtered[filtered["name"] == rname]
                if rrow.empty:
                    continue
                rdir = rrow.iloc[0]["_dir"]
                csv_p = os.path.join(rdir, "samples.csv")
                if os.path.exists(csv_p):
                    try:
                        rdf = pd.read_csv(csv_p)
                        col = "smiles" if "smiles" in rdf.columns else rdf.columns[0]
                        st.session_state["saved_runs"][rname] = {
                            "smiles": rdf[col].dropna().tolist(),
                            "cfg": {}, "time": 0,
                        }
                    except Exception:
                        pass
            st.success(f"Sent **{len(checked_names)}** runs to Compare tab")

    if len(filtered) < 2:
        return

    # ─── Configurable scatter plot ────────────────────────────────
    st.markdown("#### Metric Explorer")
    numeric_cols = [
        c for c in filtered.columns
        if filtered[c].dtype in ("float64", "int64", "float32", "int32")
        and c not in ("_dir",)
    ]
    col_x, col_y = st.columns(2)
    with col_x:
        x_axis = st.selectbox(
            "X axis", numeric_cols,
            index=numeric_cols.index("budget_per_sample") if "budget_per_sample" in numeric_cols else 0,
            key="res_x",
        )
    with col_y:
        y_axis = st.selectbox(
            "Y axis", numeric_cols,
            index=numeric_cols.index("qed_mean") if "qed_mean" in numeric_cols else 0,
            key="res_y",
        )

    colours_map = {"Beam Search": "#7C3AED", "MCTS": "#22C55E", "Standard": "#F59E0B",
                   "DFKC": "#EC4899", "SMC": "#06B6D4"}
    fig = go.Figure()
    for method in filtered["method"].unique():
        mdf = filtered[filtered["method"] == method]
        fig.add_trace(go.Scatter(
            x=mdf[x_axis], y=mdf[y_axis],
            mode="markers", name=method,
            marker=dict(size=8, color=colours_map.get(method, "#888")),
            text=mdf["name"],
            hovertemplate="%{text}<br>" + x_axis + ": %{x:.3f}<br>" + y_axis + ": %{y:.3f}<extra></extra>",
        ))
    fig.update_layout(
        xaxis_title=x_axis,
        yaxis_title=y_axis,
        height=450,
        paper_bgcolor="#11111B", plot_bgcolor="#1E1E2E",
        font=dict(color="#ccc"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="#333")
    fig.update_yaxes(gridcolor="#333")
    st.plotly_chart(fig, use_container_width=True)

    # ─── Load & inspect a single run ──────────────────────────────
    st.markdown("#### Inspect Run")
    run_names = filtered.sort_values("qed_mean", ascending=False)["name"].tolist()
    selected_run = st.selectbox("Select run", run_names)
    if selected_run:
        run_row = filtered[filtered["name"] == selected_run].iloc[0]
        run_dir = run_row["_dir"]
        csv_path = os.path.join(run_dir, "samples.csv")
        if os.path.exists(csv_path):
            run_df = pd.read_csv(csv_path)
            smi_col = "smiles" if "smiles" in run_df.columns else run_df.columns[0]
            smiles = run_df[smi_col].dropna().tolist()

            # Show top molecules
            props = compute_properties(smiles)
            valid = props[props["valid"] == True].sort_values("QED", ascending=False)
            st.caption(f"{len(valid)} valid molecules from {selected_run}")

            top = valid.head(8)
            grid_cols = st.columns(4)
            for i, (_, row) in enumerate(top.iterrows()):
                svg = mol_to_svg(row["smiles"], (240, 190))
                cap = (
                    f"{row['smiles'][:40]}"
                    f"<br><span style='color:#7C3AED'>QED {row.get('QED',0):.3f}</span>"
                    f" · MW {row.get('MW',0):.0f}"
                )
                grid_cols[i % 4].markdown(
                    svg_to_html(svg, cap), unsafe_allow_html=True,
                )
        else:
            st.info("No samples.csv found for this run.")


# ═══════════════════════════════════════════════════════════════════════
# Sweep tab
# ═══════════════════════════════════════════════════════════════════════

# type: "int" = discrete integers, "float" = real-valued (supports lin/log range),
#       "choice" = literal string options
_SWEEP_PARAMS = {
    "Beam Search": {
        "sampler.beam_width":          {"label": "Beam width (N)",          "default": "4, 8, 16",  "type": "int"},
        "sampler.branching_factor":    {"label": "Branching factor (L)",    "default": "2, 4, 8",   "type": "int"},
        "sampler.steps_per_interval":  {"label": "Steps per interval (K)", "default": "5",          "type": "int"},
        "sampler.diversity_penalty":   {"label": "Diversity penalty (λ)",  "default": "0.0",        "type": "float"},
    },
    "MCTS": {
        "sampler.branching_factor":    {"label": "Branching factor (L)",    "default": "2, 4, 8",   "type": "int"},
        "sampler.steps_per_interval":  {"label": "Steps per interval (K)", "default": "5",          "type": "int"},
        "sampler.c_uct":              {"label": "UCB exploration (c)",     "default": "0.5, 1.0, 2.0", "type": "float"},
    },
    "DAPS": {
        "sampler.num_steps":           {"label": "Annealing steps",        "default": "50",         "type": "int"},
        "sampler.alpha":               {"label": "α (reward weight)",      "default": "100",        "type": "float"},
        "sampler.mh_steps":            {"label": "MH steps",               "default": "2",          "type": "int"},
        "sampler.ode_steps":           {"label": "ODE sub-steps",          "default": "20",         "type": "int"},
        "sampler.mutate_strategy":     {"label": "Proposal strategy",      "default": "infill",     "type": "choice",
                                        "options": ["infill", "fragment"]},
        "sampler.proposal_mask_frac":  {"label": "Mask fraction",          "default": "0.1",        "type": "float"},
    },
    "DFKC": {
        "sampler.num_particles":    {"label": "Particles (K)",         "default": "4, 8, 16",      "type": "int"},
        "sampler.beta":             {"label": "Beta (sharpening)",     "default": "1.5, 2.0, 3.0", "type": "float"},
        "sampler.mode":             {"label": "Mode",                  "default": "annealing",      "type": "choice",
                                     "options": ["annealing", "reward"]},
        "sampler.beta_schedule":    {"label": "Beta schedule",         "default": "linear",         "type": "choice",
                                     "options": ["linear", "constant", "cosine"]},
    },
    "SMC": {
        "sampler.num_particles":    {"label": "Particles (K)",         "default": "4, 8, 16",      "type": "int"},
        "sampler.alpha":            {"label": "Alpha (reward weight)", "default": "5, 10, 20",      "type": "float"},
        "sampler.resample_interval":{"label": "Resample interval",     "default": "5",              "type": "int"},
        "sampler.resample_start":   {"label": "Resample start frac",   "default": "0.5",            "type": "float"},
    },
    "Standard": {},
}


def _generate_range(lo: float, hi: float, n: int, scale: str, dtype: str) -> list:
    """Generate n values between lo and hi on linear or log scale."""
    if n <= 1:
        return [lo]
    if scale == "log":
        import math
        lo_log = math.log10(max(lo, 1e-8))
        hi_log = math.log10(max(hi, 1e-8))
        vals = [10 ** (lo_log + i * (hi_log - lo_log) / (n - 1)) for i in range(n)]
    else:
        vals = [lo + i * (hi - lo) / (n - 1) for i in range(n)]
    if dtype == "int":
        vals = sorted(set(int(round(v)) for v in vals))
    else:
        vals = [round(v, 4) for v in vals]
    return vals


def _parse_values(text: str):
    """Parse comma-separated values into a list of floats/ints."""
    vals = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = float(tok)
            vals.append(int(v) if v == int(v) else v)
        except ValueError:
            vals.append(tok)
    return vals


def _sweep_analysis(sweep_dir: str, swept_params: list[str]):
    """Render sweep analysis visualizations."""
    import json as _json

    # Load completed metrics (walk subdirs to handle reward_name/ nesting)
    rows = []
    for dirpath, _dirnames, filenames in os.walk(sweep_dir):
        if "metrics.json" not in filenames:
            continue
        mj = os.path.join(dirpath, "metrics.json")
        try:
            with open(mj) as f:
                m = _json.load(f)
            m["_dir"] = dirpath
            rows.append(m)
        except Exception:
            continue
    if len(rows) < 2:
        return
    df = pd.DataFrame(rows)

    st.markdown("---")
    st.markdown("### Sweep Analysis")

    metric_options = [c for c in ["qed_mean", "qed_top10", "qed_max",
                                   "validity", "uniqueness"] if c in df.columns]
    target = st.selectbox("Target metric", metric_options,
                          index=0, key="sweep_target")

    # ── Best configs ──────────────────────────────────────────────
    st.markdown("#### Best Configurations")
    show_cols = ["name"] + [c for c in swept_params if c in df.columns] + [
        target, "elapsed_sec",
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    st.dataframe(
        df.sort_values(target, ascending=False).head(10)[show_cols].reset_index(drop=True),
        use_container_width=True,
    )

    # Identify which swept params actually vary in df
    varying = []
    for p in swept_params:
        col = p.split(".")[-1]  # sampler.beam_width -> beam_width
        if col in df.columns and df[col].nunique() > 1:
            varying.append(col)

    if not varying:
        return

    # ── Parameter importance ──────────────────────────────────────
    st.markdown("#### Parameter Importance")
    total_var = df[target].var()
    importances = {}
    for p in varying:
        group_means = df.groupby(p)[target].mean()
        between_var = group_means.var()
        importances[p] = between_var / total_var if total_var > 0 else 0

    imp_df = pd.DataFrame({
        "Parameter": list(importances.keys()),
        "Importance": list(importances.values()),
    }).sort_values("Importance", ascending=True)
    fig = px.bar(imp_df, x="Importance", y="Parameter", orientation="h",
                 color="Importance", color_continuous_scale="Viridis")
    fig.update_layout(height=max(200, 40 * len(varying)),
                      paper_bgcolor="#11111B", plot_bgcolor="#1E1E2E",
                      font=dict(color="#ccc"), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # ── Heatmap (2D interaction) ──────────────────────────────────
    if len(varying) >= 2:
        st.markdown("#### Parameter Interaction Heatmap")
        hc1, hc2 = st.columns(2)
        with hc1:
            hx = st.selectbox("X axis", varying, index=0, key="sweep_hx")
        with hc2:
            hy = st.selectbox("Y axis", varying,
                              index=min(1, len(varying) - 1), key="sweep_hy")
        if hx != hy:
            pivot = df.pivot_table(index=hy, columns=hx, values=target,
                                   aggfunc="mean")
            fig = px.imshow(pivot, text_auto=".3f", aspect="auto",
                            color_continuous_scale="Viridis",
                            labels=dict(color=target))
            fig.update_layout(paper_bgcolor="#11111B",
                              font=dict(color="#ccc"), height=400)
            st.plotly_chart(fig, use_container_width=True)

    # ── Parallel coordinates ──────────────────────────────────────
    st.markdown("#### Parallel Coordinates")
    par_cols = varying + [target]
    par_df = df[par_cols].dropna()
    if len(par_df) >= 2:
        dims = []
        for c in varying:
            dims.append(dict(label=c, values=par_df[c]))
        dims.append(dict(label=target, values=par_df[target]))
        fig = go.Figure(data=go.Parcoords(
            line=dict(
                color=par_df[target],
                colorscale="Viridis",
                showscale=True,
                cmin=par_df[target].min(),
                cmax=par_df[target].max(),
            ),
            dimensions=dims,
        ))
        fig.update_layout(paper_bgcolor="#11111B", font=dict(color="#ccc"),
                          height=450)
        st.plotly_chart(fig, use_container_width=True)

    # ── Pareto frontier ───────────────────────────────────────────
    st.markdown("#### Pareto Frontier")
    pc1, pc2 = st.columns(2)
    cost_cols = [c for c in ["elapsed_sec", "budget_per_sample", "fp_per_sample",
                              "forward_passes"] if c in df.columns]
    with pc1:
        pareto_x = st.selectbox("Cost (X)", cost_cols,
                                index=0, key="sweep_pareto_x")
    with pc2:
        pareto_y = st.selectbox("Quality (Y)", metric_options,
                                index=0, key="sweep_pareto_y")

    pdf = df[[pareto_x, pareto_y, "name"]].dropna()
    if len(pdf) >= 2:
        # Compute Pareto front (maximize Y, minimize X)
        pdf_sorted = pdf.sort_values(pareto_x)
        pareto = []
        best_y = -float("inf")
        for _, row in pdf_sorted.iterrows():
            if row[pareto_y] > best_y:
                pareto.append(row)
                best_y = row[pareto_y]
        pareto_df = pd.DataFrame(pareto)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pdf[pareto_x], y=pdf[pareto_y], mode="markers",
            marker=dict(size=8, color="#7C3AED"),
            text=pdf["name"],
            hovertemplate="%{text}<br>" + pareto_x + ": %{x:.2f}<br>" + pareto_y + ": %{y:.4f}<extra></extra>",
            name="All runs",
        ))
        fig.add_trace(go.Scatter(
            x=pareto_df[pareto_x], y=pareto_df[pareto_y],
            mode="lines+markers",
            marker=dict(size=12, color="#22C55E", symbol="star"),
            line=dict(color="#22C55E", width=2),
            text=pareto_df["name"],
            hovertemplate="%{text}<br>" + pareto_x + ": %{x:.2f}<br>" + pareto_y + ": %{y:.4f}<extra></extra>",
            name="Pareto optimal",
        ))
        fig.update_layout(
            xaxis_title=pareto_x, yaxis_title=pareto_y,
            height=400, paper_bgcolor="#11111B", plot_bgcolor="#1E1E2E",
            font=dict(color="#ccc"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        fig.update_xaxes(gridcolor="#333")
        fig.update_yaxes(gridcolor="#333")
        st.plotly_chart(fig, use_container_width=True)


def tab_sweep(cfg):
    from sweep_runner import SweepRunner, build_grid, build_random, detect_gpus

    st.markdown("### 🔬 Hyperparameter Sweep")
    st.markdown(
        "Define parameter grids, launch parallel runs across GPUs, "
        "and analyze results interactively."
    )

    # ── Sweep configuration ───────────────────────────────────────
    sc1, sc2 = st.columns([2, 1])
    with sc1:
        sweep_sampler = st.selectbox(
            "Sampler", ["Beam Search", "MCTS", "DAPS", "Standard"],
            key="sweep_sampler",
        )
    with sc2:
        n_gpus = detect_gpus()
        gpu_count = st.number_input(
            "GPUs", min_value=1, max_value=max(n_gpus, 8),
            value=max(1, n_gpus), key="sweep_gpus",
        )

    rc1, rc2 = st.columns([2, 1])
    with rc1:
        sweep_reward = st.selectbox(
            "Reward", ["none", "qed", "mw", "logp", "tpsa"],
            help="none = QED default for beam/mcts · qed/mw/logp/tpsa = explicit reward",
            key="sweep_reward",
        )
    with rc2:
        sweep_type = st.pills("Sweep type", ["Grid Search", "Random Search"],
                              default="Grid Search", key="sweep_type")

    if sweep_type == "Random Search":
        random_n = st.number_input("Random samples", min_value=1, max_value=1000,
                                   value=10, key="sweep_random_n")

    # Common params
    st.markdown("##### Common parameters")
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        temp_str = st.text_input("softmax_temp", value="0.5, 0.8",
                                 key="sweep_temp")
    with cc2:
        rand_str = st.text_input("randomness", value="0.5",
                                 key="sweep_rand")
    with cc3:
        ns_str = st.text_input("num_samples", value="50", key="sweep_ns")

    # Sampler-specific params
    sampler_params = _SWEEP_PARAMS.get(sweep_sampler, {})
    param_inputs = {}
    if sampler_params:
        st.markdown(f"##### {sweep_sampler} parameters")
        for key, info in sampler_params.items():
            ptype = info.get("type", "float")

            if ptype == "choice":
                # Literal options as multiselect pills
                options = info.get("options", [info["default"]])
                selected = st.pills(
                    info["label"], options, selection_mode="multi",
                    default=[info["default"]], key=f"sweep_p_{key}",
                )
                param_inputs[key] = ", ".join(selected) if selected else info["default"]

            elif ptype in ("int", "float"):
                # Two modes: list values, or generate a range
                mode = st.radio(
                    info["label"], ["List values", "Range"],
                    horizontal=True, key=f"sweep_m_{key}",
                    label_visibility="visible",
                )
                if mode == "List values":
                    param_inputs[key] = st.text_input(
                        f"{info['label']} values", value=info["default"],
                        key=f"sweep_p_{key}", label_visibility="collapsed",
                    )
                else:
                    rc1, rc2, rc3, rc4 = st.columns(4)
                    default_vals = _parse_values(info["default"])
                    lo_def = float(min(default_vals)) if default_vals else 1.0
                    hi_def = float(max(default_vals)) if default_vals else 10.0
                    if lo_def == hi_def:
                        hi_def = lo_def * 2 if lo_def > 0 else lo_def + 10
                    with rc1:
                        lo = st.number_input("Min", value=lo_def, key=f"sweep_lo_{key}",
                                             label_visibility="visible")
                    with rc2:
                        hi = st.number_input("Max", value=hi_def, key=f"sweep_hi_{key}",
                                             label_visibility="visible")
                    with rc3:
                        n_pts = st.number_input("N", min_value=1, max_value=50,
                                                value=3, key=f"sweep_n_{key}",
                                                label_visibility="visible")
                    with rc4:
                        scale = st.selectbox("Scale", ["linear", "log"],
                                             key=f"sweep_sc_{key}",
                                             label_visibility="visible")
                    generated = _generate_range(lo, hi, n_pts, scale, ptype)
                    param_inputs[key] = ", ".join(str(v) for v in generated)
                    st.caption(f"→ {param_inputs[key]}")
            else:
                param_inputs[key] = st.text_input(
                    info["label"], value=info["default"],
                    key=f"sweep_p_{key}",
                )

    # ── Build grid ────────────────────────────────────────────────
    param_ranges = {}
    for text, hydra_key in [
        (temp_str, "softmax_temp"),
        (rand_str, "randomness"),
        (ns_str, "num_samples"),
    ]:
        vals = _parse_values(text)
        if vals and len(vals) > 0:
            param_ranges[hydra_key] = vals

    if sampler_params:
        for key, info in sampler_params.items():
            vals = _parse_values(param_inputs.get(key, info["default"]))
            if vals:
                param_ranges[key] = vals

    if sweep_type == "Random Search":
        configs = build_random(sweep_sampler, param_ranges, random_n)
    else:
        configs = build_grid(sweep_sampler, param_ranges)

    # ── Preview ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"#### Preview: {len(configs)} configurations")

    preview_rows = []
    for c in configs:
        row = {"name": c["name"]}
        row.update(c["overrides"])
        preview_rows.append(row)
    preview_df = pd.DataFrame(preview_rows)
    st.dataframe(preview_df, use_container_width=True,
                 height=min(400, 40 + 35 * len(configs)))

    # ── Execution ─────────────────────────────────────────────────
    st.markdown("---")

    sweep_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = os.path.join(RESULTS_DIR, "app_sweeps", f"sweep_{sweep_id}")

    runner: SweepRunner | None = st.session_state.get("sweep_runner")

    col_launch, col_stop = st.columns(2)
    with col_launch:
        if st.button("🚀 Launch Sweep", use_container_width=True,
                      disabled=runner is not None and not runner.is_done):
            sweep_dir = os.path.join(RESULTS_DIR, "app_sweeps",
                                     f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            runner = SweepRunner(
                sweep_dir=sweep_dir,
                n_gpus=gpu_count,
                configs=configs,
                model_path=cfg.get("model_path", "model_v2.ckpt"),
                reward=sweep_reward,
            )
            runner.launch()
            st.session_state["sweep_runner"] = runner
            st.session_state["sweep_dir"] = sweep_dir
            st.session_state["sweep_params"] = list(param_ranges.keys())
            _load_all_metrics.clear()
            st.rerun()

    with col_stop:
        if runner and not runner.is_done:
            if st.button("⏹ Stop Sweep", use_container_width=True, type="secondary"):
                runner.stop()
                st.session_state["sweep_runner"] = None
                st.warning("Sweep stopped.")

    # ── Progress ──────────────────────────────────────────────────
    if runner:
        status = runner.poll()
        total = status["total"]
        done = status["completed"]
        running = status["running"]
        failed = status["failed"]

        st.progress(done / max(total, 1),
                    text=f"{done}/{total} complete · {running} running · {failed} failed")

        if status["running_names"]:
            gpu_text = " · ".join(f"GPU{g}: {n}" for n, g in status["running_names"].items())
            st.caption(gpu_text)

        if status["failed_names"]:
            with st.expander(f"Failed runs ({failed})", expanded=False):
                for fn in status["failed_names"]:
                    err = runner.errors.get(fn, "unknown error")
                    st.error(f"**{fn}**: {err[:200]}")

        # Auto-refresh while running
        if not runner.is_done:
            time.sleep(3)
            st.rerun()
        else:
            st.success(f"Sweep complete! {done} runs finished, {failed} failed.")
            st.session_state["sweep_runner"] = None

        # ── Analysis ──────────────────────────────────────────────
        sweep_dir_active = st.session_state.get("sweep_dir", "")
        swept_params = st.session_state.get("sweep_params", [])
        if sweep_dir_active and done >= 2:
            _sweep_analysis(sweep_dir_active, swept_params)

    # ── Load previous sweep ───────────────────────────────────────
    elif not runner:
        sweep_base = os.path.join(RESULTS_DIR, "app_sweeps")
        if os.path.isdir(sweep_base):
            prev_sweeps = sorted(
                [d for d in os.listdir(sweep_base)
                 if os.path.isdir(os.path.join(sweep_base, d))],
                reverse=True,
            )
            if prev_sweeps:
                st.markdown("---")
                st.markdown("#### Previous Sweeps")
                selected_sweep = st.selectbox("Load sweep", prev_sweeps,
                                              key="sweep_prev")
                if selected_sweep:
                    prev_dir = os.path.join(sweep_base, selected_sweep)
                    n_runs = sum(1 for e in os.scandir(prev_dir)
                                 if e.is_dir() and
                                 os.path.exists(os.path.join(e.path, "metrics.json")))
                    st.caption(f"{n_runs} completed runs in `{selected_sweep}`")
                    if n_runs >= 2:
                        # Infer swept params from variation in results
                        all_param_keys = [
                            "beam_width", "branching_factor", "steps_per_interval",
                            "diversity_penalty", "c_uct", "num_steps", "alpha",
                            "mh_steps", "ode_steps", "softmax_temp", "num_samples",
                        ]
                        _sweep_analysis(prev_dir, all_param_keys)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    cfg = render_sidebar()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏆 Results", "🚀 Generate", "🔬 Visualise", "📊 Evaluate",
        "⚖️ Compare", "🔬 Sweep",
    ])

    with tab1:
        tab_results(cfg)
    with tab2:
        tab_generate(cfg)
    with tab3:
        tab_visualize(cfg)
    with tab4:
        tab_evaluate(cfg)
    with tab5:
        tab_compare(cfg)
    with tab6:
        tab_sweep(cfg)


if __name__ == "__main__":
    main()
