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


# ═══════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════

def _default_slider_ranges():
    """Default min/max/step for each slider parameter."""
    return {
        "samples":      {"min": 1,    "max": 500,   "step": 5,    "default": 50},
        "softmax_temp": {"min": 0.1,  "max": 3.0,   "step": 0.1,  "default": 1.2},
        "randomness":   {"min": 0.0,  "max": 5.0,   "step": 0.1,  "default": 2.0},
        "gamma":        {"min": 0.0,  "max": 1.0,   "step": 0.05, "default": 0.0},
        "num_steps":    {"min": 5,    "max": 100,   "step": 1,    "default": 50},
        "mh_steps":     {"min": 0,    "max": 10,    "step": 1,    "default": 2},
        "alpha":        {"min": 0.0,  "max": 500.0, "step": 10.0, "default": 100.0},
        "ode_steps":    {"min": 5,    "max": 200,   "step": 5,    "default": 20},
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

        model_path = st.text_input(
            "Model checkpoint",
            value="model.ckpt",
            help="Path to GenMol .ckpt file",
        )

        sampler_type = st.selectbox(
            "Sampler", ["Standard", "DAPS"],
            help="Standard = confidence‑based denoising · DAPS = annealing + MH",
        )

        st.divider()
        st.markdown("### Generation parameters")

        sr = _get_slider_ranges()

        task = st.selectbox(
            "Task",
            ["De novo", "Fragment linking", "Fragment linking (1‑step)",
             "Motif extension", "Scaffold decoration",
             "Superstructure", "Mask modification"],
        )

        num_samples = st.slider("Samples",
            sr["samples"]["min"], sr["samples"]["max"],
            sr["samples"]["default"], step=sr["samples"]["step"])
        softmax_temp = st.slider("Softmax temperature",
            sr["softmax_temp"]["min"], sr["softmax_temp"]["max"],
            sr["softmax_temp"]["default"], sr["softmax_temp"]["step"])
        randomness = st.slider("Randomness",
            sr["randomness"]["min"], sr["randomness"]["max"],
            sr["randomness"]["default"], sr["randomness"]["step"])
        gamma = st.slider("MCG γ (context guidance)",
            sr["gamma"]["min"], sr["gamma"]["max"],
            sr["gamma"]["default"], sr["gamma"]["step"])

        fragment_input = ""
        if task not in ("De novo",):
            fragment_input = st.text_area(
                "Fragment / SMILES input",
                placeholder="e.g. c1ccc(*)cc1 or CC(=O)O.c1ccccc1",
                height=80,
            )

        daps_kwargs = {}
        if sampler_type == "DAPS":
            st.divider()
            st.markdown("### DAPS parameters")
            daps_kwargs["reward"] = st.selectbox(
                "Reward function",
                ["None", "MW", "QED", "LogP", "TPSA"],
                help="Property to optimise during MH refinement",
            )
            daps_kwargs["num_steps"] = st.slider("Annealing steps",
                sr["num_steps"]["min"], sr["num_steps"]["max"],
                sr["num_steps"]["default"], step=sr["num_steps"]["step"])
            daps_kwargs["mh_steps"] = st.slider("MH steps / anneal step",
                sr["mh_steps"]["min"], sr["mh_steps"]["max"],
                sr["mh_steps"]["default"], step=sr["mh_steps"]["step"])
            daps_kwargs["alpha"] = st.slider("α (reward weight)",
                sr["alpha"]["min"], sr["alpha"]["max"],
                sr["alpha"]["default"], sr["alpha"]["step"])
            daps_kwargs["ode_steps"] = st.slider("ODE sub‑steps",
                sr["ode_steps"]["min"], sr["ode_steps"]["max"],
                sr["ode_steps"]["default"], sr["ode_steps"]["step"])

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
            "gamma": gamma,
            "fragment": fragment_input.strip(),
            "daps_kwargs": daps_kwargs,
            "highlight": highlight_smi.strip(),
        }


# ═══════════════════════════════════════════════════════════════════════
# Generation
# ═══════════════════════════════════════════════════════════════════════

def _get_sampler(cfg: dict):
    """Return the appropriate sampler instance (Standard or DAPS)."""
    path = cfg["model_path"]
    if cfg["sampler_type"] == "DAPS":
        dk = cfg["daps_kwargs"]
        return load_daps_sampler(
            path,
            reward_name=dk.get("reward", "None"),
            num_steps=dk.get("num_steps", 50),
            mh_steps=dk.get("mh_steps", 2),
            alpha=dk.get("alpha", 100.0),
            ode_steps=dk.get("ode_steps", 20),
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
        gamma=cfg["gamma"],
    )
    sampler = _get_sampler(cfg)
    is_daps = cfg["sampler_type"] == "DAPS"

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
    if is_daps and results:
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
    st.markdown(
        f"**{cfg['sampler_type']}** sampler · **{cfg['task']}** · "
        f"**{cfg['num_samples']}** samples · temp={cfg['softmax_temp']} "
        f"· rand={cfg['randomness']} · γ={cfg['gamma']}"
    )

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
        default_name = (
            f"{gen_cfg.get('sampler_type', 'Std')}_"
            f"{gen_cfg.get('task', 'denovo').replace(' ', '')}_"
            f"n{n_total}"
        )
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
            st.success(f"Saved **{run_name}** ({n_total} molecules)")

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

    # Gather datasets: saved runs + current session + uploaded files
    datasets = {}

    # Saved runs from session
    saved_runs = st.session_state.get("saved_runs", {})
    if saved_runs:
        st.markdown(f"**Saved runs** ({len(saved_runs)}):")
        selected_runs = st.multiselect(
            "Select saved runs to compare",
            list(saved_runs.keys()),
            default=list(saved_runs.keys()),
        )
        for name in selected_runs:
            datasets[name] = saved_runs[name]["smiles"]

        # Option to clear saved runs
        if st.button("🗑️ Clear all saved runs", key="clear_saved"):
            st.session_state["saved_runs"] = {}
            st.rerun()

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
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    cfg = render_sidebar()

    tab1, tab2, tab3, tab4 = st.tabs([
        "🚀 Generate", "🔬 Visualise", "📊 Evaluate", "⚖️ Compare",
    ])

    with tab1:
        tab_generate(cfg)
    with tab2:
        tab_visualize(cfg)
    with tab3:
        tab_evaluate(cfg)
    with tab4:
        tab_compare(cfg)


if __name__ == "__main__":
    main()
