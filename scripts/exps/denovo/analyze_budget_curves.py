#!/usr/bin/env python3
"""Compare quality vs compute budget: default vs softmax_temp=0.5 vs diversity_cutoff=0.6."""

import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED

T_APPROX = 50
BASE = "outputs/none"

# (name, K, L, curve, label)
CONFIGS = [
    ("budget_standard",        None, None, "baseline", "Standard (B=0)"),

    # Curve A: default temp=0.8, vary K (L=4)
    ("budget_K20_L4_default",  20,   4,    "default",  "K=20, L=4"),
    ("budget_K10_L4_default",  10,   4,    "default",  "K=10, L=4"),
    ("budget_K5_L2_default",   5,    2,    "default",  "K=5,  L=2"),
    ("budget_K5_L4_default",   5,    4,    "default",  "K=5,  L=4"),
    ("budget_K5_L8_default",   5,    8,    "default",  "K=5,  L=8"),
    ("budget_K2_L4_default",   2,    4,    "default",  "K=2,  L=4"),
    ("budget_K1_L4_default",   1,    4,    "default",  "K=1,  L=4"),

    # Curve B: temp=0.5, same sweep
    ("budget_K20_L4_t05",      20,   4,    "t=0.5",    "K=20, L=4"),
    ("budget_K10_L4_t05",      10,   4,    "t=0.5",    "K=10, L=4"),
    ("budget_K5_L2_t05",       5,    2,    "t=0.5",    "K=5,  L=2"),
    ("budget_K5_L4_t05",       5,    4,    "t=0.5",    "K=5,  L=4"),
    ("budget_K5_L8_t05",       5,    8,    "t=0.5",    "K=5,  L=8"),
    ("budget_K2_L4_t05",       2,    4,    "t=0.5",    "K=2,  L=4"),
    ("budget_K1_L4_t05",       1,    4,    "t=0.5",    "K=1,  L=4"),

    # Curve C: diversity_cutoff=0.6, vary K (L=4)
    ("budget_K20_L4_div",      20,   4,    "div=0.6",  "K=20, L=4"),
    ("budget_K10_L4_div",      10,   4,    "div=0.6",  "K=10, L=4"),
    ("budget_K5_L4_div",       5,    4,    "div=0.6",  "K=5,  L=4"),
    ("budget_K2_L4_div",       2,    4,    "div=0.6",  "K=2,  L=4"),
    ("budget_K1_L4_div",       1,    4,    "div=0.6",  "K=1,  L=4"),

    # Curve D: randomness=2.0, vary K (L=4)
    ("budget_K20_L4_rand",     20,   4,    "rand=2.0", "K=20, L=4"),
    ("budget_K10_L4_rand",     10,   4,    "rand=2.0", "K=10, L=4"),
    ("budget_K5_L4_rand",      5,    4,    "rand=2.0", "K=5,  L=4"),
    ("budget_K2_L4_rand",      2,    4,    "rand=2.0", "K=2,  L=4"),
    ("budget_K1_L4_rand",      1,    4,    "rand=2.0", "K=1,  L=4"),
]


def budget(K, L):
    return 0 if K is None else round(L * T_APPROX / K)


def analyze(path):
    df = pd.read_csv(path)
    smiles = df["smiles"].dropna().tolist()
    mols = [Chem.MolFromSmiles(s) for s in smiles if s]
    valid = [m for m in mols if m is not None]
    qeds = sorted([QED.qed(m) for m in valid], reverse=True)
    if not qeds:
        return None
    top10 = qeds[:max(1, len(qeds) // 10)]
    return {
        "qed_mean":  sum(qeds) / len(qeds),
        "qed_top10": sum(top10) / len(top10),
        "qed_max":   qeds[0],
        "unique":    len(set(smiles)) / max(len(smiles), 1) * 100,
    }


if __name__ == "__main__":
    rows = []
    for name, K, L, curve, label in CONFIGS:
        path = os.path.join(BASE, name, "samples.csv")
        if not os.path.exists(path):
            continue
        s = analyze(path)
        if s:
            rows.append({"budget": budget(K, L), "curve": curve,
                         "label": label, **s})

    if not rows:
        print("No results found yet.")
        exit()

    # ── Table: side-by-side default vs t=0.5 vs div=0.6 ──────────────────
    def get_curve(curve_name):
        return {r["label"]: r for r in rows if r["curve"] == curve_name}

    default_rows = get_curve("default")
    t05_rows     = get_curve("t=0.5")
    div_rows     = get_curve("div=0.6")
    rand_rows    = get_curve("rand=2.0")
    base_row     = next((r for r in rows if r["curve"] == "baseline"), None)

    hdr = (f"{'Budget':>7}  {'Config':<12}  {'default':>9}  {'t=0.5':>7}  "
           f"{'div=0.6':>9}  {'rand=2.0':>9}  {'Δ(t05)':>7}  {'Δ(div)':>7}  {'Δ(rnd)':>7}")
    print(hdr)
    print("-" * len(hdr))

    if base_row:
        print(f"{'0':>7}  {'standard':<12}  {base_row['qed_mean']:>9.4f}  "
              f"{'—':>7}  {'—':>9}  {'—':>9}  {'—':>7}  {'—':>7}  {'—':>7}")

    # unique budget points (L=4 only for clean comparison)
    l4_configs = [(n, K, L, c, lb) for n, K, L, c, lb in CONFIGS if L == 4 and K is not None]
    budget_labels = {}
    for n, K, L, c, lb in l4_configs:
        B = budget(K, L)
        if B not in budget_labels:
            budget_labels[B] = lb

    for B, label in sorted(budget_labels.items()):
        d = default_rows.get(label)
        t = t05_rows.get(label)
        v = div_rows.get(label)
        r = rand_rows.get(label)
        d_mean = f"{d['qed_mean']:.4f}" if d else "—"
        t_mean = f"{t['qed_mean']:.4f}" if t else "—"
        v_mean = f"{v['qed_mean']:.4f}" if v else "—"
        r_mean = f"{r['qed_mean']:.4f}" if r else "—"
        dt = f"{t['qed_mean'] - d['qed_mean']:+.4f}" if (d and t) else "—"
        dv = f"{v['qed_mean'] - d['qed_mean']:+.4f}" if (d and v) else "—"
        dr = f"{r['qed_mean'] - d['qed_mean']:+.4f}" if (d and r) else "—"
        print(f"{B:>7}  {label:<12}  {d_mean:>9}  {t_mean:>7}  {v_mean:>9}  {r_mean:>9}  {dt:>7}  {dv:>7}  {dr:>7}")

    # ── Summary: best config at each budget ───────────────────────────────
    print("\n--- Best config per budget (by QED mean, L=4 only) ---")
    for B, label in sorted(budget_labels.items()):
        candidates = []
        for curve_name, curve_dict in [("default", default_rows), ("t=0.5", t05_rows),
                                        ("div=0.6", div_rows), ("rand=2.0", rand_rows)]:
            r = curve_dict.get(label)
            if r:
                candidates.append((r["qed_mean"], curve_name, r))
        if candidates:
            best_mean, best_curve, best_r = max(candidates)
            print(f"  B={B:>4}: {best_curve:<9}  QED={best_mean:.4f}  top10={best_r['qed_top10']:.4f}")
