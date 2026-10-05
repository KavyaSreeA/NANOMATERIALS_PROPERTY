"""Regenerate every figure of the paper from the result files in mat-ml/results.

Nothing here is simulated: each bar/point is read from a CSV or JSON written by the mat-ml pipeline.
Run:  python3 make_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "mat-ml" / "results"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.grid": True, "grid.color": "#e4e4e0", "grid.linewidth": 0.5, "axes.axisbelow": True,
    "savefig.dpi": 300, "figure.dpi": 150, "mathtext.fontset": "stix",
})
INK, MUTED = "#0b0b0b", "#52514e"
C_RANDOM, C_CHEM, C_FAM = "#2a78d6", "#eb6834", "#1baf7a"   # validated categorical slots 1-3
GREY = "#9a9994"


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ Fig. 1 workflow
def fig_workflow():
    fig, ax = plt.subplots(figsize=(7.1, 2.55))
    ax.set_xlim(0, 100); ax.set_ylim(0, 36); ax.axis("off"); ax.grid(False)

    def box(x, y, w, h, title, body, fc="#eef3fa", ec="#2a78d6", ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.0",
                                    fc=fc, ec=ec, lw=0.9, ls=ls))
        ax.text(x + w / 2, y + h - 2.1, title, ha="center", va="top", fontsize=7.6, weight="bold", color=INK)
        ax.text(x + w / 2, y + h - 6.0, body, ha="center", va="top", fontsize=6.7, color=MUTED, linespacing=1.25)

    def arrow(x1, y1, x2, y2, ls="-"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color=INK, ls=ls, shrinkA=0, shrinkB=0))

    w, h, y1 = 18.2, 13.6, 20.5
    box(1, y1, w, h, "C2DB monolayers", "16,905 entries\n8,462 with stiffness\n(GPAW / PBE)")
    box(21.4, y1, w, h, "Tensor cleaning", "positive definite\nasymmetry $\\leq$ 10 %\n$\\Rightarrow$ 7,258 rows")
    box(41.8, y1, w, h, "Features (141)", "132 Magpie composition\n9 geometry / symmetry\nno DFT outputs")
    box(62.2, y1, w, h, "Models", "Ridge, Random forest,\nLightGBM (fixed\nhyper-parameters)")
    box(82.6, y1, 16.4, h, "Grouped CV", "random, chem. system,\nstructure family;\n5 folds $\\times$ 5 seeds", fc="#e9f7f1", ec="#1baf7a")
    for x in (19.2, 39.6, 60.0, 80.4):
        arrow(x + 0.4, y1 + h / 2, x + 2.0, y1 + h / 2)

    y2 = 2.0
    box(41.8, y2, w, h, "Feature ablation", "composition / geometry /\nspace group / prototype\n(group-level only)", fc="#e9f7f1", ec="#1baf7a")
    box(62.2, y2, w, h, "External check", "train on C2DB, test on\nJARVIS-DFT (186 rows)\nbootstrap intervals", fc="#e9f7f1", ec="#1baf7a")
    box(82.6, y2, 16.4, h, "MLIP calibration", "MACE-MP-0/-MPA-0,\nCHGNet vs DFT\n(monolayers only)", fc="#fdf0ea", ec="#eb6834")
    box(1, y2, 38.6, h, "Planned, not yet run: bilayer labels", "stacking enumeration $\\rightarrow$ MLIP labels\n$\\rightarrow$ DFT spot check $\\rightarrow$ bilayer model", fc="#f6f6f4", ec=GREY, ls=(0, (3, 2)))
    for x in (51.0, 71.4):
        arrow(x, y1 - 0.4, x, y2 + h + 0.4)
    arrow(91.0, y1 - 0.4, 91.0, y2 + h + 0.4)
    save(fig, "fig1_workflow.png")


# ------------------------------------------------------------------ Fig. 2 data funnel
def fig_funnel():
    audit = json.loads((RES / "data_audit.json").read_text())
    c = audit["c2db"]["clean_steps"]
    j = audit["jarvis"]["steps"]
    c_lab = ["C2DB folders", "has stiffness file", "finite tensor", "positive definite", "asymmetry $\\leq$ 10 %", "$Y_{2D}>0$ (final)"]
    c_val = [16905, c["with_stiffness"], c["finite_tensor"], c["positive_definite"], c["asymmetry_ok"], c["Y2D_positive"]]
    j_lab = ["JARVIS tensors", "not corrupt", "stable xx,yy block (final)"]
    j_val = [j["with_tensor_string"], j["with_tensor_string"] - j["corrupt_tensor"], j["stable_xx_yy_block"]]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 2.3), gridspec_kw={"width_ratios": [1.35, 1]})
    for ax, lab, val, col, ttl in ((a, c_lab, c_val, C_RANDOM, "(a) C2DB (training)"), (b, j_lab, j_val, C_CHEM, "(b) JARVIS-DFT 2D (external)")):
        y = np.arange(len(val))[::-1]
        ax.barh(y, val, color=col, height=0.62)
        ax.set_yticks(y); ax.set_yticklabels(lab)
        for yi, v in zip(y, val):
            ax.text(v + max(val) * 0.012, yi, f"{v:,}", va="center", fontsize=7.5, color=INK)
        ax.set_xlim(0, max(val) * 1.18); ax.set_title(ttl, loc="left", color=INK)
        ax.set_xlabel("number of materials"); ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig2_data_funnel.png")


# ------------------------------------------------------------------ Fig. 3 leakage
def fig_leakage():
    schemes = ["random", "cluster", "chemsys", "family"]
    lab = ["Random", "Cluster\n(K-means)", "Chemical\nsystem", "Structure\nfamily"]
    info = json.loads((RES / "task_A_run_info.json").read_text())["leakage_first_target"]
    keys = [("chemsys_in_train", "same chemical system", C_RANDOM, ""), ("formula_in_train", "same formula", C_CHEM, "\\\\"),
            ("family_in_train", "same structure family", C_FAM, "//")]
    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    x = np.arange(len(schemes)); w = 0.26
    for i, (k, name, col, hatch) in enumerate(keys):
        v = [info[s][k] * 100 for s in schemes]
        bars = ax.bar(x + (i - 1) * w, v, w, color=col, hatch=hatch, edgecolor="white", linewidth=0.6, label=name)
        for xi, vi in zip(x + (i - 1) * w, v):
            ax.text(xi, vi + 1.5, f"{vi:.0f}", ha="center", fontsize=6.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylim(0, 150)
    ax.set_yticks([0,20,40,60,80,100])
    ax.set_ylabel("test rows with a match in training (%)")
    ax.legend(frameon=False, loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.0), handlelength=1.2, fontsize=6.8)
    ax.grid(axis="x", visible=False)
    save(fig, "fig3_leakage.png")


# ------------------------------------------------------------------ Fig. 4 CV results
def fig_cv():
    d = pd.read_csv(RES / "robustness_summary.csv")
    sel = [("global_mean", "none", "Global mean (no features)", GREY, ""),
           ("family_mean", "none", "Family mean (no features)", "#c9c8c2", ""),
           ("random_forest", "all", "Random forest", C_CHEM, "\\\\"),
           ("gradient_boosting", "all", "LightGBM", C_RANDOM, "")]
    schemes = ["random", "chemsys", "family"]
    names = ["Random", "Chemical system", "Structure family"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 2.55))
    x = np.arange(3); w = 0.2
    for i, (m, f, lab, col, hatch) in enumerate(sel):
        r = [d[(d.model == m) & (d.features == f) & (d.scheme == s)].iloc[0] for s in schemes]
        a.bar(x + (i - 1.5) * w, [q.mae_mean for q in r], w, yerr=[q.mae_sd for q in r], color=col, hatch=hatch,
              edgecolor="white", linewidth=0.5, error_kw=dict(lw=0.7, capsize=1.5), label=lab)
        if m == "gradient_boosting":
            for xi, q in zip(x + (i - 1.5) * w, r):
                a.text(xi, q.mae_mean + q.mae_sd + 0.8, f"{q.mae_mean:.1f}", ha="center", fontsize=6.8, color=INK)
                if q.scheme == "family":
                    a.text(xi, q.mae_mean + q.mae_sd + 4.2, "$\\times$1.44 vs. random", ha="center", fontsize=6.8, color=INK, weight="bold")
        if m == "random_forest":
            pass
        # R2(ln)
        b.bar(x + (i - 1.5) * w, [q.r2_log_mean for q in r], w, yerr=[q.r2_log_sd for q in r], color=col, hatch=hatch,
              edgecolor="white", linewidth=0.5, error_kw=dict(lw=0.7, capsize=1.5))
        if m == "gradient_boosting":
            for xi, q in zip(x + (i - 1.5) * w, r):
                b.text(xi, max(q.r2_log_mean, 0) + 0.03, f"{q.r2_log_mean:.2f}", ha="center", fontsize=6.8, color=INK)
    a.set_xticks(x); a.set_xticklabels(names); a.set_ylabel("MAE of $Y_{2D}$ (N/m)  $\\downarrow$")
    a.set_title("(a) Error", loc="left", color=INK); a.set_ylim(0, 50); a.grid(axis="x", visible=False)
    h_, l_ = a.get_legend_handles_labels()
    fig.legend(h_, l_, frameon=False, loc="lower center", ncol=4, fontsize=7, handlelength=1.2, bbox_to_anchor=(0.5, -0.07))
    b.set_xticks(x); b.set_xticklabels(names); b.set_ylabel("$R^2$ of $\\ln Y_{2D}$  $\\uparrow$")
    b.set_title("(b) Explained variance", loc="left", color=INK); b.set_ylim(-0.05, 1.0); b.grid(axis="x", visible=False)
    b.axhline(0, color=INK, lw=0.5)
    fig.tight_layout()
    save(fig, "fig4_cv_results.png")


# ------------------------------------------------------------------ Fig. 5 ablation
def fig_ablation():
    r = pd.read_csv(RES / "robustness_summary.csv")
    s = pd.read_csv(RES / "spg_ablation_summary.csv")
    p = pd.read_csv(RES / "proto_features_summary.csv")
    allr = pd.concat([r[r.model == "gradient_boosting"], s, p])

    def get(f, sch):
        q = allr[(allr.features == f) & (allr.scheme == sch)].iloc[0]
        return q.mae_mean, q.mae_sd
    rows = [("all", "All features (141)"), ("all_no_spg", "All $-$ space group"), ("all_plus_layergroup", "All + layer group"),
            ("all_plus_anon", "All + anonymous formula"), ("all_plus_prototype", "All + both prototype sets"),
            ("composition_plus_prototype", "Composition + prototype"), ("composition_plus_spg", "Composition + space group"),
            ("composition", "Composition only (132)"), ("structure", "Structure only (geom. + space grp.)"),
            ("geometry_no_spg", "Geometry only (no space group)"), ("prototype_only", "Prototype columns only"),
            ("spg_only", "Space-group number only")]
    fig, ax = plt.subplots(figsize=(3.5, 3.9))
    y = np.arange(len(rows))[::-1]; h = 0.36
    mr = [get(f, "random") for f, _ in rows]; mf = [get(f, "family") for f, _ in rows]
    ax.barh(y + h / 2, [m[0] for m in mr], h, xerr=[m[1] for m in mr], color=C_RANDOM, edgecolor="white", lw=0.5,
            error_kw=dict(lw=0.6, capsize=1), label="random split")
    ax.barh(y - h / 2, [m[0] for m in mf], h, xerr=[m[1] for m in mf], color=C_FAM, hatch="//", edgecolor="white", lw=0.5,
            error_kw=dict(lw=0.6, capsize=1), label="family split")
    for yi, a_, b_ in zip(y, mr, mf):
        ax.text(a_[0] + 0.8, yi + h / 2, f"{a_[0]:.1f}", va="center", fontsize=6.3, color=INK)
        ax.text(b_[0] + 0.8, yi - h / 2, f"{b_[0]:.1f}", va="center", fontsize=6.3, color=INK)
    ax.axvline(42.4, color=INK, lw=0.8, ls=(0, (3, 2)))
    ax.text(42.0, y[0] + 0.55, "predict-the-mean: 42.4", ha="right", va="bottom", fontsize=6.6, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([n for _, n in rows], fontsize=6.9)
    ax.set_xlabel("MAE of $Y_{2D}$ (N/m), LightGBM, 5 seeds"); ax.set_xlim(0, 50); ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.62), fontsize=7)
    save(fig, "fig5_ablation.png")


# ------------------------------------------------------------------ Fig. 6 JARVIS
def fig_jarvis():
    d = pd.read_csv(RES / "external_summary.csv")
    d = d[d.model == "gradient_boosting"]
    subsets = [("all", "All (n=186)"), ("matched", "Matched to C2DB (n=106)"), ("unseen_chemsys", "Unseen chemistry (n=55)")]
    feats = [("all", "All features", C_RANDOM, ""), ("composition", "Composition only", C_CHEM, "\\\\"), ("structure", "Structure only", C_FAM, "//")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 2.55))
    x = np.arange(3); w = 0.26
    for i, (f, lab, col, hatch) in enumerate(feats):
        rows = [d[(d.features == f) & (d.subset == s)].iloc[0] for s, _ in subsets]
        m = np.array([q.mae for q in rows]); lo = m - np.array([q.mae_lo for q in rows]); hi = np.array([q.mae_hi for q in rows]) - m
        a.bar(x + (i - 1) * w, m, w, yerr=[lo, hi], color=col, hatch=hatch, edgecolor="white", lw=0.5,
              error_kw=dict(lw=0.7, capsize=1.5), label=lab)
        if f == "all":
            for xi, v in zip(x + (i - 1) * w, m):
                a.text(xi - 0.02, 1.0, f"{v:.1f}", ha="center", fontsize=6.6, color="white", weight="bold")
        sp = [q.spearman for q in rows]
        b.bar(x + (i - 1) * w, sp, w, color=col, hatch=hatch, edgecolor="white", lw=0.5)
        if f == "all":
            for xi, v in zip(x + (i - 1) * w, sp):
                b.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=6.6, color=INK)
    a.axhline(8.62, color=INK, lw=0.8, ls=(0, (3, 2)), label="C2DB vs JARVIS label disagreement (8.6)")
    a.set_xticks(x); a.set_xticklabels(["All\n(n=186)", "Matched to C2DB\n(n=106)", "Unseen chemistry\n(n=55)"], fontsize=7)
    a.set_ylabel("MAE of $Y_{2D}$ (N/m), 95 % bootstrap CI  $\\downarrow$"); a.set_title("(a) Error", loc="left", color=INK)
    a.set_ylim(0, 66); a.grid(axis="x", visible=False); a.legend(frameon=False, loc="upper left", fontsize=6.8, handlelength=1.4)
    b.set_xticks(x); b.set_xticklabels(["All\n(n=186)", "Matched to C2DB\n(n=106)", "Unseen chemistry\n(n=55)"], fontsize=7)
    b.set_ylabel("Spearman rank correlation  $\\uparrow$"); b.set_title("(b) Ranking quality", loc="left", color=INK)
    b.set_ylim(0, 1.05); b.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig6_jarvis_external.png")


# ------------------------------------------------------------------ Fig. 7 MLIP gate
def fig_mlip():
    g = pd.read_csv(RES / "mlip" / "calibration_gate.csv")
    models = [("mace_mp0_medium", "MACE-MP-0"), ("mace_mpa0_medium", "MACE-MPA-0"), ("chgnet_0.3.0", "CHGNet 0.3.0")]
    subs = [("c2db", "C2DB (n=150)", C_RANDOM, ""), ("jarvis", "JARVIS (n=184)", C_CHEM, "\\\\")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 2.45))
    x = np.arange(3); w = 0.32
    for i, (s, lab, col, hatch) in enumerate(subs):
        rows = [g[(g.model == m) & (g.subset == s)].iloc[0] for m, _ in models]
        mae = [q.mae for q in rows]; sp = [q.spearman for q in rows]
        a.bar(x + (i - 0.5) * w, mae, w, color=col, hatch=hatch, edgecolor="white", lw=0.5, label=lab)
        b.bar(x + (i - 0.5) * w, sp, w, color=col, hatch=hatch, edgecolor="white", lw=0.5)
        for xi, v in zip(x + (i - 0.5) * w, mae):
            a.text(xi, max(v, g.mae_threshold.iloc[0]) + 1.4, f"{v:.1f}", ha="center", fontsize=6.8, color=INK)
        for xi, v in zip(x + (i - 0.5) * w, sp):
            b.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=6.8, color=INK)
    thr = g.mae_threshold.iloc[0]
    a.axhline(thr, color=INK, lw=0.8, ls=(0, (3, 2)), label=f"pre-set gate ($\\leq${thr:.1f})")
    b.axhline(0.9, color=INK, lw=0.8, ls=(0, (3, 2)), label="pre-set gate ($\\geq$0.90)")
    for ax in (a, b):
        ax.set_xticks(x); ax.set_xticklabels([n for _, n in models]); ax.grid(axis="x", visible=False)
    a.set_ylabel("MAE of $Y_{2D}$ vs DFT (N/m)  $\\downarrow$"); a.set_title("(a) Error", loc="left", color=INK); a.set_ylim(0, 88)
    a.legend(frameon=False, loc="upper left", fontsize=6.8, handlelength=1.2)
    b.set_ylabel("Spearman vs DFT  $\\uparrow$"); b.set_title("(b) Ranking", loc="left", color=INK); b.set_ylim(0.5, 1.0)
    b.legend(frameon=False, loc="upper right", fontsize=6.8, handlelength=1.4)
    fig.tight_layout()
    save(fig, "fig7_mlip_gate.png")


if __name__ == "__main__":
    fig_workflow(); fig_funnel(); fig_leakage(); fig_cv(); fig_ablation(); fig_jarvis(); fig_mlip()
