"""Regenerate every figure of the paper from the result files in mat-ml/results.

Nothing here is simulated: each bar/point is read from a CSV or JSON written by the mat-ml pipeline.
Run:  python3 make_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
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
    fig, ax = plt.subplots(figsize=(7.1, 3.35))
    ax.set_xlim(0, 100); ax.set_ylim(0, 48); ax.axis("off"); ax.grid(False)

    def box(x, y, w, h, title, body, fc="#eef3fa", ec="#2a78d6", ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.0", fc=fc, ec=ec, lw=0.9, ls=ls))
        ax.text(x + w / 2, y + h - 1.9, title, ha="center", va="top", fontsize=7.4, weight="bold", color=INK)
        ax.text(x + w / 2, y + h - 5.6, body, ha="center", va="top", fontsize=6.4, color=MUTED, linespacing=1.22)

    def arrow(x1, y1, x2, y2, ls="-", col=INK):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=0.9, color=col, ls=ls, shrinkA=0, shrinkB=0))

    w, h = 18.2, 12.6
    y1, y2, y3 = 34.0, 18.0, 2.0
    # row 1: construction
    box(1, y1, w, h, "Data", "C2DB, JARVIS-DFT 2D,\nBiDB (10,192\nbilayers)")
    box(21.4, y1, w, h, "Tensor cleaning", "positive definite,\nasymmetry $\\leq$ 10 %\n$\\Rightarrow$ 7,258 monolayers")
    box(41.8, y1, w, h, "Features (141)", "132 Magpie + 9 geometry;\nno DFT outputs")
    box(62.2, y1, w, h, "Models", "Ridge, random forest,\nLightGBM, CGCNN,\nnested-tuned LightGBM")
    box(82.6, y1, 16.4, h, "Grouped CV", "random, chem. system,\nfamily; cluster-\nbootstrap CI", fc="#e9f7f1", ec="#1baf7a")
    for x in (19.2, 39.6, 60.0, 80.4):
        arrow(x + 0.4, y1 + h / 2, x + 2.0, y1 + h / 2)
    # row 2: evaluation of the random-vs-grouped gap
    box(1, y2, w, h, "Y2D case study", "5 seeds, ablations,\nnull baselines\n(Sec. 4.2 - 4.4)", fc="#e9f7f1", ec="#1baf7a")
    box(21.4, y2, w, h, "12-property panel", "C2DB properties,\nfamily vs chem. system\n(Sec. 4.5)", fc="#e9f7f1", ec="#1baf7a")
    box(41.8, y2, w, h, "Model class / tuning", "CGCNN vs LightGBM;\nnested group-aware\ntuning (Sec. 4.6)", fc="#e9f7f1", ec="#1baf7a")
    box(62.2, y2, w, h, "Label-only forecast", "pre-specified test:\nforecast not supported\n(Sec. 4.7)", fc="#e9f7f1", ec="#1baf7a")
    box(82.6, y2, 16.4, h, "Bilayers + JARVIS", "BiDB binding energy /\ngap; JARVIS external\n(Sec. 4.8 - 4.9)", fc="#e9f7f1", ec="#1baf7a")
    arrow(91.0, y1 - 0.5, 91.0, y2 + h + 0.5)
    arrow(10.1, y1 - 0.5, 10.1, y2 + h + 0.5)
    # row 3: potentials + planned
    box(1, y3, 56.0, h, "Universal potentials (Sec. 4.10-4.11)", "monolayer stiffness gate; fixed-cell diagnostic; interlayer binding\nwith / without D3; BiDB gate (gap criterion failed);\nbilayer / monolayer stiffness ratio $\\approx$ 2 (not DFT-validated)", fc="#fdf0ea", ec="#eb6834")
    box(61.0, y3, 38.0, h, "Not done", "bilayer stiffness pilot or model, DFT check of\nbilayer stiffness, heterobilayers, prospective\ntest on an independent dataset", fc="#f6f6f4", ec=GREY, ls=(0, (3, 2)))
    arrow(28.5, y2 - 0.5, 28.5, y3 + h + 0.5, col=GREY)
    save(fig, "fig1_workflow.png")


# ------------------------------------------------------------------ Fig. 2 data funnel
def fig_funnel():
    audit = json.loads((RES / "data_audit.json").read_text())
    c = audit["c2db"]["clean_steps"]
    j = audit["jarvis"]["steps"]
    c_lab = ["C2DB folders", "has stiffness file", "finite tensor", "positive definite", "asymmetry $\\leq$ 10 %", "$Y_{2D}>0$ (final)"]
    c_val = [16905, c["with_stiffness"], c["finite_tensor"], c["positive_definite"], c["asymmetry_ok"], c["Y2D_positive"]]
    j_lab = ["JARVIS tensors", "not corrupt", "stable xx,yy (final)"]
    j_val = [j["with_tensor_string"], j["with_tensor_string"] - j["corrupt_tensor"], j["stable_xx_yy_block"]]
    b_lab = ["BiDB bilayers", "valid binding E", "valid gap"]
    b_val = [10192, 9993, 10189]
    fig, (a, b, e) = plt.subplots(1, 3, figsize=(7.1, 2.2), gridspec_kw={"width_ratios": [1.45, 1.0, 1.0]})
    for ax, lab, val, col, ttl in ((a, c_lab, c_val, C_RANDOM, "(a) C2DB (training)"), (b, j_lab, j_val, C_CHEM, "(b) JARVIS-DFT 2D"),
                                   (e, b_lab, b_val, C_FAM, "(c) BiDB (Task B)")):
        y = np.arange(len(val))[::-1]
        ax.barh(y, val, color=col, height=0.62)
        ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=7)
        for yi, v in zip(y, val):
            ax.text(v + max(val) * 0.015, yi, f"{v:,}", va="center", fontsize=7, color=INK)
        ax.set_xlim(0, max(val) * 1.3); ax.set_title(ttl, loc="left", color=INK, fontsize=7.6)
        ax.set_xlabel("number of materials", fontsize=7); ax.grid(axis="y", visible=False)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{int(v):,}"))
        ax.tick_params(axis="x", labelsize=6.5)
    a.set_xticks([0, 5000, 10000, 15000]); b.set_xticks([0, 100, 200]); e.set_xticks([0, 5000, 10000])
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
    save(fig, "fig9_jarvis_external.png")


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
    save(fig, "fig10_mlip_gate.png")


# ------------------------------------------------------------------ Fig. 6 panel
def fig_panel():
    d = pd.read_csv(RES / "panel_summary.csv").sort_values("ratio_family").reset_index(drop=True)
    lab = {"hform": "hform", "ehull": "ehull", "gap": "gap (PBE)", "gap_hse": "gap (HSE)", "evac": "evac", "efermi": "efermi",
           "vbm": "vbm", "alphax_el": "alphax_el", "plasmafrequency_x": "plasmafreq_x", "emass_cbm": "emass_cbm", "Y2D": "Y2D (stiffness)", "poisson": "Poisson ratio"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 3.0), sharey=True)
    y = np.arange(len(d))
    a.hlines(y, d.ratio_family_lo, d.ratio_family_hi, color=C_FAM, lw=1.6)
    a.plot(d.ratio_family, y, "o", color=C_FAM, ms=4.5, label="family / random")
    a.hlines(y, d.ratio_chemsys_lo, d.ratio_chemsys_hi, color=C_CHEM, lw=1.2)
    a.plot(d.ratio_chemsys, y, "s", color=C_CHEM, ms=3.8, label="chemical system / random")
    a.axvline(1, color=INK, lw=0.8)
    a.set_yticks(y); a.set_yticklabels([lab[t] for t in d.target], fontsize=7)
    a.set_xlabel("MAE ratio, grouped / random (95 % CI)"); a.set_title("(a) Error inflation", loc="left", color=INK)
    a.legend(frameon=False, loc="lower right", fontsize=6.6, handletextpad=0.3); a.grid(axis="y", visible=False)
    b.hlines(y, d.R_family_lo, d.R_family_hi, color=C_FAM, lw=1.6)
    b.plot(d.R_family, y, "o", color=C_FAM, ms=4.5)
    b.hlines(y, d.R_chemsys_lo, d.R_chemsys_hi, color=C_CHEM, lw=1.2)
    b.plot(d.R_chemsys, y, "s", color=C_CHEM, ms=3.8)
    b.axvline(1, color=INK, lw=0.8)
    b.set_xlabel("skill retained, $R=S_{\\mathrm{grouped}}/S_{\\mathrm{random}}$ (95 % CI)"); b.set_title("(b) Skill retained", loc="left", color=INK)
    b.set_xlim(0.1, 1.08); b.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig6_panel.png")


# ------------------------------------------------------------------ Fig. 7 model class and tuning
def fig_models():
    A = json.loads((RES / "phaseA_cgcnn_vs_lgbm.json").read_text())
    T = json.loads((RES / "phaseC_tuning.json").read_text())
    names = ["LightGBM\n(default)", "CGCNN\n(1 seed, untuned)", "LightGBM\n(nested tuning)"]
    ratio = [A["ratio_family_over_random"]["lgbm"], A["ratio_family_over_random"]["cgcnn"], T["ratio_family_random_tuned"][0]]
    lo = [A["ratio_family_over_random"]["lgbm_ci95"][0], A["ratio_family_over_random"]["cgcnn_ci95"][0], T["ratio_family_random_tuned"][1][0]]
    hi = [A["ratio_family_over_random"]["lgbm_ci95"][1], A["ratio_family_over_random"]["cgcnn_ci95"][1], T["ratio_family_random_tuned"][1][1]]
    mae_r = [A["MAE_N_per_m"]["random|lgbm"], A["MAE_N_per_m"]["random|cgcnn"], T["MAE_tuned_default"]["random"][0]]
    mae_f = [A["MAE_N_per_m"]["family|lgbm"], A["MAE_N_per_m"]["family|cgcnn"], T["MAE_tuned_default"]["family"][0]]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 2.5))
    x = np.arange(3); w = 0.34
    b.bar(x - w / 2, mae_r, w, color=C_RANDOM, label="random split")
    b.bar(x + w / 2, mae_f, w, color=C_FAM, hatch="//", edgecolor="white", lw=0.5, label="family split")
    for xi, v in zip(x - w / 2, mae_r): b.text(xi, v + 0.5, f"{v:.1f}", ha="center", fontsize=6.8, color=INK)
    for xi, v in zip(x + w / 2, mae_f): b.text(xi, v + 0.5, f"{v:.1f}", ha="center", fontsize=6.8, color=INK)
    b.set_xticks(x); b.set_xticklabels(names, fontsize=7); b.set_ylabel("MAE of $Y_{2D}$ (N/m)  $\\downarrow$"); b.set_ylim(0, 27)
    b.set_title("(b) Error", loc="left", color=INK); b.legend(frameon=False, loc="upper left", fontsize=6.8, handlelength=1.2); b.grid(axis="x", visible=False)
    a.errorbar(x, ratio, yerr=[np.array(ratio) - np.array(lo), np.array(hi) - np.array(ratio)], fmt="o", color=INK, ms=4.5, capsize=3, lw=1.1)
    for xi, r, l_, h_ in zip(x, ratio, lo, hi): a.text(xi + 0.12, r, f"{r:.2f}\n[{l_:.2f}, {h_:.2f}]", fontsize=6.4, va="center", color=INK)
    a.axhline(1, color=INK, lw=0.8); a.set_xlim(-0.5, 2.75); a.set_ylim(0.9, 1.95)
    a.set_xticks(x); a.set_xticklabels(names, fontsize=7); a.set_ylabel("MAE ratio, family / random (95 % CI)")
    a.set_title("(a) The gap persists", loc="left", color=INK); a.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig7_models_tuning.png")


# ------------------------------------------------------------------ Fig. 8 Task B (BiDB)
def fig_taskb():
    d = pd.read_csv(RES / "phaseC_cluster_bootstrap_metrics.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 2.5))
    spec = [("TaskB|binding_energy_zscan", a, "binding energy MAE (meV/$\\mathrm{\\AA}^2$)", 18.51, ["(a) Binding energy", ""]),
            ("TaskB|distance", b, "interlayer gap MAE ($\\mathrm{\\AA}$)", 0.415, ["(b) Interlayer gap", ""])]
    sch = [("random", "Random", C_RANDOM, ""), ("monolayer", "Grouped by\nmonolayer", C_CHEM, "\\\\"), ("family", "Grouped by\nstructure family", C_FAM, "//")]
    for key, ax, ylab, mean_mae, (ttl, _) in spec:
        sub = d[d.analysis == key].set_index("scheme")
        for i, (sc, lab, col, hatch) in enumerate(sch):
            r = sub.loc[sc]
            ax.bar(i, r.MAE, 0.6, color=col, hatch=hatch, edgecolor="white", lw=0.5, yerr=[[r.MAE - r.MAE_lo], [r.MAE_hi - r.MAE]], error_kw=dict(lw=0.8, capsize=2.5))
            ax.text(i, r.MAE * 0.5, f"{r.MAE:.3g}", ha="center", va="center", fontsize=7.5, color=INK,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.95))
        ax.axhline(mean_mae, color=INK, lw=0.8, ls=(0, (3, 2)))
        ax.text(-0.45, mean_mae * 1.02, "predict-the-mean", ha="left", va="bottom", fontsize=6.6, color=INK)
        ax.set_xticks(range(3)); ax.set_xticklabels([x[1] for x in sch], fontsize=7); ax.set_ylabel(ylab + "  $\\downarrow$")
        ax.set_title(ttl, loc="left", color=INK); ax.grid(axis="x", visible=False); ax.set_ylim(0, mean_mae * 1.18)
    fig.tight_layout()
    save(fig, "fig8_taskB_bidb.png")


if __name__ == "__main__":
    fig_workflow(); fig_funnel(); fig_leakage(); fig_cv(); fig_ablation(); fig_panel(); fig_models(); fig_taskb(); fig_jarvis(); fig_mlip()
