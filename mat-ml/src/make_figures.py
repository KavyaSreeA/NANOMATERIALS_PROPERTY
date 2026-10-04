"""Summary figures for the README, drawn from the saved result files only.   python -m src.make_figures   -> results/figures/*.png

fig1  Task A: Y2D MAE by split scheme (LightGBM, random forest, baselines; 5 seeds)
fig2  Task A: feature-group ablation (LightGBM, random vs family split; 5 seeds)
fig3  Leakage: fraction of test materials sharing a group with the training data, per scheme (Task A and Task B)
fig4  Task B: MAE by split scheme vs the mean predictor (binding energy and gap; 5 seeds)
fig5  Task B: per-fold MAE spread (seed 42)
fig6  ML-potential gate: Y2D MAE and Spearman for three potentials on C2DB / JARVIS / combined
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import data as D

cfg = D.load_config()
RD = D.ROOT / cfg["paths"]["results_dir"]
OUT = RD / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
C = {"mean": "#b0b0b0", "chem": "#d9c28f", "fam": "#e6a86c", "rf": "#6fa8dc", "gb": "#2a6fbb", "ridge": "#8e7cc3"}


def save(fig, name, rect=None):
    fig.tight_layout(rect=rect) if rect else fig.tight_layout()
    fig.savefig(OUT / name, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig1
rob = pd.read_csv(RD / "robustness_summary.csv")
schemes = ["random", "chemsys", "family"]
rows = [("mean of all\ntraining targets", "global_mean", "none", C["mean"]), ("chemical-system\nmean", "chemsys_mean", "none", C["chem"]),
        ("structure-family\nmean", "family_mean", "none", C["fam"]), ("random forest", "random_forest", "all", C["rf"]), ("LightGBM", "gradient_boosting", "all", C["gb"])]
fig, ax = plt.subplots(figsize=(9, 4.4))
w = 0.16
for i, (lab, model, fs, col) in enumerate(rows):
    g = rob[(rob.model == model) & (rob.features == fs)].set_index("scheme").loc[schemes]
    b = ax.bar(np.arange(3) + (i - 2) * w, g.mae_mean, w, yerr=g.mae_sd, color=col, label=lab, capsize=2)
    for x, v in zip(np.arange(3) + (i - 2) * w, g.mae_mean):
        ax.text(x, v + 0.8, f"{v:.1f}", ha="center", fontsize=7)
ax.set_xticks(range(3)); ax.set_xticklabels(["random split", "held-out chemical systems", "held-out structure families"])
ax.set_ylabel("Y2D MAE (N/m), lower is better"); ax.set_title("Task A: stiffness error by split scheme (mean +/- sd over 5 seeds)", pad=34)
ax.legend(fontsize=8, ncol=5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
save(fig, "fig1_taskA_random_vs_grouped.png")

# ---------------------------------------------------------------- fig2
parts = [rob[(rob.model == "gradient_boosting")], pd.read_csv(RD / "spg_ablation_summary.csv"), pd.read_csv(RD / "proto_features_summary.csv")]
A = pd.concat([p[p.model == "gradient_boosting"] if "model" in p else p for p in parts])
names = {"all": "all 141 features", "composition": "composition only (Magpie)", "structure": "structure only (geometry + space group)", "geometry_no_spg": "geometry only (no space group)",
         "spg_only": "space-group number only", "all_no_spg": "all without space group", "composition_plus_spg": "composition + space group",
         "all_plus_layergroup": "all + layer group", "all_plus_anon": "all + anonymous formula", "all_plus_prototype": "all + layer group + anonymous formula",
         "composition_plus_prototype": "composition + layer group + anon. formula", "prototype_only": "layer group + anonymous formula only"}
order = ["all", "all_no_spg", "all_plus_layergroup", "all_plus_anon", "all_plus_prototype", "composition", "composition_plus_spg", "composition_plus_prototype", "structure",
         "geometry_no_spg", "prototype_only", "spg_only"]
A = A[A.features.isin(order) & A.scheme.isin(["random", "family"])].drop_duplicates(["features", "scheme"])
fig, ax = plt.subplots(figsize=(9, 5.4))
y = np.arange(len(order))
for off, sc, col in ((-0.2, "random", C["gb"]), (0.2, "family", C["fam"])):
    g = A[A.scheme == sc].set_index("features").loc[order]
    ax.barh(y + off, g.mae_mean, 0.38, xerr=g.mae_sd, color=col, label={"random": "random split", "family": "held-out families"}[sc], capsize=2)
    for yy, v in zip(y + off, g.mae_mean):
        ax.text(v + 0.5, yy, f"{v:.1f}", va="center", fontsize=7)
ax.set_yticks(y); ax.set_yticklabels([names[o] for o in order]); ax.invert_yaxis()
ax.set_xlabel("Y2D MAE (N/m), LightGBM, 5 seeds"); ax.set_title("Task A: which feature groups carry the signal")
ax.legend(frameon=False)
save(fig, "fig2_taskA_feature_ablation.png")

# ---------------------------------------------------------------- fig3
la = json.load(open(RD / "task_A_run_info.json"))["leakage_first_target"]
lb = json.load(open(RD / "taskB_run_info.json"))["leakage"]["binding_energy_zscan"]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
keys = [("chemsys_in_train", "chemical system"), ("formula_in_train", "formula"), ("family_in_train", "structure family")]
for j, (sc, lab) in enumerate((("random", "random"), ("chemsys", "grouped by\nchemical system"), ("family", "grouped by\nfamily"))):
    for i, (k, kl) in enumerate(keys):
        ax[0].bar(j + (i - 1) * 0.26, la[sc][k], 0.26, color=["#6fa8dc", "#9fc5e8", "#e6a86c"][i], label=kl if j == 0 else None)
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(["random", "grouped by\nchemical system", "grouped by\nfamily"]); ax[0].set_ylim(0, 1.08)
ax[0].set_title("Task A (C2DB monolayers)"); ax[0].set_ylabel("share of test materials whose group is also in training"); ax[0].legend(frameon=False, fontsize=8)
keysb = [("monolayer_in_train", "monolayer"), ("family_in_train", "structure family"), ("chemsys_in_train", "chemical system")]
for j, sc in enumerate(("random", "monolayer", "family")):
    for i, (k, kl) in enumerate(keysb):
        ax[1].bar(j + (i - 1) * 0.26, lb[sc][k], 0.26, color=["#2a6fbb", "#e6a86c", "#9fc5e8"][i], label=kl if j == 0 else None)
ax[1].set_xticks(range(3)); ax[1].set_xticklabels(["random", "grouped by\nmonolayer", "grouped by\nfamily"]); ax[1].set_ylim(0, 1.08)
ax[1].set_title("Task B (BiDB bilayers)"); ax[1].legend(frameon=False, fontsize=8)
fig.suptitle("Why random splits flatter the models: overlap between test and training groups")
save(fig, "fig3_leakage.png", rect=[0, 0, 1, 0.95])

# ---------------------------------------------------------------- fig4
ts = pd.read_csv(RD / "taskB_summary_seeds.csv")
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
for a, (t, unit, lab) in zip(ax, (("binding_energy_zscan", "meV/A^2", "binding energy"), ("distance", "A", "interlayer gap"))):
    s = ["random", "monolayer", "family"]
    bars = [("mean predictor", "mean_arith", "none", C["mean"]), ("LightGBM, monolayer features", "gradient_boosting", "mono_basic", "#9fc5e8"),
            ("LightGBM, + C2DB stiffness", "gradient_boosting", "mono_stiffness", C["gb"])]
    for i, (bl, m, fs, col) in enumerate(bars):
        g = ts[(ts.target == t) & (ts.model == m) & (ts.features == fs)].set_index("scheme").loc[s]
        a.bar(np.arange(3) + (i - 1) * 0.26, g.mae_mean, 0.26, yerr=g.mae_sd, color=col, label=bl, capsize=2)
        for x, v in zip(np.arange(3) + (i - 1) * 0.26, g.mae_mean):
            a.text(x, v * 1.02, f"{v:.2f}" if v < 1 else f"{v:.1f}", ha="center", fontsize=7)
    a.set_xticks(range(3)); a.set_xticklabels(["random split", "new monolayers", "new families"]); a.set_ylabel(f"MAE ({unit})"); a.set_title(lab)
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9)
fig.suptitle("Task B: error by split scheme (mean +/- sd over 5 seeds)")
save(fig, "fig4_taskB_schemes.png", rect=[0, 0.07, 1, 0.95])

# ---------------------------------------------------------------- fig5
F = pd.read_csv(RD / "taskB_fold_metrics.csv")
F = F[(F.tag == "main") & (F.seed == 42) & (F.target == "binding_energy_zscan")]
fig, ax = plt.subplots(figsize=(7.5, 4.3))
for j, sc in enumerate(["random", "monolayer", "family"]):
    g = F[(F.scheme == sc) & (F.model == "gradient_boosting") & (F.features == "mono_stiffness")].sort_values("fold")
    b = F[(F.scheme == sc) & (F.model == "mean_arith")].sort_values("fold")
    ax.scatter([j - 0.12] * 5, g.mae, color=C["gb"], s=40, label="LightGBM, one dot per fold" if j == 0 else None, zorder=3)
    ax.scatter([j + 0.12] * 5, b.mae, color=C["mean"], s=40, marker="s", label="mean predictor, same folds" if j == 0 else None, zorder=3)
    for fg, fb in zip(g.mae, b.mae):
        ax.plot([j - 0.12, j + 0.12], [fg, fb], color="#cccccc", lw=0.8, zorder=1)
ax.set_xticks(range(3)); ax.set_xticklabels(["random split", "new monolayers", "new families"]); ax.set_ylabel("binding-energy MAE (meV/A^2)")
ax.set_title("Task B: per-fold spread (seed 42); grey lines join the same fold"); ax.legend(frameon=False, fontsize=8)
save(fig, "fig5_taskB_fold_spread.png")

# ---------------------------------------------------------------- fig6
G = pd.read_csv(RD / "mlip" / "calibration_gate.csv")
G = G[G.subset.isin(["c2db", "jarvis", "all"])]
mods = [("mace_mpa0_medium", "MACE-MPA-0"), ("mace_mp0_medium", "MACE-MP-0"), ("chgnet_0.3.0", "CHGNet")]
sub = [("c2db", "C2DB (150)"), ("jarvis", "JARVIS (184)"), ("all", "combined (334)")]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for i, (s, sl) in enumerate(sub):
    g = G[G.subset == s].set_index("model").loc[[m for m, _ in mods]]
    ax[0].bar(np.arange(3) + (i - 1) * 0.26, g.mae, 0.26, color=["#9fc5e8", "#6fa8dc", "#2a6fbb"][i], label=sl)
    ax[1].bar(np.arange(3) + (i - 1) * 0.26, g.spearman, 0.26, color=["#9fc5e8", "#6fa8dc", "#2a6fbb"][i], label=sl)
ax[0].axhline(12.93, color="k", ls="--", lw=1); ax[0].text(2.45, 14, "gate: 12.93", ha="right", fontsize=8); ax[0].set_yscale("log")
from matplotlib.ticker import ScalarFormatter
ax[0].set_yticks([10, 20, 30, 50, 80]); ax[0].yaxis.set_major_formatter(ScalarFormatter()); ax[0].minorticks_off()
ax[1].axhline(0.9, color="k", ls="--", lw=1); ax[1].text(2.45, 0.91, "gate: 0.90", ha="right", fontsize=8); ax[1].set_ylim(0.6, 1.0)
for a, yl, t in ((ax[0], "Y2D MAE (N/m), log scale", "error (all computed structures)"), (ax[1], "Spearman correlation", "ranking")):
    a.set_xticks(range(3)); a.set_xticklabels([m for _, m in mods]); a.set_ylabel(yl); a.set_title(t)
ax[0].legend(frameon=False, fontsize=8)
fig.suptitle("ML-potential stiffness calibration against the pre-registered gate")
save(fig, "fig6_mlip_gate.png", rect=[0, 0, 1, 0.95])
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
