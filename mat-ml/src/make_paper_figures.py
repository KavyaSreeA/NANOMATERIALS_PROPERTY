"""Publication figures for the manuscript: per-property family/random MAE ratio (C2DB + JARVIS replication) and the tuning/GNN robustness summary.
python -m src.make_paper_figures"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import data as D

cfg = D.load_config()
rd = D.ROOT / cfg["paths"]["results_dir"]
fd = rd / "figures"
fd.mkdir(exist_ok=True)
BLUE, ORANGE, GREY = "#2a6fbb", "#d97b29", "#888888"


def forest():
    parts = [(pd.read_csv(rd / "panel_summary.csv"), "C2DB", BLUE)]
    if (rd / "panel_summary_jarvis.csv").exists():
        parts.append((pd.read_csv(rd / "panel_summary_jarvis.csv"), "JARVIS-2D (replication)", ORANGE))
    fig, axes = plt.subplots(1, len(parts), figsize=(5.2 * len(parts) + 1, 5.2), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (T, name, col) in zip(axes, parts):
        T = T.sort_values("ratio_family").reset_index(drop=True)
        y = np.arange(len(T))
        ax.errorbar(T.ratio_family, y, xerr=[T.ratio_family - T.ratio_family_lo, T.ratio_family_hi - T.ratio_family], fmt="o", color=col, capsize=2, label="family-grouped / random")
        ax.scatter(T.ratio_chemsys, y, marker="s", s=22, color=GREY, label="chemical-system-grouped / random", zorder=3)
        ax.axvline(1, color="#bbbbbb", lw=.8)
        ax.set_yticks(y); ax.set_yticklabels(T.target, fontsize=8)
        ax.set_xlabel("MAE ratio vs random CV (95% cluster-bootstrap CI)"); ax.set_title(name)
    axes[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(fd / "fig8_family_penalty_panel.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def robustness():
    A = json.load(open(rd / "phaseA_cgcnn_vs_lgbm.json"))
    C = json.load(open(rd / "phaseC_tuning.json"))
    B = json.load(open(rd / "phaseC_cluster_bootstrap.json"))
    return A, C, B


if __name__ == "__main__":
    forest()
    print("wrote fig8_family_penalty_panel.png")
