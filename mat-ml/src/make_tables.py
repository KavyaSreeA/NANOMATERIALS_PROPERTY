"""Generate results/headline_metrics.md (MAE, RMSE, R2, ln R2, Spearman where available) directly from the saved result CSVs.

  python -m src.make_tables
"""
import pandas as pd

from . import data as D

cfg = D.load_config()
rd = D.ROOT / cfg["paths"]["results_dir"]


def md(df):
    c = list(df.columns)
    o = ["| " + " | ".join(c) + " |", "|" + "|".join("---" for _ in c) + "|"]
    for _, r in df.iterrows():
        o.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(o)


L = ["# Headline metrics (auto-generated from the saved result CSVs; `python -m src.make_tables`)", "",
     "MAE and RMSE are in original units (Task A: Y2D in N/m, Poisson ratio dimensionless; Task B: binding energy in meV/A^2, gap in A). "
     "`+/-` is the standard deviation over the five folds. ln R2 = R2 of the log-transformed target. Seed 42, 5-fold.", ""]
# ---- Task A
ta = pd.read_csv(rd / "task_A_summary.csv")
ta = ta[~ta.scheme.str.startswith("external")].copy()
ta["MAE"] = ta.apply(lambda r: f"{r.mae_mean:.2f} +/- {r.mae_std:.2f}", axis=1)
ta["R2"] = ta.apply(lambda r: f"{r.r2_mean:.3f} +/- {r.r2_std:.3f}", axis=1)
for t in ("Y2D", "poisson"):
    s = ta[ta.target == t][["scheme", "model", "MAE", "rmse_mean", "R2", "r2_log_mean"]].rename(columns={"rmse_mean": "RMSE", "r2_log_mean": "ln R2"})
    L += [f"## Task A: {t} (141 features)", "", md(s), ""]
# ---- Task B
tb = pd.read_csv(rd / "taskB_summary_folds.csv")
tb = tb[tb.rule == "upper"].copy()
tb["MAE"] = tb.apply(lambda r: f"{r.mae_mean:.2f} +/- {r.mae_std:.2f}", axis=1)
tb["R2"] = tb.apply(lambda r: f"{r.r2_mean:.3f} +/- {r.r2_std:.3f}", axis=1)
for t in ("binding_energy_zscan", "distance", "binding_energy_gs"):
    s = tb[tb.target == t][["scheme", "features", "model", "MAE", "rmse_mean", "R2", "r2_log_mean", "spearman_mean"]].rename(
        columns={"rmse_mean": "RMSE", "r2_log_mean": "ln R2", "spearman_mean": "Spearman"})
    L += [f"## Task B: {t}", "", md(s), ""]
# ---- five seeds
rob = pd.read_csv(rd / "robustness_summary.csv")
r = rob[(rob.model == "gradient_boosting") & (rob.features == "all")][["scheme", "mae_mean", "mae_sd", "r2_mean", "r2_log_mean"]]
L += ["## Task A, Y2D: LightGBM, five seeds (mean and std over seeds of the fold means)", "", md(r), ""]
ts = pd.read_csv(rd / "taskB_summary_seeds.csv")
r = ts[(ts.model == "gradient_boosting") & (ts.features == "mono_stiffness")][["target", "scheme", "mae_mean", "mae_sd", "r2_mean", "r2_log_mean", "spearman_mean"]]
L += ["## Task B: LightGBM on mono_stiffness, five seeds", "", md(r), ""]
(rd / "headline_metrics.md").write_text("\n".join(L), encoding="utf-8")
print("wrote", rd / "headline_metrics.md")
