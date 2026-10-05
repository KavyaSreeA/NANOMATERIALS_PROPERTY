"""Out-of-fold predictions, parity plots and residual plots for the primary models; reproduces the reported fold metrics as a check.

  python -m src.diagnostics

Task A : LightGBM on all 141 features, targets Y2D (ln-fitted) and Poisson ratio, schemes random / chemsys / family, seed 42.
Task B : LightGBM on `mono_stiffness`, targets binding_energy_zscan and distance (ln-fitted), schemes random / monolayer / family, seed 42.
External: JARVIS parity from results/external_ci_predictions.csv (LightGBM, all features; seed-averaged predictions).
Same splits and seeds as the main runs, so per-fold MAE must equal the values in task_A_fold_metrics.csv / taskB_fold_metrics.csv (checked and written to diagnostics/reproduction_check.json).
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.metrics import r2_score

from . import data as D
from . import taskb as T
from .features import build_features
from .models import get_models
from .splits import make_splits
from .train import fwd, inv

SEED = 42


def oof(df, Xmodel, Xsplit, y, scheme, kind, cfg, k=5):
    splits = make_splits(scheme, df, Xsplit, y, k, cfg["cv"]["n_clusters"], SEED)
    model = get_models(cfg, SEED, ["gradient_boosting"])["gradient_boosting"]
    pred, fold = np.full(len(y), np.nan), np.full(len(y), -1)
    for f, (tr, te) in enumerate(splits):
        m = clone(model).fit(Xmodel.iloc[tr], fwd(y[tr], kind))
        pred[te] = inv(m.predict(Xmodel.iloc[te]), kind)
        fold[te] = f
    return pred, fold


def parity_resid(d, title, unit, fname, log=False, schemes=None):
    schemes = schemes or list(d.scheme.unique())
    fig, ax = plt.subplots(3, len(schemes), figsize=(4.6 * len(schemes), 12), squeeze=False)
    for j, sc in enumerate(schemes):
        g = d[d.scheme == sc]
        y, p = g.y.to_numpy(), g.pred.to_numpy()
        mae = np.abs(p - y).mean()
        r2 = r2_score(y, p)
        rho = spearmanr(y, p)[0]
        a = ax[0, j]
        a.scatter(y, p, s=5, alpha=.25, c=g.fold, cmap="viridis")
        lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
        if log:
            lo = max(lo, 0.5)
            a.set_xscale("log"); a.set_yscale("log")
        a.plot([lo, hi], [lo, hi], "k-", lw=.8)
        a.set_xlim(lo, hi); a.set_ylim(lo, hi)
        a.set_xlabel(f"reference ({unit})"); a.set_ylabel(f"out-of-fold prediction ({unit})")
        a.set_title(f"{sc}\nMAE {mae:.3g}  R2 {r2:.3f}  rho {rho:.3f}", fontsize=10)
        res = p - y
        q1, q99 = np.percentile(res, [1, 99])
        b = ax[1, j]
        b.scatter(p, res, s=5, alpha=.25, c=g.fold, cmap="viridis")
        b.axhline(0, c="k", lw=.8)
        if log:
            b.set_xscale("log")
        b.set_ylim(min(q1, -abs(q99)) * 1.1, max(q99, abs(q1)) * 1.1)
        b.set_xlabel(f"prediction ({unit})"); b.set_ylabel(f"residual = prediction - reference ({unit})")
        c = ax[2, j]
        c.hist(np.clip(res, q1, q99), bins=50)
        c.axvline(0, c="k", lw=.8)
        c.set_xlabel(f"residual ({unit}), clipped to 1-99 percentile"); c.set_ylabel("count")
        c.set_title(f"mean {res.mean():.3g}, median {np.median(res):.3g}, sd {res.std():.3g}", fontsize=9)
    fig.suptitle(title + "   (colour = CV fold)", fontsize=12)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)


def main():
    cfg = D.load_config()
    out = D.ROOT / cfg["paths"]["results_dir"] / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    checks = {}
    # ------------------------------------------------------------------ Task A
    c2, _ = D.clean_c2db(D.load_c2db(cfg), cfg)
    recA = []
    ref_a = pd.read_csv(D.ROOT / cfg["paths"]["results_dir"] / "task_A_fold_metrics.csv")
    for target, kind, unit, log in (("Y2D", "log", "N/m", True), ("poisson", "none", "dimensionless", False)):
        df = D.restrict_for_target(c2, target, cfg)
        X, comp = build_features(df, cfg)
        y = df[target].to_numpy(float)
        rows = []
        schemes = ["random", "chemsys", "family"]
        for sc in schemes:
            pred, fold = oof(df, X, X[comp], y, sc, kind, cfg)
            r = pd.DataFrame({"uid": df.uid.values, "formula": df.reduced_formula.values, "family": df.family.values, "chemsys": df.chemsys.values,
                              "target": target, "scheme": sc, "fold": fold, "y": y, "pred": pred})
            rows.append(r)
            for f in range(5):
                m = r[r.fold == f]
                ref = ref_a[(ref_a.target == target) & (ref_a.scheme == sc) & (ref_a.model == "gradient_boosting") & (ref_a.fold == f)].mae.iloc[0]
                checks[f"A|{target}|{sc}|fold{f}"] = {"recomputed_mae": float(np.abs(m.pred - m.y).mean()), "reported_mae": float(ref)}
        d = pd.concat(rows)
        recA.append(d)
        parity_resid(d, f"Task A, {target}: LightGBM, 141 features, seed {SEED}", unit, str(out / f"taskA_{target}_parity_residuals.png"), log=log, schemes=schemes)
    pd.concat(recA).to_csv(out / "taskA_oof_predictions.csv", index=False)
    # ------------------------------------------------------------------ Task B
    bi = T.load_table(D.load_config())
    X_all = T.feature_matrices(bi, cfg)
    ref_b = pd.read_csv(D.ROOT / cfg["paths"]["results_dir"] / "taskB_fold_metrics.csv")
    ref_b = ref_b[(ref_b.tag == "main") & (ref_b.seed == SEED) & (ref_b.features == T.PRIMARY_FS) & (ref_b.model == "gradient_boosting")]
    recB = []
    for target, unit, log in (("binding_energy_zscan", "meV/A^2", False), ("distance", "A", False)):
        ok, _ = T.valid_mask(bi[target], "upper")
        d_ = bi[ok].reset_index(drop=True)
        Xs = X_all[T.PRIMARY_FS][ok].reset_index(drop=True)
        y = d_[target].to_numpy(float)
        rows = []
        schemes = ["random", "monolayer", "family"]
        for sc in schemes:
            pred, fold = oof(d_, Xs, Xs, y, sc, "log", cfg)
            r = pd.DataFrame({"uid": d_.uid.values, "monolayer_uid": d_.monolayer_uid.values, "family": d_.family.values, "target": target, "scheme": sc,
                              "fold": fold, "y": y, "pred": pred})
            rows.append(r)
            for f in range(5):
                m = r[r.fold == f]
                ref = ref_b[(ref_b.target == target) & (ref_b.scheme == sc) & (ref_b.fold == f)].mae.iloc[0]
                checks[f"B|{target}|{sc}|fold{f}"] = {"recomputed_mae": float(np.abs(m.pred - m.y).mean()), "reported_mae": float(ref)}
        d = pd.concat(rows)
        recB.append(d)
        parity_resid(d, f"Task B, {target}: LightGBM, mono_stiffness features, seed {SEED}", unit, str(out / f"taskB_{target}_parity_residuals.png"), log=False, schemes=schemes)
    pd.concat(recB).to_csv(out / "taskB_oof_predictions.csv", index=False)
    # ------------------------------------------------------------------ external JARVIS
    p = pd.read_csv(D.ROOT / cfg["paths"]["results_dir"] / "external_ci_predictions.csv")
    col = "pred_gradient_boosting_all"
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
    for a, (name, g) in zip(ax, (("all 186 JARVIS materials", p), ("unseen chemical system (55)", p[p.unseen_chemsys]))):
        a.scatter(g.y_ref, g[col], s=14, alpha=.7)
        lim = [1, 1000]
        a.plot(lim, lim, "k-", lw=.8); a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(lim); a.set_ylim(lim)
        a.set_xlabel("JARVIS Y2D (N/m)"); a.set_ylabel("prediction, model trained on C2DB (N/m)")
        a.set_title(f"{name}\nMAE {np.abs(g[col] - g.y_ref).mean():.1f} N/m, Spearman {spearmanr(g.y_ref, g[col])[0]:.2f}", fontsize=10)
    fig.suptitle("External check: LightGBM (all features), seed-averaged")
    fig.tight_layout(); fig.savefig(out / "external_jarvis_parity.png", dpi=120); plt.close(fig)
    # ------------------------------------------------------------------ reproduction check
    diff = np.array([abs(v["recomputed_mae"] - v["reported_mae"]) for v in checks.values()])
    summary = {"n_folds_checked": len(checks), "max_abs_diff_mae": float(diff.max()), "all_within_1e-6": bool(diff.max() < 1e-6)}
    json.dump({"summary": summary, "folds": checks}, open(out / "reproduction_check.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
