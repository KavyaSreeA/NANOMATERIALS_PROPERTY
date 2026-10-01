"""Run baselines.   python -m src.train --task A      (C2DB, JARVIS external check)
                   python -m src.train --task B      (BiDB; needs files that are not in this checkout)
Outputs go to results/: fold metrics, summary, comparison table, leakage stats, run info.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import data as D
from .features import build_features
from .models import MODEL_NAMES, get_models
from .splits import SCHEMES, leakage_stats, make_splits


def fwd(y, kind):
    return np.log(y) if kind == "log" else np.asarray(y, float)


def inv(z, kind):
    return np.exp(z) if kind == "log" else z


def score(y, p, yt, pt) -> dict:
    """Metrics in original units (N/m for Y2D) and in the fitting (transformed) space."""
    return {"mae": mean_absolute_error(y, p), "rmse": mean_squared_error(y, p) ** 0.5, "r2": r2_score(y, p),
            "mae_t": mean_absolute_error(yt, pt), "r2_t": r2_score(yt, pt)}


def cross_validate(df, X, comp_cols, target, kind, schemes, models, cfg, args):
    seed, k = cfg["seed"], args.n_splits or cfg["cv"]["n_splits"]
    y = df[target].to_numpy(float)
    yt = fwd(y, kind)
    rows, leak = [], {}
    for scheme in schemes:
        splits = make_splits(scheme, df, X[comp_cols], y, k, cfg["cv"]["n_clusters"], seed)
        leak[scheme] = leakage_stats(splits, df)
        for mname, model in models.items():
            for f, (tr, te) in enumerate(splits):
                m = clone(model).fit(X.iloc[tr], yt[tr])
                pt = m.predict(X.iloc[te])
                rows.append({"target": target, "scheme": scheme, "model": mname, "fold": f,
                             "n_train": len(tr), "n_test": len(te),
                             **score(y[te], inv(pt, kind), yt[te], pt)})
            print(f"  {target:<18}{scheme:<10}{mname:<18}"
                  f"MAE={np.mean([r['mae'] for r in rows[-k:]]):.4g}  R2={np.mean([r['r2'] for r in rows[-k:]]):.3f}",
                  flush=True)
    return rows, leak


def external_check(df_tr, X_tr, jar, X_jar, target, kind, models, comp_chemsys):
    """Fit on all labelled C2DB rows, predict JARVIS. Also report the chemsys-disjoint subset."""
    rows = []
    y_tr = fwd(df_tr[target].to_numpy(float), kind)
    disjoint = ~jar.chemsys.isin(comp_chemsys).to_numpy()
    for mname, model in models.items():
        m = clone(model).fit(X_tr, y_tr)
        pt = m.predict(X_jar)
        y = jar[target].to_numpy(float)
        yt = fwd(y, kind)
        for name, mask in (("external_jarvis_all", np.ones(len(jar), bool)),
                           ("external_jarvis_unseen_chemsys", disjoint)):
            if mask.sum() > 2:
                rows.append({"target": target, "scheme": name, "model": mname, "fold": 0, "n_train": len(df_tr),
                             "n_test": int(mask.sum()),
                             **score(y[mask], inv(pt[mask], kind), yt[mask], pt[mask])})
    return rows


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(lines)


def write_outputs(task, folds: pd.DataFrame, leak: dict, info: dict, cfg):
    rd = D.ROOT / cfg["paths"]["results_dir"]
    rd.mkdir(exist_ok=True)
    folds.to_csv(rd / f"task_{task}_fold_metrics.csv", index=False)
    g = folds.groupby(["target", "scheme", "model"], sort=False)
    summ = g.agg(mae_mean=("mae", "mean"), mae_std=("mae", "std"), rmse_mean=("rmse", "mean"),
                 r2_mean=("r2", "mean"), r2_std=("r2", "std"), r2_log_mean=("r2_t", "mean"),
                 n_folds=("fold", "count")).reset_index()
    summ.to_csv(rd / f"task_{task}_summary.csv", index=False)
    # comparison: one row per target x model, schemes side by side
    cv = summ[~summ.scheme.str.startswith("external")]
    wide = cv.pivot_table(index=["target", "model"], columns="scheme", values=["mae_mean", "r2_mean"])
    order = [s for s in SCHEMES if s in cv.scheme.unique()]
    comp = pd.DataFrame(index=wide.index)
    for s in order:
        comp[f"MAE {s}"] = wide[("mae_mean", s)]
    for s in order:
        comp[f"R2 {s}"] = wide[("r2_mean", s)]
    if "random" in order:
        for s in order:
            if s != "random":
                comp[f"MAE x({s}/random)"] = wide[("mae_mean", s)] / wide[("mae_mean", "random")]
    comp = comp.reset_index()
    ext = summ[summ.scheme.str.startswith("external")][["target", "scheme", "model", "mae_mean", "rmse_mean", "r2_mean", "r2_log_mean"]]
    leak_df = pd.DataFrame(leak).T.reset_index().rename(columns={"index": "scheme"})
    text = [f"# Task {task} results (seed {cfg['seed']})", "",
            "Metrics are means over folds, in original units (N/m for Y2D). `x(s/random)` = MAE under scheme s divided by MAE under random CV.",
            "", "## Random vs cluster-stratified vs grouped", "", md_table(comp), "",
            "## Leakage: fraction of test rows whose key also occurs in training (mean over folds)", "",
            md_table(leak_df) if len(leak_df) else "n/a", ""]
    if len(ext):
        text += ["## External check (train on all labelled C2DB, test on JARVIS)", "", md_table(ext), ""]
    text += ["## Run info", "", "```json", json.dumps(info, indent=2, default=str), "```"]
    (rd / f"task_{task}_comparison.md").write_text("\n".join(text), encoding="utf-8")
    (rd / f"task_{task}_run_info.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    print("\n".join(text[:12]))
    print(f"\nWrote results to {rd}")


def run_task_a(cfg, args):
    t0 = time.time()
    targets = args.targets or list(cfg["targets"]["taskA"])
    c2, clog = D.clean_c2db(D.load_c2db(cfg, rebuild=args.rebuild), cfg)
    jar, jlog = D.clean_jarvis(D.load_jarvis(cfg))
    print(f"C2DB cleaning: {clog}\nJARVIS cleaning: {jlog}")
    schemes = args.schemes or cfg["cv"]["schemes"]
    models = get_models(cfg, cfg["seed"], args.models or MODEL_NAMES, args.quick)
    all_rows, all_leak, rowcounts = [], {}, {}
    for target in targets:
        kind = cfg["targets"]["taskA"][target]
        df = D.restrict_for_target(c2, target, cfg)
        jt = D.restrict_for_target(jar, target, cfg)
        rowcounts[target] = {"c2db_rows": len(df), "jarvis_rows": len(jt)}
        X, comp_cols = build_features(df, cfg)
        Xj, _ = build_features(jt, cfg)
        rows, leak = cross_validate(df, X, comp_cols, target, kind, schemes, models, cfg, args)
        all_rows += rows
        all_leak[target] = leak
        if not args.no_external:
            all_rows += external_check(df, X, jt, Xj, target, kind, models, set(df.chemsys))
    leak_flat = {s: v for s, v in all_leak[targets[0]].items()}
    info = {"task": "A", "seed": cfg["seed"], "c2db_cleaning": clog, "jarvis_cleaning": jlog, "rows_per_target": rowcounts,
            "leakage_first_target": leak_flat, "n_features": X.shape[1], "targets": {t: cfg["targets"]["taskA"][t] for t in targets},
            "convention": "n_layers=1 for all rows; Y2D (total) == Y2D_per_layer", "quick": args.quick,
            "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__,
                         "pandas": pd.__version__}, "runtime_s": round(time.time() - t0, 1)}
    write_outputs("A", pd.DataFrame(all_rows), leak_flat, info, cfg)


def run_task_b(cfg, args):
    t0 = time.time()
    try:
        b = D.load_bidb(cfg)
    except (FileNotFoundError, KeyError) as e:
        print(f"Task B cannot run: {e}", file=sys.stderr)
        sys.exit(2)
    targets = args.targets or list(cfg["targets"]["taskB"])
    schemes = [s for s in (args.schemes or ["random", "cluster", "chemsys", "monolayer"])]
    models = get_models(cfg, cfg["seed"], args.models or MODEL_NAMES, args.quick)
    all_rows, leak_flat, counts = [], {}, {}
    for target in targets:
        kind = cfg["targets"]["taskB"][target]
        df, log = D.clean_bidb(b, target, cfg)
        counts[target] = log
        print(f"BiDB cleaning for {target}: {log}")
        X, comp_cols = build_features(df, cfg)
        rows, leak = cross_validate(df, X, comp_cols, target, kind, schemes, models, cfg, args)
        all_rows += rows
        leak_flat = leak_flat or leak
    info = {"task": "B", "seed": cfg["seed"], "cleaning": counts, "n_layers": 2,
            "convention": "binding energy = per interface (one interface per bilayer), NOT divided by layers; unit unconfirmed",
            "runtime_s": round(time.time() - t0, 1), "status": "UNTESTED on real data"}
    write_outputs("B", pd.DataFrame(all_rows), leak_flat, info, cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["A", "B"], required=True)
    ap.add_argument("--targets", nargs="+")
    ap.add_argument("--schemes", nargs="+", choices=list(SCHEMES))
    ap.add_argument("--models", nargs="+", choices=list(MODEL_NAMES))
    ap.add_argument("--n-splits", type=int)
    ap.add_argument("--quick", action="store_true", help="fewer trees; smoke test only")
    ap.add_argument("--no-external", action="store_true")
    ap.add_argument("--rebuild", action="store_true", help="re-read C2DB from disk instead of cache")
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = D.load_config(args.config)
    (run_task_a if args.task == "A" else run_task_b)(cfg, args)


if __name__ == "__main__":
    main()
