"""Multi-seed runs + feature ablation + null baselines for Task A (Y2D).

  python -m src.robustness --seeds 42 43 44 45 46

Feature sets: all | composition (Magpie + n_elements) | structure (a, b, gamma, area/atom, nat, thickness, spg).
Null baselines (no features): global_mean, family_mean (mean of ln Y2D over training rows of the same structure
family; unseen families fall back to the global training mean), chemsys_mean (same for chemical system).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.base import clone

from . import data as D
from .features import build_features, prototype_features
from .models import get_models
from .splits import make_splits
from .train import fwd, inv, md_table, score

STRUCT = ["a", "b", "gamma", "area_per_atom", "nat", "z_extent", "n_layers", "spg_number"]
GEOM = [c for c in STRUCT if c not in ("spg_number", "n_layers")]  # n_layers is constant (1)


def feature_sets(X: pd.DataFrame) -> dict:
    proto = [c for c in X.columns if c == "lgnum" or c.startswith(("lg_", "anon_"))]
    lg = [c for c in proto if c == "lgnum" or c.startswith("lg_")]
    anon = [c for c in proto if c.startswith("anon_")]
    base = [c for c in X.columns if c not in proto]
    comp = [c for c in base if c.startswith("mp_")] + ["n_elements"]
    return {"all": base, "composition": comp, "structure": STRUCT,
            # prototype features (layer group, anonymous formula)
            "all_plus_layergroup": base + lg, "all_plus_anon": base + anon, "all_plus_prototype": base + proto,
            "composition_plus_prototype": comp + proto, "prototype_only": proto,
            # spg_number ablation
            "geometry_no_spg": GEOM,                                   # a, b, gamma, area/atom, nat, thickness
            "spg_only": ["spg_number"],
            "all_no_spg": [c for c in base if c != "spg_number"],
            "composition_plus_spg": comp + ["spg_number"]}


def group_mean_predict(df, tr, te, col, yt):
    m = pd.Series(yt[tr]).groupby(df[col].iloc[tr].to_numpy()).mean()
    g = yt[tr].mean()
    return df[col].iloc[te].map(m).fillna(g).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--schemes", nargs="+", default=["random", "chemsys", "family"])
    ap.add_argument("--rf-schemes", nargs="+", default=["random", "chemsys", "family"])
    ap.add_argument("--no-rf", action="store_true")
    ap.add_argument("--sets", nargs="+", default=["all", "composition", "structure"])
    ap.add_argument("--prefix", default="robustness", help="output file prefix")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--prototype", action="store_true", help="append layer-group and anonymous-formula columns")
    ap.add_argument("--proto-min-count", type=int, default=10)
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = D.load_config(args.config)
    t0 = time.time()
    target, kind = "Y2D", cfg["targets"]["taskA"]["Y2D"]
    df, _ = D.clean_c2db(D.load_c2db(cfg), cfg)
    X, comp_cols = build_features(df, cfg)
    if args.prototype:
        X = pd.concat([X, prototype_features(df, args.proto_min_count)], axis=1)
    sets = feature_sets(X)
    print({k: len(v) for k, v in sets.items()}, flush=True)
    y = df[target].to_numpy(float)
    yt = fwd(y, kind)
    n_splits = cfg["cv"]["n_splits"]
    rows = []

    def add(seed, scheme, model, fset, f, tr, te, pt):
        rows.append({"seed": seed, "scheme": scheme, "model": model, "features": fset, "fold": f,
                     "n_test": len(te), **score(y[te], inv(pt, kind), yt[te], pt)})

    for seed in args.seeds:
        for scheme in args.schemes:
            splits = make_splits(scheme, df, X[comp_cols], y, n_splits, cfg["cv"]["n_clusters"], seed)
            for f, (tr, te) in enumerate(splits):
                if args.no_baselines:
                    continue
                add(seed, scheme, "global_mean", "none", f, tr, te, np.full(len(te), yt[tr].mean()))
                add(seed, scheme, "family_mean", "none", f, tr, te, group_mean_predict(df, tr, te, "family", yt))
                add(seed, scheme, "chemsys_mean", "none", f, tr, te, group_mean_predict(df, tr, te, "chemsys", yt))
            for mname in ["gradient_boosting"] + ([] if args.no_rf or scheme not in args.rf_schemes else ["random_forest"]):
                model = get_models(cfg, seed, [mname])[mname]
                for fset in args.sets:
                    cols = sets[fset]
                    if mname == "random_forest" and fset != "all":
                        continue
                    for f, (tr, te) in enumerate(splits):
                        m = clone(model).fit(X.iloc[tr][cols], yt[tr])
                        add(seed, scheme, mname, fset, f, tr, te, m.predict(X.iloc[te][cols]))
            print(f"seed {seed} scheme {scheme} done  ({time.time() - t0:.0f}s)", flush=True)

    d = pd.DataFrame(rows)
    rd = D.ROOT / cfg["paths"]["results_dir"]
    d.to_csv(rd / f"{args.prefix}_fold_metrics.csv", index=False)
    # one number per (seed, scheme, model, features): mean over folds; then mean/std over seeds
    per_seed = d.groupby(["seed", "scheme", "model", "features"]).agg(
        mae=("mae", "mean"), r2=("r2", "mean"), r2_log=("r2_t", "mean"), mae_log=("mae_t", "mean")).reset_index()
    per_seed.to_csv(rd / f"{args.prefix}_per_seed.csv", index=False)
    agg = per_seed.groupby(["model", "features", "scheme"]).agg(
        mae_mean=("mae", "mean"), mae_sd=("mae", "std"), r2_mean=("r2", "mean"), r2_sd=("r2", "std"),
        r2_log_mean=("r2_log", "mean"), r2_log_sd=("r2_log", "std"), n_seeds=("seed", "nunique")).reset_index()
    agg.to_csv(rd / f"{args.prefix}_summary.csv", index=False)

    order = {"random": 0, "chemsys": 1, "family": 2}
    tbl = agg.assign(o=agg.scheme.map(order)).sort_values(["model", "features", "o"])
    show = tbl[["model", "features", "scheme", "mae_mean", "mae_sd", "r2_mean", "r2_log_mean", "r2_log_sd"]]
    # paired family/random MAE ratio per seed
    pv = per_seed.pivot_table(index=["seed", "model", "features"], columns="scheme", values="mae").reset_index()
    for s in ("family", "chemsys"):
        if s in pv and "random" in pv:
            pv[f"{s}/random"] = pv[s] / pv["random"]
    spec = {f"{s}_over_random_{stat}": (f"{s}/random", stat)
            for s in ("family", "chemsys") if f"{s}/random" in pv for stat in ("mean", "std", "min")}
    ratio = pv.groupby(["model", "features"]).agg(**spec).reset_index() if spec else pv[["model", "features"]].drop_duplicates()
    ratio.to_csv(rd / f"{args.prefix}_ratios.csv", index=False)
    text = [f"# Robustness: Y2D, seeds {args.seeds}, {n_splits}-fold", "",
            "MAE in N/m; r2_log is R2 of ln(Y2D) (stable under extrapolation). sd = std across seeds of the fold-mean.", "",
            "## Summary by model / features / scheme", "", md_table(show), "",
            "## Paired MAE ratios across seeds (grouped / random)", "", md_table(ratio), "",
            f"runtime {time.time() - t0:.0f}s"]
    (rd / f"{args.prefix}_report.md").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
