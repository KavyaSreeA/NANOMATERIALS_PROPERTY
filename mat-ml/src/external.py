"""JARVIS external check for Y2D: multi-seed, feature-set variants, bootstrap CIs, label-agreement ceiling.

  python -m src.external --seeds 42 43 44 45 46

Train on ALL clean C2DB rows, predict the 186 usable JARVIS rows. Subsets of JARVIS:
  all              every usable JARVIS row
  unseen_chemsys   chemical system never occurs in C2DB (clean) -> no material-level overlap
  matched          JARVIS row matched to a C2DB row (same reduced formula and atom count, in-plane a,b within 3%);
                   these are the SAME material in both databases, so a model trained on C2DB has seen it. Used only
                   to measure label agreement (C2DB label vs JARVIS label), not model skill.

Seeds change only the model's own randomness (the data are fixed), so the seed std is small by construction. The
uncertainty that matters is the 186-row JARVIS sample, so a bootstrap over JARVIS rows (on the seed-averaged
prediction) gives 95% intervals.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score

from . import data as D
from .features import build_features
from .models import get_models
from .robustness import feature_sets

TOL = 0.03


def match_materials(jar: pd.DataFrame, c2: pd.DataFrame) -> pd.Series:
    """For each JARVIS row, index of the best-matching C2DB row (or -1)."""
    out = np.full(len(jar), -1)
    groups = {k: g for k, g in c2.groupby("reduced_formula")}
    for i, r in enumerate(jar.itertuples()):
        g = groups.get(r.reduced_formula)
        if g is None:
            continue
        g = g[g.nat == r.nat]
        if g.empty:
            continue
        ja, jb = sorted([r.a, r.b])
        ca = np.sort(g[["a", "b"]].to_numpy(), axis=1)
        err = np.abs(ca[:, 0] - ja) / ja + np.abs(ca[:, 1] - jb) / jb
        k = int(np.argmin(err))
        if abs(ca[k, 0] - ja) / ja <= TOL and abs(ca[k, 1] - jb) / jb <= TOL:
            out[i] = g.index[k]
    return pd.Series(out, index=jar.index)


def metrics(y, p):
    return {"mae": mean_absolute_error(y, p), "r2": r2_score(y, p),
            "r2_log": r2_score(np.log(y), np.log(np.clip(p, 1e-6, None))),
            "mae_log": mean_absolute_error(np.log(y), np.log(np.clip(p, 1e-6, None))),
            "spearman": spearmanr(y, p)[0]}


def boot_ci(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    stats = {"mae": [], "r2_log": []}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        stats["mae"].append(mean_absolute_error(y[i], p[i]))
        stats["r2_log"].append(r2_score(np.log(y[i]), np.log(np.clip(p[i], 1e-6, None))))
    return {f"{k}_lo": np.percentile(v, 2.5) for k, v in stats.items()} | {f"{k}_hi": np.percentile(v, 97.5) for k, v in stats.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--sets", nargs="+", default=["all", "composition", "structure"])
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = D.load_config(args.config)
    t0 = time.time()
    kind = cfg["targets"]["taskA"]["Y2D"]
    c2, _ = D.clean_c2db(D.load_c2db(cfg), cfg)
    jar, _ = D.clean_jarvis(D.load_jarvis(cfg))
    X, _ = build_features(c2, cfg)
    Xj, _ = build_features(jar, cfg)
    sets = feature_sets(X)
    yc = np.log(c2.Y2D.to_numpy(float))
    yj = jar.Y2D.to_numpy(float)
    unseen = ~jar.chemsys.isin(set(c2.chemsys)).to_numpy()
    mi = match_materials(jar, c2)
    matched = (mi >= 0).to_numpy()
    masks = {"all": np.ones(len(jar), bool), "unseen_chemsys": unseen, "matched": matched}
    print(f"JARVIS rows {len(jar)}; unseen chemsys {unseen.sum()}; matched to a C2DB material {matched.sum()}", flush=True)

    # label-agreement ceiling: C2DB label vs JARVIS label on matched materials
    ceil = {}
    if matched.sum() > 2:
        ycl = c2.loc[mi[matched], "Y2D"].to_numpy(float)
        ceil = metrics(yj[matched], ycl) | {"n": int(matched.sum()), "median_ratio_jarvis_over_c2db": float(np.median(yj[matched] / ycl))}
        print("label agreement on matched materials:", {k: round(float(v), 3) for k, v in ceil.items()}, flush=True)

    rows, preds = [], {}
    for mname in ("gradient_boosting", "random_forest"):
        for fset in args.sets:
            if mname == "random_forest" and fset != "all":
                continue
            cols = sets[fset]
            P = []
            for seed in args.seeds:
                m = clone(get_models(cfg, seed, [mname])[mname]).fit(X[cols], yc)
                p = np.exp(m.predict(Xj[cols]))
                P.append(p)
                for sub, mask in masks.items():
                    rows.append({"model": mname, "features": fset, "seed": seed, "subset": sub,
                                 "n": int(mask.sum()), **metrics(yj[mask], p[mask])})
            preds[(mname, fset)] = np.mean(P, axis=0)
            print(f"{mname} {fset} done ({time.time() - t0:.0f}s)", flush=True)

    d = pd.DataFrame(rows)
    rd = D.ROOT / cfg["paths"]["results_dir"]
    d.to_csv(rd / "external_per_seed.csv", index=False)
    agg = d.groupby(["model", "features", "subset"]).agg(
        n=("n", "first"), mae=("mae", "mean"), mae_sd=("mae", "std"), r2=("r2", "mean"), r2_log=("r2_log", "mean"),
        r2_log_sd=("r2_log", "std"), spearman=("spearman", "mean")).reset_index()
    ci = []
    for (mname, fset), p in preds.items():
        for sub, mask in masks.items():
            if mask.sum() > 5:
                ci.append({"model": mname, "features": fset, "subset": sub, **boot_ci(yj[mask], p[mask])})
    agg = agg.merge(pd.DataFrame(ci), on=["model", "features", "subset"], how="left")
    agg.to_csv(rd / "external_summary.csv", index=False)

    from .train import md_table
    show = agg[["model", "features", "subset", "n", "mae", "mae_sd", "mae_lo", "mae_hi", "r2_log", "r2_log_lo", "r2_log_hi", "spearman"]]
    text = [f"# JARVIS external check, seeds {args.seeds}", "",
            "Fit on all clean C2DB rows (ln Y2D), predict JARVIS Y2D (N/m). mae_sd = std across model seeds; "
            "[lo, hi] = 95% bootstrap interval over JARVIS rows on the seed-averaged prediction.", "",
            "## Models vs JARVIS", "", md_table(show), "",
            "## Label agreement: C2DB label vs JARVIS label on matched materials (not model skill)", "",
            md_table(pd.DataFrame([ceil])) if ceil else "no matches", "", f"runtime {time.time() - t0:.0f}s"]
    (rd / "external_report.md").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
