"""Phase B analysis of the stored out-of-fold predictions, following results/panel_design.md.   python -m src.panel_analysis [--oof panel_oof]

Per target: pooled out-of-fold MAE (seed-averaged) for LightGBM and the three baselines under random / chemsys / family splits,
skill S = 1 - MAE/MAE_global-mean, skill retention R = S_scheme / S_random, label-only diagnostics D_family, D_chem (out-of-fold group-mean skill under RANDOM splits), ICC_family.
Tests H1-H5 with bootstrap over families (inside targets) and over targets (for the correlations).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import data as D

SCHEMES = ["random", "chemsys", "family"]
RNG = np.random.default_rng(0)
NB = 1000


def load_target(oof, t):
    """-> {scheme: {seed: dict(y, preds..., family, chemsys)}}"""
    out = {s: {} for s in SCHEMES}
    for p in glob.glob(str(oof / f"{t}__*__seed*.npz")):
        sc, sd = os.path.basename(p)[:-4].split("__")[1:]
        z = np.load(p, allow_pickle=True)
        out[sc][int(sd.replace("seed", ""))] = {k: z[k] for k in z.files}
    return out


def mae(a, b):
    return float(np.abs(a - b).mean())


def seed_avg_mae(runs, pred, idx=None):
    """MAE per seed (pooled over folds), averaged over seeds; idx = row subset (same ordering in every seed because splits only reorder folds)."""
    v = []
    for z in runs.values():
        y, p = z["y"], z[f"pred_{pred}"]
        v.append(mae(y if idx is None else y[idx], p if idx is None else p[idx]))
    return float(np.mean(v))


def icc1(y, g):
    s = pd.Series(y)
    grp = s.groupby(g)
    n_i, means = grp.size().to_numpy(), grp.mean().to_numpy()
    N, k = len(s), len(n_i)
    if k < 2 or N <= k:
        return np.nan
    ssb = (n_i * (means - s.mean()) ** 2).sum()
    ssw = ((s - grp.transform("mean")) ** 2).sum()
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - (n_i**2).sum() / N) / (k - 1)
    return float((msb - msw) / (msb + (n0 - 1) * msw))


def skill_retention(runs, scheme_runs):
    """R = (1 - MAE_s/MAE0_s) / (1 - MAE_rand/MAE0_rand); 'MAE0' = global-mean baseline of the same scheme."""
    def skill(r):
        return 1 - seed_avg_mae(r, "lgbm") / seed_avg_mae(r, "global")
    return skill(scheme_runs) / skill(runs["random"])


def boot_over_families(per, n=NB):
    """per: dict scheme -> dict seed -> arrays. Resample families (from the family-split run's labels; row order is identical across schemes)."""
    z0 = next(iter(per["random"].values()))
    fam = z0["family"]
    codes, uniq = pd.factorize(fam)
    rows_of = [np.flatnonzero(codes == i) for i in range(len(uniq))]
    out = {"ratio_family": [], "R_family": [], "R_chemsys": [], "ratio_chemsys": []}
    for _ in range(n):
        pick = RNG.integers(0, len(rows_of), len(rows_of))
        idx = np.concatenate([rows_of[i] for i in pick])
        m = {s: seed_avg_mae(per[s], "lgbm", idx) for s in SCHEMES}
        m0 = {s: seed_avg_mae(per[s], "global", idx) for s in SCHEMES}
        sk = {s: 1 - m[s] / m0[s] for s in SCHEMES}
        out["ratio_family"].append(m["family"] / m["random"])
        out["ratio_chemsys"].append(m["chemsys"] / m["random"])
        out["R_family"].append(sk["family"] / sk["random"])
        out["R_chemsys"].append(sk["chemsys"] / sk["random"])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default="panel_oof")
    args = ap.parse_args()
    cfg = D.load_config()
    rd = D.ROOT / cfg["paths"]["results_dir"]
    oof = D.ROOT / "cache" / args.oof
    meta = {t["target"]: t for t in json.load(open(rd / "panel_targets.json"))["included"]}
    rows = []
    for t in meta:
        per = load_target(oof, t)
        if any(len(per[s]) == 0 for s in SCHEMES):
            continue
        z = next(iter(per["random"].values()))
        y = z["y"]
        yt = np.log(y) if meta[t]["transform"] == "log" else y
        m = {s: seed_avg_mae(per[s], "lgbm") for s in SCHEMES}
        m0 = {s: seed_avg_mae(per[s], "global") for s in SCHEMES}
        sk = {s: 1 - m[s] / m0[s] for s in SCHEMES}
        d_fam = 1 - seed_avg_mae(per["random"], "family") / m0["random"]
        d_chem = 1 - seed_avg_mae(per["random"], "chemsys") / m0["random"]
        ci = boot_over_families(per, n=NB)
        rows.append({"target": t, "n": len(y), "transform": meta[t]["transform"], "n_families": meta[t]["n_families"], "n_seeds": len(per["random"]),
                     "MAE_random": m["random"], "MAE_chemsys": m["chemsys"], "MAE_family": m["family"], "MAE_mean_random": m0["random"],
                     "skill_random": sk["random"], "skill_chemsys": sk["chemsys"], "skill_family": sk["family"],
                     "R_family": sk["family"] / sk["random"], "R_family_lo": ci["R_family"][0], "R_family_hi": ci["R_family"][1],
                     "R_chemsys": sk["chemsys"] / sk["random"], "R_chemsys_lo": ci["R_chemsys"][0], "R_chemsys_hi": ci["R_chemsys"][1],
                     "ratio_family": m["family"] / m["random"], "ratio_family_lo": ci["ratio_family"][0], "ratio_family_hi": ci["ratio_family"][1],
                     "ratio_chemsys": m["chemsys"] / m["random"], "ratio_chemsys_lo": ci["ratio_chemsys"][0], "ratio_chemsys_hi": ci["ratio_chemsys"][1],
                     "D_family": d_fam, "D_chem": d_chem, "ICC_family": icc1(yt, z["family"]), "ICC_chemsys": icc1(yt, z["chemsys"]),
                     "kurtosis": float(pd.Series(yt).kurt())})
        print(f"  {t:<18} n={len(y):>6}  R_fam={rows[-1]['R_family']:.2f}  D_fam={d_fam:.2f}  ratio_fam={rows[-1]['ratio_family']:.2f}", flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(rd / "panel_summary.csv", index=False)
    res = {"n_targets": len(T), "H1_share_ratio_family_gt_1.05": float((T.ratio_family > 1.05).mean()), "H1_all_R_family_lt_1": bool((T.R_family < 1).all())}

    def corr(x, y, n=NB):
        rho = spearmanr(x, y)[0]
        bs = []
        for _ in range(n):
            i = RNG.integers(0, len(x), len(x))
            if len(set(np.asarray(x)[i])) > 2 and len(set(np.asarray(y)[i])) > 2:
                bs.append(spearmanr(np.asarray(x)[i], np.asarray(y)[i])[0])
        return float(rho), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

    def loto(x, y):
        """leave-one-target-out: linear fit of y on x vs intercept only; returns ratio of mean absolute errors."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        e1, e0 = [], []
        for i in range(len(x)):
            m = np.arange(len(x)) != i
            b, a = np.polyfit(x[m], y[m], 1)
            e1.append(abs(y[i] - (a + b * x[i])))
            e0.append(abs(y[i] - y[m].mean()))
        return float(np.mean(e1) / np.mean(e0))

    if len(T) >= 4:
        res["H2_spearman_D_family_vs_R_family"] = dict(zip(("rho", "ci95"), corr(T.D_family, T.R_family)))
        res["H3_LOTO_error_ratio_linear_vs_intercept"] = loto(T.D_family, T.R_family)
        res["H4_spearman_D_chem_vs_R_chemsys"] = dict(zip(("rho", "ci95"), corr(T.D_chem, T.R_chemsys)))
        res["H4_LOTO_error_ratio"] = loto(T.D_chem, T.R_chemsys)
        res["H5_spearman_ICC_family_vs_R_family"] = dict(zip(("rho", "ci95"), corr(T.ICC_family.fillna(0), T.R_family)))
        res["H5_covariates_vs_R_family"] = {c: float(spearmanr(T[c], T.R_family)[0]) for c in ("n", "n_families", "kurtosis")}
        res["secondary_spearman_D_family_vs_log_ratio_family"] = dict(zip(("rho", "ci95"), corr(T.D_family, np.log(T.ratio_family))))
        # axis classification (descriptive): which grouping removes more skill
        T["axis"] = np.where((T.R_family < T.R_chemsys - 0.05), "structure-family loss", np.where(T.R_chemsys < T.R_family - 0.05, "chemistry loss", "similar"))
        res["axis_counts"] = T.axis.value_counts().to_dict()
        T.to_csv(rd / "panel_summary.csv", index=False)
    json.dump(res, open(rd / "panel_hypotheses.json", "w"), indent=2)
    print(json.dumps(res, indent=2))
    # ---- figures
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for a, (xc, yc, xl, yl, lo, hi) in zip(ax, (("D_family", "R_family", "label-only diagnostic D_family\n(skill of a pure family-mean predictor, random CV)", "skill retained under family-grouped CV", "R_family_lo", "R_family_hi"),
                                               ("D_chem", "R_chemsys", "label-only diagnostic D_chem\n(skill of a pure chemical-system-mean predictor, random CV)", "skill retained under chemical-system-grouped CV", "R_chemsys_lo", "R_chemsys_hi"))):
        a.errorbar(T[xc], T[yc], yerr=[T[yc] - T[lo], T[hi] - T[yc]], fmt="o", color="#2a6fbb", capsize=2)
        for _, r in T.iterrows():
            a.annotate(r.target, (r[xc], r[yc]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        a.set_xlabel(xl); a.set_ylabel(yl); a.axhline(1, color="#bbbbbb", lw=.8)
    ax[0].set_title("Does the diagnostic forecast the loss? (family axis)"); ax[1].set_title("chemical-system axis")
    fig.tight_layout(); (rd / "figures").mkdir(exist_ok=True); fig.savefig(rd / "figures" / "fig7_panel_diagnostic.png", dpi=140, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
