"""Analyse the bilayer / monolayer in-plane stiffness ratio (descriptive; see results/mlip/bidb_preregistration.md).

  python -m src.mlip.bidb_stiffness_analysis

R = Y2D_total(bilayer) / Y2D(monolayer); ~2 if the layers act independently (per-layer ratio R/2 ~ 1).
Bases: "all_computed" (ML-unstable tensors kept, as in the calibration gate) and "strict" (both tensors stable, all relaxations converged, stacking preserved).
CIs: cluster bootstrap over monolayers (2000 resamples, seed 0).
"""
import glob
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = "results/mlip"


def load():
    recs = {}
    for p in sorted(glob.glob(f"{OUT}/raw_bidb_stiffness_mace_mpa0_medium*.jsonl")):
        for l in open(p):
            if l.strip():
                r = json.loads(l)
                recs[r["key"]] = r
    mono = {r["uid"]: r for r in recs.values() if r["type"] == "mono"}
    rows = []
    for r in recs.values():
        if r["type"] != "bi":
            continue
        m = mono.get(r["monolayer_uid"])
        row = {"uid": r["uid"], "monolayer_uid": r["monolayer_uid"], "formula": r["formula"], "pick": r["pick"], "ref_zscan": r["ref_zscan"],
               "ref_distance": r["ref_distance"], "bi_error": r.get("error"), "mono_error": None if m is None else m.get("error"), "mono_missing": m is None}
        if "result" in r and m is not None and "result" in m:
            b, o = r["result"], m["result"]
            row.update({"Y_bi_total": b["Y2D"], "Y_mono": o["Y2D"], "c11_bi": b["c11"], "c22_bi": b["c22"], "c11_mono": o["c11"], "c22_mono": o["c22"],
                        "bi_stable": b["stable_tensor"], "mono_stable": o["stable_tensor"], "bi_asym": b["asym_frac"], "mono_asym": o["asym_frac"],
                        "bi_conv": b["strain_relax_converged"] and b["ref_relax"]["converged"], "mono_conv": o["strain_relax_converged"] and o["ref_relax"]["converged"],
                        "preserved": b["stacking_preserved"], "shift_A": b["layer_shift_change_A"], "gap": b["gap_after_relax"]})
        rows.append(row)
    d = pd.DataFrame(rows)
    d["R"] = d.Y_bi_total / d.Y_mono
    d["R_c11"] = d.c11_bi / d.c11_mono
    d["R_c22"] = d.c22_bi / d.c22_mono
    d["Y_bi_per_layer"] = d.Y_bi_total / 2
    d["strict"] = (d.bi_stable == True) & (d.mono_stable == True) & (d.bi_conv == True) & (d.mono_conv == True) & (d.preserved == True)  # noqa: E712
    return d


def describe(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return {"n": int(len(x)), "median": float(np.median(x)), "mean": float(x.mean()), "std": float(x.std(ddof=1)), "iqr": [float(np.percentile(x, 25)), float(np.percentile(x, 75))],
            "p5_p95": [float(np.percentile(x, 5)), float(np.percentile(x, 95))], "frac_1.8_2.2": float(((x >= 1.8) & (x <= 2.2)).mean()),
            "median_abs_dev_from_2_rel": float(np.median(np.abs(x - 2) / 2))}


def cboot(d, fn, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    gs = [g for _, g in d.groupby("monolayer_uid")]
    v = []
    for _ in range(n):
        s = pd.concat([gs[i] for i in rng.integers(0, len(gs), len(gs))])
        v.append(fn(s))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    d = load()
    res = {"n_bilayers_set": len(d), "bilayer_exceptions": int(d.bi_error.notna().sum()), "mono_exceptions_or_missing": int((d.mono_error.notna() | d.mono_missing).sum())}
    a = d[d.R.notna() & (d.Y_mono > 0)].copy()                       # all computed with a defined, positive monolayer reference
    s = a[a.strict]
    for name, g in (("all_computed", a), ("strict", s)):
        res[name] = {"R_Y2D": describe(g.R), "R_C11": describe(g.R_c11), "R_C22": describe(g.R_c22), "per_layer_ratio_median": float(g.R.median() / 2),
                     "median_ci95": cboot(g, lambda x: x.R.median()) if len(g) > 5 else None,
                     "spearman_R_vs_ref_Eb": float(spearmanr(g.R, g.ref_zscan)[0]) if len(g) > 5 else None,
                     "spearman_R_vs_relaxed_gap": float(spearmanr(g.R, g.gap)[0]) if len(g) > 5 else None,
                     "by_pick": {k: describe(v.R) for k, v in g.groupby(g.pick.str.contains("best").map({True: "best", False: "worst"}))}}
    res["flags"] = {"bilayer_unstable": int((a.bi_stable == False).sum()), "mono_unstable": int((a.mono_stable == False).sum()),  # noqa: E712
                    "not_converged": int(((a.bi_conv == False) | (a.mono_conv == False)).sum()),  # noqa: E712
                    "stacking_changed_on_relaxation": int((a.preserved == False).sum()), "asym_gt_10pct_bi": int((a.bi_asym > .10).sum())}  # noqa: E712
    # within-monolayer: best vs worst stacking
    w = []
    for m, g in s.groupby("monolayer_uid"):
        b = g[g.pick.str.contains("best")]
        o = g[g.pick.str.contains("worst")]
        if len(b) and len(o) and b.uid.iloc[0] != o.uid.iloc[0]:
            w.append({"monolayer_uid": m, "formula": g.formula.iloc[0], "R_best": b.R.iloc[0], "R_worst": o.R.iloc[0], "abs_diff": abs(b.R.iloc[0] - o.R.iloc[0]),
                      "Y_bi_best": b.Y_bi_total.iloc[0], "Y_bi_worst": o.Y_bi_total.iloc[0]})
    W = pd.DataFrame(w)
    if len(W):
        res["within_monolayer_best_vs_worst"] = {"n_monolayers": len(W), "median_abs_diff_R": float(W.abs_diff.median()), "mean_abs_diff_R": float(W.abs_diff.mean()),
                                                 "median_abs_diff_Y_N_m": float((W.Y_bi_best - W.Y_bi_worst).abs().median())}
    res["calibration_context"] = {"monolayer_Y2D_MAE_N_m_C2DB_150": 9.91, "monolayer_ratio_ml_over_ref_IQR": [0.77, 1.10],
                                  "note": "R is a ratio of two MACE calculations; errors may partly cancel. This is not tested here."}
    d.to_csv(f"{OUT}/bidb_stiffness_per_bilayer.csv", index=False)
    W.to_csv(f"{OUT}/bidb_stiffness_best_vs_worst.csv", index=False)
    json.dump(res, open(f"{OUT}/bidb_stiffness_summary.json", "w"), indent=2, default=float)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(a.R.clip(0, 4), bins=30, alpha=.6, label="all computed"); ax[0].hist(s.R.clip(0, 4), bins=30, alpha=.6, label="strict")
    ax[0].axvline(2, c="k", lw=1); ax[0].set_xlabel("R = Y2D_total(bilayer) / Y2D(monolayer), clipped to [0, 4]"); ax[0].legend()
    ax[1].scatter(a.gap, a.R.clip(0, 4), s=14); ax[1].axhline(2, c="k", lw=1); ax[1].set_xlabel("relaxed interlayer gap (Å)"); ax[1].set_ylabel("R")
    fig.tight_layout(); fig.savefig(f"{OUT}/plots/bidb_stiffness_ratio.png", dpi=130); plt.close(fig)
    print(json.dumps(res, indent=2, default=float))
    print(a.sort_values("R")[["uid", "pick", "Y_mono", "Y_bi_total", "R", "gap", "bi_stable", "preserved"]].round(2).head(8).to_string(index=False))
    print(a.sort_values("R")[["uid", "pick", "Y_mono", "Y_bi_total", "R", "gap", "bi_stable", "preserved"]].round(2).tail(6).to_string(index=False))


if __name__ == "__main__":
    main()
