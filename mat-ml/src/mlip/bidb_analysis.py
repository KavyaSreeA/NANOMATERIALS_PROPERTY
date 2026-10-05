"""BiDB validation analysis against the criteria in results/mlip/bidb_preregistration.md (fixed before any MACE-BiDB number existed).

  python -m src.mlip.bidb_analysis

Confidence intervals: cluster bootstrap over MONOLAYERS (stackings of one monolayer are not independent), 2000 resamples, seed 0.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = "results/mlip"
RAW = f"{OUT}/raw_bidb_zscan_mace_mpa0_medium_dispon.jsonl"
UNBOUND = 2.0       # meV/A^2
REF_MIN_FOR_FAIL = 5.0


def load() -> pd.DataFrame:
    recs = {}
    for l in open(RAW):
        if l.strip():
            r = json.loads(l)
            recs[r["uid"]] = r
    rows = []
    for r in recs.values():
        row = {"uid": r["uid"], "monolayer_uid": r["monolayer_uid"], "formula": r["formula"], "d_ref": r["ref"]["distance"], "eb_ref": r["ref"]["zscan"],
               "gs_ref": r["ref"]["gs"], "slide": r["ref"]["slide_stability"], "ref_valid": r["ref"]["ref_valid"], "error": r.get("error"), "wall_s": r["wall_s"]}
        if "result" in r:
            x = r["result"]
            row.update({"d_ml": x["gap"], "eb_ml": x["E_b_meV_A2"], "edge_high": x["edge_high"], "edge_low": x["edge_low"], "sanity6": x["sanity_E_at_6A_meV_A2"]})
        rows.append(row)
    return pd.DataFrame(rows)


def cluster_boot(d, fn, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    groups = [g for _, g in d.groupby("monolayer_uid")]
    v = []
    for _ in range(n):
        idx = rng.integers(0, len(groups), len(groups))
        s = pd.concat([groups[i] for i in idx])
        try:
            v.append(fn(s))
        except Exception:
            pass
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    d = load()
    res = {"n_set": len(d), "n_exceptions": int(d.error.notna().sum()), "n_ref_invalid": int((~d.ref_valid).sum())}
    v = d[d.ref_valid & d.error.isna()].copy()
    res["n_analysed"] = len(v)
    v["dd"] = v.d_ml - v.d_ref
    v["ratio"] = v.eb_ml / v.eb_ref
    v["fail"] = ((v.eb_ml < UNBOUND) & (v.eb_ref >= REF_MIN_FOR_FAIL)) | v.edge_high
    # ---- distance
    ad = v.dd.abs()
    res["distance"] = {"median_abs": float(ad.median()), "p90_abs": float(ad.quantile(.9)), "mae": float(ad.mean()), "rmse": float(np.sqrt((v.dd ** 2).mean())),
                       "bias": float(v.dd.mean()), "frac_within_0.1": float((ad <= 0.1).mean()), "frac_within_0.3": float((ad <= 0.3).mean()),
                       "median_abs_ci95": cluster_boot(v, lambda s: (s.d_ml - s.d_ref).abs().median()),
                       "p90_abs_ci95": cluster_boot(v, lambda s: (s.d_ml - s.d_ref).abs().quantile(.9)),
                       "spearman": float(spearmanr(v.d_ml, v.d_ref)[0])}
    # ---- binding energy
    ratio_all = v.ratio.where(~v.fail, 0.0).clip(lower=0)
    res["binding_energy"] = {"spearman": float(spearmanr(v.eb_ml, v.eb_ref)[0]), "spearman_ci95": cluster_boot(v, lambda s: spearmanr(s.eb_ml, s.eb_ref)[0]),
                             "pearson": float(np.corrcoef(v.eb_ml, v.eb_ref)[0, 1]), "mae": float((v.eb_ml - v.eb_ref).abs().mean()),
                             "rmse": float(np.sqrt(((v.eb_ml - v.eb_ref) ** 2).mean())), "bias": float((v.eb_ml - v.eb_ref).mean()),
                             "median_ratio": float(v.ratio.median()), "median_ratio_incl_failures_as_0": float(ratio_all.median()),
                             "median_ratio_ci95": cluster_boot(v, lambda s: (s.eb_ml / s.eb_ref).median()),
                             "frac_ratio_0.7_1.3": float(((ratio_all >= .7) & (ratio_all <= 1.3)).mean()),
                             "frac_ratio_0.5_2": float(((ratio_all >= .5) & (ratio_all <= 2)).mean()),
                             "failures": int(v.fail.sum()) + res["n_exceptions"], "failure_frac": float((v.fail.sum() + res["n_exceptions"]) / (len(v) + res["n_exceptions"])),
                             "spearman_vs_gs_column_informational": float(spearmanr(v.dropna(subset=["gs_ref"]).eb_ml, v.dropna(subset=["gs_ref"]).gs_ref)[0]),
                             "sanity_E_at_6A_median_meV_A2": float(v.sanity6.median())}
    # ---- stacking dependence
    per = []
    for m, g in v.groupby("monolayer_uid"):
        if len(g) >= 4:
            rho = spearmanr(g.eb_ml, g.eb_ref)[0] if g.eb_ref.nunique() > 1 and g.eb_ml.nunique() > 1 else np.nan
            top2 = g.sort_values("eb_ml", ascending=False).head(2).uid.tolist()
            best = g.sort_values("eb_ref", ascending=False).iloc[0].uid
            cen_err = ((g.eb_ml - g.eb_ml.mean()) - (g.eb_ref - g.eb_ref.mean())).abs().mean()
            per.append({"monolayer_uid": m, "formula": g.formula.iloc[0], "n": len(g), "spearman": rho, "best_ref_in_ml_top2": bool(best in top2),
                        "ref_spread_meV_A2": float(g.eb_ref.max() - g.eb_ref.min()), "ml_spread_meV_A2": float(g.eb_ml.max() - g.eb_ml.min()),
                        "centered_mae_meV_A2": float(cen_err)})
    P = pd.DataFrame(per)
    res["stacking"] = {"n_monolayers": len(P), "mean_within_spearman": float(P.spearman.mean()), "median_within_spearman": float(P.spearman.median()),
                       "frac_best_in_top2": float(P.best_ref_in_ml_top2.mean()), "median_ref_spread": float(P.ref_spread_meV_A2.median()),
                       "median_ml_spread": float(P.ml_spread_meV_A2.median()), "median_centered_mae": float(P.centered_mae_meV_A2.median())}
    res["stacking"]["mean_within_spearman_ci95"] = cluster_boot(v, lambda s: np.nanmean([spearmanr(g.eb_ml, g.eb_ref)[0] for _, g in s.groupby("monolayer_uid") if len(g) >= 4]))
    # ---- criteria (fixed beforehand)
    D1 = res["distance"]["median_abs"] <= 0.10 and res["distance"]["p90_abs"] <= 0.30
    E1 = res["binding_energy"]["spearman"] >= 0.80
    be = res["binding_energy"]
    E2 = 0.8 <= be["median_ratio_incl_failures_as_0"] <= 1.25 and be["frac_ratio_0.7_1.3"] >= 0.7 and be["frac_ratio_0.5_2"] >= 0.9
    E3 = be["failure_frac"] <= 0.10
    S1 = res["stacking"]["mean_within_spearman"] >= 0.6 and res["stacking"]["frac_best_in_top2"] >= 0.6
    res["criteria"] = {"D1_distance": bool(D1), "E1_spearman_ge_0.80": bool(E1), "E2_ratio": bool(E2), "E3_failures_le_10pct": bool(E3),
                       "binding_and_distance_reproduced": bool(D1 and E1 and E2 and E3), "S1_stacking": bool(S1),
                       "stacking_dependence_reproduced": bool(D1 and E1 and E2 and E3 and S1)}
    # ---- breakdowns (descriptive)
    v["ref_bin"] = pd.cut(v.eb_ref, [0, 10, 15, 20, 30, 1e9], labels=["<10", "10-15", "15-20", "20-30", ">30"])
    res["by_reference_range"] = {str(k): {"n": int(len(g)), "median_ratio": float(g.ratio.median()), "median_abs_dd": float(g.dd.abs().median())}
                                 for k, g in v.groupby("ref_bin", observed=True)}
    res["by_slide_stability"] = {str(k): {"n": int(len(g)), "median_ratio": float(g.ratio.median()), "median_abs_dd": float(g.dd.abs().median())}
                                 for k, g in v.groupby("slide")}
    Pm = v.groupby("monolayer_uid").apply(lambda g: pd.Series({"formula": g.formula.iloc[0], "n": len(g), "median_ratio": g.ratio.median(),
                                                               "median_abs_dd": g.dd.abs().median(), "fails": int(g.fail.sum())}), include_groups=False)
    Pm.to_csv(f"{OUT}/bidb_per_monolayer.csv")
    v.to_csv(f"{OUT}/bidb_per_bilayer.csv", index=False)
    d[d.error.notna()][["uid", "formula", "error"]].to_csv(f"{OUT}/bidb_exceptions.csv", index=False)
    P.to_csv(f"{OUT}/bidb_stacking_per_monolayer.csv", index=False)
    json.dump(res, open(f"{OUT}/bidb_summary.json", "w"), indent=2, default=float)
    # ---- plots
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    ax[0].scatter(v.eb_ref, v.eb_ml, s=10, alpha=.6); lim = [0, max(v.eb_ref.max(), v.eb_ml.max()) * 1.05]
    ax[0].plot(lim, lim, "k-", lw=.8); ax[0].set_xlabel("BiDB E_b zscan (meV/Å²)"); ax[0].set_ylabel("MACE-MPA-0+D3 E_b (meV/Å²)")
    ax[1].scatter(v.d_ref, v.d_ml, s=10, alpha=.6); ax[1].plot([0, 5], [0, 5], "k-", lw=.8); ax[1].set_xlabel("BiDB gap (Å)"); ax[1].set_ylabel("MACE gap (Å)")
    ax[2].hist(v.dd.clip(-1, 1), bins=40); ax[2].set_xlabel("gap error MACE - BiDB (Å), clipped ±1")
    fig.tight_layout(); fig.savefig(f"{OUT}/plots/bidb_parity.png", dpi=130); plt.close(fig)
    print(json.dumps(res, indent=2, default=float))
    print(P.round(2).to_string(index=False))
    print(v.assign(a=v.dd.abs()).sort_values("a", ascending=False).head(8)[["uid", "d_ref", "d_ml", "eb_ref", "eb_ml", "ratio"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
