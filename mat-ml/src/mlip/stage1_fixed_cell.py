"""Stage 1 analysis: fixed-DFT-cell vs MACE-relaxed-cell stiffness for MACE-MPA-0 on the 150 C2DB benchmark structures.

Runs in the MLIP environment. Rules are fixed in results/mlip/stage1_preregistration.md (written before the full run).

  python -m src.mlip.stage1_fixed_cell
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/mlip"
MODEL = "mace_mpa0_medium"
Q = ["c11", "c12", "c22", "Y2D", "poisson"]
GATE_MAE = 1.5 * json.load(open(f"{OUT}/db_disagreement.json"))["Y2D"]["mae"]   # 12.93 N/m, unchanged
GATE_SPEARMAN = 0.9


def load(variant: str) -> pd.DataFrame:
    inp = {i["id"]: i for i in json.load(open(f"{OUT}/benchmark_inputs.json")) if i["source"] == "c2db"}
    recs = {}
    for l in open(f"{OUT}/raw_{MODEL}_{variant}.jsonl"):
        if l.strip():
            r = json.loads(l)
            recs[r["id"]] = r          # last record per id wins (retries)
    rows = []
    for k, it in inp.items():
        r = recs.get(k)
        c = np.array(it["cell"])
        a_ref, b_ref = np.linalg.norm(c[0]), np.linalg.norm(c[1])
        row = {"id": k, "formula": it["formula"], "family": it["family"], "nat": it["nat"], "present": r is not None,
               "a_ref": a_ref, "b_ref": b_ref, "error": None if r is None else r.get("error")}
        if r is not None and "result" in r:
            x = r["result"]
            row.update({f"{q}_ml": x[q] for q in Q})
            row.update({"ml_stable": x["stable_tensor"], "asym": x["asym_frac"], "strain_conv": x["strain_relax_converged"],
                        "ref_conv": x["ref_relax"]["converged"], "a_ml": x["a_b_gamma"][0], "b_ml": x["a_b_gamma"][1],
                        "drift": x.get("cell_drift_rel"), "cell_fixed_ok": r.get("cell_fixed_ok"),
                        "pre_xx": x.get("pre_stress_xx_yy_xy_N_m", [np.nan] * 3)[0], "pre_yy": x.get("pre_stress_xx_yy_xy_N_m", [np.nan] * 3)[1],
                        "wall_s": r["wall_s"]})
        for q in Q:
            row[f"{q}_ref"] = it["ref"][q]
        rows.append(row)
    return pd.DataFrame(rows).set_index("id")


def metrics(p, r) -> dict:
    p, r = np.asarray(p, float), np.asarray(r, float)
    m = np.isfinite(p) & np.isfinite(r)
    p, r = p[m], r[m]
    ss = ((r - r.mean()) ** 2).sum()
    return {"n": int(len(p)), "mae": float(np.abs(p - r).mean()), "median_abs_err": float(np.median(np.abs(p - r))),
            "rmse": float(np.sqrt(((p - r) ** 2).mean())), "r2": float(1 - ((p - r) ** 2).sum() / ss),
            "spearman": float(spearmanr(p, r)[0]), "slope": float((p * r).sum() / (r * r).sum()), "bias": float((p - r).mean()),
            "median_ratio": float(np.median(p / r)) if np.all(r != 0) else np.nan}


def boot_f(a_err, b_err, n=2000, seed=0):
    """f = (mean(a_err) - mean(b_err)) / mean(a_err), paired bootstrap over structures."""
    rng = np.random.default_rng(seed)
    N = len(a_err)
    f = []
    for _ in range(n):
        i = rng.integers(0, N, N)
        f.append((a_err[i].mean() - b_err[i].mean()) / a_err[i].mean())
    return float((a_err.mean() - b_err.mean()) / a_err.mean()), float(np.percentile(f, 2.5)), float(np.percentile(f, 97.5))


def main():
    R, F = load("relaxed_cell"), load("dft_cell")
    both = R.index[R.get("Y2D_ml").notna() & F.get("Y2D_ml").notna()]
    summary = {"n_benchmark": len(R), "n_relaxed_computed": int(R.Y2D_ml.notna().sum()), "n_fixed_computed": int(F.Y2D_ml.notna().sum()),
               "n_paired": int(len(both))}

    # --- independent verification that the fixed cell is the C2DB cell
    Fp = F.loc[both]
    drift_vs_c2db = np.maximum(np.abs(Fp.a_ml / Fp.a_ref - 1), np.abs(Fp.b_ml / Fp.b_ref - 1))
    summary["fixed_cell_max_rel_diff_vs_original_C2DB_cell"] = float(drift_vs_c2db.max())
    summary["fixed_cell_flag_failures"] = int((Fp.cell_fixed_ok != True).sum())  # noqa: E712
    Rp = R.loc[both]
    lat_err = 100 * ((Rp.a_ml / Rp.a_ref - 1).abs().add((Rp.b_ml / Rp.b_ref - 1).abs()) / 2)
    summary["relaxed_cell_lattice_error_pct"] = {"median": float(lat_err.median()), "mean": float(lat_err.mean()), "p90": float(lat_err.quantile(.9)),
                                                 "max": float(lat_err.max())}

    # --- metrics per variant and quantity, two bases
    rows = []
    for name, D in (("relaxed_cell", R.loc[both]), ("dft_cell", F.loc[both])):
        for basis, d in (("all_computed", D), ("ml_stable_only", D[D.ml_stable == True])):  # noqa: E712
            for q in Q:
                dd = d
                if q == "poisson":   # robust subset: reference |nu|<=1 and finite prediction; R2 not used for the verdict
                    dd = d[d.poisson_ref.abs() <= 1]
                rows.append({"variant": name, "basis": basis, "quantity": q, **metrics(dd[f"{q}_ml"], dd[f"{q}_ref"])})
    M = pd.DataFrame(rows)
    M.to_csv(f"{OUT}/stage1_metrics.csv", index=False)

    # --- gate (unchanged) evaluated for information
    gate = []
    for v in ("relaxed_cell", "dft_cell"):
        m = M[(M.variant == v) & (M.basis == "all_computed") & (M.quantity == "Y2D")].iloc[0]
        gate.append({"variant": v, "n": m.n, "spearman": m.spearman, "mae": m.mae, "mae_threshold": GATE_MAE,
                     "pass": bool(m.spearman >= GATE_SPEARMAN and m.mae <= GATE_MAE)})
    G = pd.DataFrame(gate)
    G.to_csv(f"{OUT}/stage1_gate.csv", index=False)

    # --- attribution (pre-registered): paired f_geom
    att = []
    for q in ("Y2D", "c11", "c12", "c22"):
        ea = (R.loc[both, f"{q}_ml"] - R.loc[both, f"{q}_ref"]).abs().to_numpy()
        eb = (F.loc[both, f"{q}_ml"] - F.loc[both, f"{q}_ref"]).abs().to_numpy()
        f, lo, hi = boot_f(ea, eb)
        sl_r = M[(M.variant == "relaxed_cell") & (M.basis == "all_computed") & (M.quantity == q)].slope.iloc[0]
        sl_f = M[(M.variant == "dft_cell") & (M.basis == "all_computed") & (M.quantity == q)].slope.iloc[0]
        att.append({"quantity": q, "mae_relaxed": ea.mean(), "mae_fixed": eb.mean(), "f_geom": f, "ci_lo": lo, "ci_hi": hi,
                    "slope_relaxed": sl_r, "slope_fixed": sl_f, "slope_shift": sl_f - sl_r})
    A = pd.DataFrame(att)
    A.to_csv(f"{OUT}/stage1_attribution.csv", index=False)
    y = A[A.quantity == "Y2D"].iloc[0]
    rule = ("A (f_geom >= 0.33 and CI > 0)" if (y.f_geom >= 0.33 and y.ci_lo > 0) else "B (f_geom <= 0.10)" if y.f_geom <= 0.10 else "Mixed")
    summary["rule_output_preregistered"] = rule
    # wording used in reports (reworded 2026-10-04 at the author's request; the rule and its numeric output are unchanged):
    if y.ci_lo <= 0 <= y.ci_hi:
        statement = (f"No detectable geometry contribution: fixing the cell to the C2DB geometry changes the Y2D MAE by f_geom = {y.f_geom:.3f} "
                     f"(95% CI [{y.ci_lo:.3f}, {y.ci_hi:.3f}], includes zero; the upper bound is below the 0.33 level that would indicate a geometry-dominated error). "
                     "The analysis does not by itself identify the cause of the remaining error; it is consistent with the potential's strain response.")
    elif y.ci_lo > 0:
        statement = f"A geometry contribution is detectable: f_geom = {y.f_geom:.3f} (95% CI [{y.ci_lo:.3f}, {y.ci_hi:.3f}], excludes zero)."
    else:
        statement = f"Fixing the geometry made the error worse: f_geom = {y.f_geom:.3f} (95% CI [{y.ci_lo:.3f}, {y.ci_hi:.3f}])."
    summary["statement"] = statement
    summary["f_geom_Y2D"] = {"point": y.f_geom, "ci95": [y.ci_lo, y.ci_hi]}

    # --- per-structure changes and their drivers (descriptive)
    d = pd.DataFrame({"dY": F.loc[both, "Y2D_ml"] - R.loc[both, "Y2D_ml"], "lat_err_signed_pct": 100 * ((Rp.a_ml / Rp.a_ref - 1) + (Rp.b_ml / Rp.b_ref - 1)) / 2,
                      "pre_stress_mean_N_m": (Fp.pre_xx + Fp.pre_yy) / 2, "C11_ref": F.loc[both, "c11_ref"]})
    d["pre_over_C11_pct"] = 100 * d.pre_stress_mean_N_m.abs() / d.C11_ref
    summary["dY_fixed_minus_relaxed"] = {"median": float(d.dY.median()), "mean": float(d.dY.mean()), "frac_positive": float((d.dY > 0).mean())}
    summary["corr_dY_vs_lattice_error"] = {"pearson": float(pearsonr(d.dY, d.lat_err_signed_pct)[0]), "spearman": float(spearmanr(d.dY, d.lat_err_signed_pct)[0])}
    summary["pre_stress_over_C11_pct"] = {"median": float(d.pre_over_C11_pct.median()), "p90": float(d.pre_over_C11_pct.quantile(.9)),
                                          "frac_gt_2pct": float((d.pre_over_C11_pct > 2).mean())}
    # error vs lattice error: does the Y2D error shrink where the lattice was wrong?
    big = lat_err > lat_err.median()
    for nm, mask in (("lattice_error_above_median", big), ("lattice_error_below_median", ~big)):
        ea = (R.loc[both][mask.values].Y2D_ml - R.loc[both][mask.values].Y2D_ref).abs().to_numpy()
        eb = (F.loc[both][mask.values].Y2D_ml - F.loc[both][mask.values].Y2D_ref).abs().to_numpy()
        f, lo, hi = boot_f(ea, eb)
        summary[f"f_geom_Y2D_{nm}"] = {"n": int(mask.sum()), "mae_relaxed": float(ea.mean()), "mae_fixed": float(eb.mean()), "f": f, "ci95": [lo, hi]}

    # --- failures / instabilities / outliers
    fl = {}
    for name, D in (("relaxed_cell", R), ("dft_cell", F)):
        fl[name] = {"missing": int((~D.present).sum()), "exceptions": int(D.error.notna().sum()),
                    "ref_relax_not_converged": int((D.ref_conv == False).sum()), "strain_relax_not_converged": int((D.strain_conv == False).sum()),  # noqa: E712
                    "ml_unstable_tensor": int((D.ml_stable == False).sum()), "asym_gt_10pct": int((D.asym > 0.10).sum())}  # noqa: E712
    summary["failures"] = fl
    out = []
    for name, D in (("relaxed_cell", R), ("dft_cell", F)):
        t = D.loc[both].assign(err=lambda x: x.Y2D_ml - x.Y2D_ref).assign(abs_err=lambda x: x.err.abs()).sort_values("abs_err", ascending=False).head(8)
        for k, r in t.iterrows():
            out.append({"variant": name, "id": k, "formula": r.formula, "family": r.family, "Y2D_ref": r.Y2D_ref, "Y2D_ml": r.Y2D_ml,
                        "err": r.err, "ml_stable": r.ml_stable})
    pd.DataFrame(out).to_csv(f"{OUT}/stage1_worst_outliers.csv", index=False)
    pair = pd.DataFrame({"id": both, "formula": R.loc[both].formula.values, "Y2D_ref": R.loc[both].Y2D_ref.values, "Y2D_relaxed": R.loc[both].Y2D_ml.values,
                         "Y2D_fixed": F.loc[both].Y2D_ml.values, "a_err_pct_relaxed": 100 * (Rp.a_ml / Rp.a_ref - 1).values, "pre_stress_N_m": d.pre_stress_mean_N_m.values,
                         "stable_relaxed": R.loc[both].ml_stable.values, "stable_fixed": F.loc[both].ml_stable.values})
    pair.to_csv(f"{OUT}/stage1_paired_structures.csv", index=False)
    json.dump(summary, open(f"{OUT}/stage1_summary.json", "w"), indent=2, default=float)

    # --- plots
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for a, (nm, D, c) in zip(ax[:2], (("relaxed cell (MACE)", R.loc[both], "#2a6fbb"), ("fixed C2DB cell", F.loc[both], "#d9822b"))):
        a.scatter(D.Y2D_ref, D.Y2D_ml.clip(lower=0.5), s=14, alpha=.7, c=c)
        lim = [1, 1000]
        a.plot(lim, lim, "k-", lw=.8)
        a.fill_between(lim, [x - GATE_MAE for x in lim], [x + GATE_MAE for x in lim], color="grey", alpha=.15)
        a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(lim); a.set_ylim(lim)
        a.set_xlabel("C2DB Y2D (N/m)"); a.set_ylabel("MACE-MPA-0 Y2D (N/m)"); a.set_title(nm)
    ax[2].scatter(d.lat_err_signed_pct, d.dY, s=14, alpha=.7, c="#444")
    ax[2].axhline(0, c="k", lw=.6); ax[2].axvline(0, c="k", lw=.6)
    ax[2].set_xlabel("MACE lattice error vs C2DB (%, + = too large)"); ax[2].set_ylabel("Y2D(fixed) - Y2D(relaxed) (N/m)")
    fig.tight_layout(); fig.savefig(f"{OUT}/plots/stage1_fixed_vs_relaxed.png", dpi=130); plt.close(fig)

    def md(df):
        return df.round(3).to_string(index=False)
    print(json.dumps(summary, indent=2, default=float)); print("\nGATE (information only)\n" + md(G))
    print("\nATTRIBUTION\n" + md(A)); print("\nMETRICS (all computed)\n" + md(M[M.basis == "all_computed"]))
    print("\nMETRICS (ML-stable only)\n" + md(M[M.basis == "ml_stable_only"]))


if __name__ == "__main__":
    main()
