"""Analyse the calibration runs: metrics, gate, failures, runtime, plots. Runs in the MLIP environment.

  python -m src.mlip.report_calibration

GATE (fixed before looking at results; proposed thresholds, not scientific facts):
  basis    Y2D, every structure the potential computed (ML-unstable tensors are NOT removed)
  pass if  Spearman >= 0.9  AND  MAE <= 1.5 x (C2DB-vs-JARVIS MAE on 106 materials present in both)
The yardstick is read from results/mlip/db_disagreement.json (Y2D: 8.62 N/m -> threshold 12.93 N/m).
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = "results/mlip"
Q = ["c11", "c12", "c22", "Y2D", "poisson"]


def load_runs() -> pd.DataFrame:
    inp = {i["id"]: i for i in json.load(open(f"{OUT}/benchmark_inputs.json"))}
    rows = []
    for path in sorted(glob.glob(f"{OUT}/raw_*.jsonl")):
        for l in open(path):
            if not l.strip():
                continue
            r = json.loads(l)
            x = r.get("result", {})
            it = inp[r["id"]]
            cell = np.array(it["cell"])
            row = {"id": r["id"], "source": r["source"], "formula": r["formula"], "family": r["family"], "nat": r["nat"],
                   "model": r["model"], "variant": r["settings"]["variant"], "error": r.get("error"), "wall_s": r["wall_s"],
                   "peak_gpu_mb": r["peak_gpu_mb"], "a_ref": float(np.linalg.norm(cell[0])), "b_ref": float(np.linalg.norm(cell[1])),
                   "ref_relax_converged": x.get("ref_relax", {}).get("converged"), "strain_converged": x.get("strain_relax_converged"),
                   "ml_stable": x.get("stable_tensor"), "asym_frac": x.get("asym_frac")}
            ab = x.get("a_b_gamma")
            row["a_ml"], row["b_ml"] = (ab[0], ab[1]) if ab else (np.nan, np.nan)
            for q in Q + ["c66"]:
                row[f"{q}_ml"] = x.get(q, np.nan)
                row[f"{q}_ref"] = r["ref"].get(q)
            rows.append(row)
    d = pd.DataFrame(rows).drop_duplicates(subset=["id", "model", "variant"], keep="last")  # retries append later records
    for q in Q + ["c66"]:
        d[f"{q}_ref"] = pd.to_numeric(d[f"{q}_ref"], errors="coerce")
    return d


def metrics(pred, ref) -> dict:
    m = np.isfinite(pred) & np.isfinite(ref)
    p, r = pred[m], ref[m]
    if len(p) < 3:
        return {"n": int(len(p))}
    ss = ((r - r.mean()) ** 2).sum()
    return {"n": int(len(p)), "mae": float(np.abs(p - r).mean()), "rmse": float(np.sqrt(((p - r) ** 2).mean())),
            "r2": float(1 - ((p - r) ** 2).sum() / ss), "spearman": float(spearmanr(p, r)[0]),
            "slope_through_origin": float((p * r).sum() / (r * r).sum()), "bias": float((p - r).mean()),
            "median_ratio": float(np.median(p / r)) if np.all(r != 0) else np.nan}


def main():
    os.makedirs(f"{OUT}/plots", exist_ok=True)
    d = load_runs()
    dis = json.load(open(f"{OUT}/db_disagreement.json"))
    ok = d[d.error.isna()].copy()
    rows = []
    for (model, variant), g in ok.groupby(["model", "variant"]):
        for subset, gg in (("all", g), ("c2db", g[g.source == "c2db"]), ("jarvis", g[g.source == "jarvis"])):
            for stab, g3 in (("all_computed", gg), ("ml_stable_only", gg[gg.ml_stable == True])):  # noqa: E712
                for q in Q + ["c66"]:
                    if q == "c66" and subset == "jarvis":
                        continue
                    rows.append({"model": model, "variant": variant, "subset": subset, "filter": stab, "quantity": q,
                                 **metrics(g3[f"{q}_ml"].to_numpy(float), g3[f"{q}_ref"].to_numpy(float))})
    M = pd.DataFrame(rows)
    M.to_csv(f"{OUT}/calibration_metrics.csv", index=False)

    # gate
    thr_mae = 1.5 * dis["Y2D"]["mae"]
    gate = []
    for (model, variant, subset), g in M[(M.quantity == "Y2D") & (M["filter"] == "all_computed")].groupby(["model", "variant", "subset"]):
        r = g.iloc[0]
        gate.append({"model": model, "variant": variant, "subset": subset, "n": r.n, "spearman": r.spearman, "mae": r.mae,
                     "mae_threshold": thr_mae, "pass_spearman": r.spearman >= 0.9, "pass_mae": r.mae <= thr_mae,
                     "GATE_PASS": bool(r.spearman >= 0.9 and r.mae <= thr_mae)})
    G = pd.DataFrame(gate)
    G.to_csv(f"{OUT}/calibration_gate.csv", index=False)

    # failures
    F = d.groupby(["model", "variant"]).agg(n=("id", "size"), exceptions=("error", lambda s: int(s.notna().sum())),
                                            ref_relax_not_converged=("ref_relax_converged", lambda s: int((s == False).sum())),  # noqa: E712
                                            strain_not_converged=("strain_converged", lambda s: int((s == False).sum())),  # noqa: E712
                                            ml_unstable=("ml_stable", lambda s: int((s == False).sum())),  # noqa: E712
                                            asym_gt_10pct=("asym_frac", lambda s: int((s > 0.10).sum()))).reset_index()
    # runtime / memory
    T = d.groupby(["model", "variant"]).agg(median_s=("wall_s", "median"), p90_s=("wall_s", lambda s: s.quantile(.9)),
                                            total_h=("wall_s", lambda s: s.sum() / 3600), peak_gpu_mb=("peak_gpu_mb", "max")).reset_index()
    # lattice error of the potential's own equilibrium
    ok["a_err_pct"] = 100 * (ok.a_ml / ok.a_ref - 1)
    L = ok[ok.variant == "relaxed_cell"].groupby("model").a_err_pct.agg(["median", "mean", lambda s: s.abs().quantile(.9)]).rename(
        columns={"<lambda_0>": "p90_abs"}).reset_index()
    # worst families (relative Y2D error)
    ok["rel_err_Y2D"] = (ok.Y2D_ml - ok.Y2D_ref) / ok.Y2D_ref
    W = ok[ok.variant == "relaxed_cell"].assign(abs_rel=lambda x: x.rel_err_Y2D.abs()).sort_values("abs_rel", ascending=False)
    W = W.groupby("model").head(8)[["model", "id", "formula", "family", "Y2D_ref", "Y2D_ml", "rel_err_Y2D", "ml_stable"]]

    # plots
    for variant in sorted(ok.variant.unique()):
        models = sorted(ok[ok.variant == variant].model.unique())
        fig, ax = plt.subplots(1, len(models), figsize=(4.6 * len(models), 4.4), squeeze=False)
        for a, m in zip(ax[0], models):
            g = ok[(ok.model == m) & (ok.variant == variant)]
            for src, c in (("c2db", "#2a6fbb"), ("jarvis", "#d9822b")):
                s = g[g.source == src]
                a.scatter(s.Y2D_ref, s.Y2D_ml.clip(lower=0.5), s=14, alpha=.7, c=c, label=src)
            lim = [1, 1000]
            a.plot(lim, lim, "k-", lw=.8)
            a.fill_between(lim, [x - thr_mae for x in lim], [x + thr_mae for x in lim], color="grey", alpha=.15, label="±gate MAE")
            a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(lim); a.set_ylim(lim)
            a.set_xlabel("reference Y2D (N/m)"); a.set_ylabel("potential Y2D (N/m)"); a.set_title(f"{m}\n{variant}", fontsize=10)
            a.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(f"{OUT}/plots/parity_Y2D_{variant}.png", dpi=130); plt.close(fig)
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        for m in models:
            g = ok[(ok.model == m) & (ok.variant == variant)]
            ax[0].hist((g.Y2D_ml - g.Y2D_ref).clip(-150, 150), bins=40, alpha=.5, label=m)
            ax[1].hist(g.rel_err_Y2D.clip(-1, 1.5), bins=40, alpha=.5, label=m)
        ax[0].set_xlabel("Y2D error (N/m), clipped ±150"); ax[1].set_xlabel("relative Y2D error, clipped"); ax[0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(f"{OUT}/plots/error_hist_{variant}.png", dpi=130); plt.close(fig)

    def md(df):
        df = df.copy()
        return df.to_markdown(index=False, floatfmt=".3g") if hasattr(df, "to_markdown") else df.to_string(index=False)

    try:
        import tabulate  # noqa: F401
    except ImportError:
        md = lambda df: df.to_string(index=False)  # noqa: E731
    main = M[(M["filter"] == "all_computed") & (M.quantity.isin(["Y2D", "c11", "c12", "c22", "poisson"]))]
    text = ["# Calibration tables (auto-generated)", "", f"Database disagreement yardstick (C2DB vs JARVIS, {dis['n_matched']} materials): "
            + ", ".join(f"{q} MAE {dis[q]['mae']:.3g}" for q in Q), f"Gate MAE threshold for Y2D: {thr_mae:.2f} N/m", "",
            "## Gate (Y2D, all computed structures)", md(G.round(3)), "", "## Metrics, all computed structures", md(main.round(3)), "",
            "## Metrics, ML-stable structures only (Y2D)", md(M[(M["filter"] == "ml_stable_only") & (M.quantity == "Y2D")].round(3)), "",
            "## Failures / instabilities", md(F), "", "## Runtime and memory", md(T.round(3)), "",
            "## Lattice constant error of the potential's own equilibrium (a, %)", md(L.round(3)), "",
            "## Worst Y2D errors (relaxed_cell)", md(W.round(3))]
    open(f"{OUT}/calibration_tables.md", "w", encoding="utf-8").write("\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    main()
