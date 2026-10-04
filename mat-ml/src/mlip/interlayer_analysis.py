"""Analyse the JARVIS-referenced interlayer check against the criteria in results/mlip/stage2_preregistration.md (fixed beforehand).

  python -m src.mlip.interlayer_analysis [--dispersion on|off]

Reference: Eb_ref [meV/A^2] = exfoliation_energy [meV/atom] * nat / A  (JARVIS: E_exf = E_2D - E_bulk per atom, so meV/atom;
           1 meV/A^2 = 0.01602 J/m^2).  vdW regime := Eb_ref <= 40 meV/A^2 (defined from the reference only).
Criteria (vdW regime, dispersion on):
  C2  median ratio MACE/ref in [0.80, 1.25]
  C3  >= 70% of materials with ratio in [0.70, 1.30] and >= 90% in [0.50, 2.00]
  C4  failures (exception, collapsed gap < 2.0 A, or unbound E_b < 2 meV/A^2) <= 10%
Informational: Spearman, Pearson, MAE/RMSE/bias, gap plausibility. Interlayer distance and stacking are NOT validated by this reference.
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

OUT = "results/mlip"
VDW_MAX = 40.0
COLLAPSE_GAP = 2.0
UNBOUND = 2.0


def boot(fn, a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n):
        i = rng.integers(0, len(a), len(a))
        v.append(fn(a[i], b[i]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def load(disp):
    rows = []
    for l in open(f"{OUT}/raw_interlayer_mace_mpa0_medium_disp{disp}.jsonl"):
        r = json.loads(l)
        row = {"id": r["id"], "formula": r["formula"], "nat": r["nat"], "ref": r["ref"]["Eb_meV_A2"], "exf": r["ref"]["exf_meV_atom"],
               "error": r.get("error")}
        if "result" in r:
            b = r["result"]["best"]
            row.update({"eb": b["E_b_meV_A2"], "d_int": b["d_int"], "gap": b["gap"], "converged": b["converged"], "op": b["op"],
                        "scan_range": r["result"]["scan_energy_range_meV_A2"], "mono_conv": r["result"]["mono"]["converged"]})
        rows.append(row)
    return pd.DataFrame(rows)


def stats(d):
    m = d.eb.notna()
    x, y = d.eb[m].to_numpy(), d.ref[m].to_numpy()
    out = {"n": int(len(d)), "n_computed": int(m.sum())}
    if m.sum() >= 4:
        ratio = x / y
        out.update({"mae": float(np.abs(x - y).mean()), "rmse": float(np.sqrt(((x - y) ** 2).mean())), "bias": float((x - y).mean()),
                    "median_ratio": float(np.median(ratio)), "median_ratio_ci95": boot(lambda a, b: np.median(a / b), x, y),
                    "frac_ratio_0.7_1.3": float(((ratio >= 0.7) & (ratio <= 1.3)).mean()), "frac_ratio_0.5_2": float(((ratio >= 0.5) & (ratio <= 2)).mean()),
                    "spearman": float(spearmanr(x, y)[0]), "spearman_ci95": boot(lambda a, b: spearmanr(a, b)[0], x, y),
                    "pearson": float(pearsonr(x, y)[0])})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dispersion", choices=["on", "off"], default="on")
    args = ap.parse_args()
    d = load(args.dispersion)
    d["regime"] = np.where(d.ref <= VDW_MAX, "vdW", "bonded")
    d["failure"] = d.error.notna() | (d.gap < COLLAPSE_GAP) | (d.eb < UNBOUND)
    d["ratio"] = d.eb / d.ref
    res = {"dispersion": args.dispersion, "full": stats(d)}
    for reg, g in d.groupby("regime"):
        s = stats(g)
        s["failures"] = int(g.failure.sum()); s["failure_frac"] = float(g.failure.mean())
        res[reg] = s
    v = res.get("vdW", {})
    if v.get("n_computed", 0) >= 4:
        # failed structures stay in: unbound ones count as ratio ~0 (a number exists); exceptions count as fails below
        g = d[d.regime == "vdW"]
        ratio_all = g.ratio.where(g.eb.notna(), 0.0).clip(lower=0)
        c2 = 0.8 <= np.median(ratio_all) <= 1.25
        c3 = ((ratio_all >= 0.7) & (ratio_all <= 1.3)).mean() >= 0.7 and ((ratio_all >= 0.5) & (ratio_all <= 2)).mean() >= 0.9
        c4 = g.failure.mean() <= 0.10
        res["criteria_vdW_regime"] = {"C2_median_ratio_in_0.8_1.25": bool(c2), "median_ratio_incl_failures": float(np.median(ratio_all)),
                                      "C3_within_30pct_ge_70pct_and_factor2_ge_90pct": bool(c3),
                                      "frac_within_30pct": float(((ratio_all >= 0.7) & (ratio_all <= 1.3)).mean()),
                                      "frac_within_factor2": float(((ratio_all >= 0.5) & (ratio_all <= 2)).mean()),
                                      "C4_failures_le_10pct": bool(c4), "failure_frac": float(g.failure.mean()),
                                      "ALL_PASS": bool(c2 and c3 and c4)}
    res["gap_A_vdW_regime"] = d[d.regime == "vdW"].gap.describe().round(2).to_dict()
    res["converged_fraction"] = float(d.converged.mean()) if "converged" in d else None
    json.dump(res, open(f"{OUT}/stage2_jarvis_disp{args.dispersion}_summary.json", "w"), indent=2, default=float)
    d.to_csv(f"{OUT}/stage2_jarvis_disp{args.dispersion}_per_material.csv", index=False)
    print(json.dumps(res, indent=2, default=float))
    cols = ["id", "formula", "regime", "ref", "eb", "ratio", "gap", "d_int", "op", "converged", "failure", "error"]
    print(d.sort_values("ref")[cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
