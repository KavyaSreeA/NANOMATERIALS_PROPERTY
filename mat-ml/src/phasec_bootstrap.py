"""Phase C1: cluster-bootstrap CIs for the headline Task A / Task B quantities (design: results/phaseC_design.md).   python -m src.phasec_bootstrap"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from . import data as D

NB = 2000
RNG = np.random.default_rng(0)
cfg = D.load_config()
RD = D.ROOT / cfg["paths"]["results_dir"]


def pooled(df):
    return float(np.abs(df.pred - df.y).mean()), float(r2_score(df.y, df.pred))


def boot(d, schemes, unit_of, ratio_pairs, retention=None):
    """d: long oof table (rows aligned across schemes by uid). unit_of: dict scheme/pair -> column used as the resampling unit."""
    out = {}
    base = {s: d[d.scheme == s].reset_index(drop=True) for s in schemes}
    point = {s: pooled(base[s]) for s in schemes}
    res = {"point": {s: {"MAE": point[s][0], "R2": point[s][1]} for s in schemes}, "ratios": {}, "ci": {}}
    for s in schemes:
        res["ci"][s] = {"MAE": [], "R2": []}
    ratio_draws = {k: [] for k in ratio_pairs}
    # per-scheme CIs resample that scheme's own unit
    for s in schemes:
        g = base[s].groupby(unit_of[s]).indices
        keys = list(g)
        for _ in range(NB):
            idx = np.concatenate([g[keys[i]] for i in RNG.integers(0, len(keys), len(keys))])
            m, r = pooled(base[s].iloc[idx])
            res["ci"][s]["MAE"].append(m); res["ci"][s]["R2"].append(r)
    for s in schemes:
        for q in ("MAE", "R2"):
            v = res["ci"][s][q]
            res["ci"][s][q] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    # ratios: the SAME resampled units for numerator and denominator
    for (num, den), unit in ratio_pairs.items():
        a, b = base[num], base[den]
        assert (a.uid.values == b.uid.values).all()
        g = a.groupby(unit).indices
        keys = list(g)
        draws = []
        for _ in range(NB):
            idx = np.concatenate([g[keys[i]] for i in RNG.integers(0, len(keys), len(keys))])
            draws.append(pooled(a.iloc[idx])[0] / pooled(b.iloc[idx])[0])
        res["ratios"][f"{num}/{den}"] = {"point": point[num][0] / point[den][0], "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))], "unit": unit}
    return res


out = {}
A = pd.read_csv(RD / "diagnostics" / "taskA_oof_predictions.csv")
for tgt in ("Y2D", "poisson"):
    d = A[A.target == tgt]
    out[f"TaskA|{tgt}"] = boot(d, ["random", "chemsys", "family"], {"random": "family", "chemsys": "family", "family": "family"},
                              {("chemsys", "random"): "family", ("family", "random"): "family"})
B = pd.read_csv(RD / "diagnostics" / "taskB_oof_predictions.csv")
for tgt in ("binding_energy_zscan", "distance"):
    d = B[B.target == tgt]
    out[f"TaskB|{tgt}"] = boot(d, ["random", "monolayer", "family"], {"random": "monolayer_uid", "monolayer": "monolayer_uid", "family": "family"},
                              {("monolayer", "random"): "monolayer_uid", ("family", "random"): "family"})
json.dump(out, open(RD / "phaseC_cluster_bootstrap.json", "w"), indent=2)
rows = []
for k, v in out.items():
    for s, p in v["point"].items():
        rows.append({"analysis": k, "scheme": s, "MAE": p["MAE"], "MAE_lo": v["ci"][s]["MAE"][0], "MAE_hi": v["ci"][s]["MAE"][1], "R2": p["R2"], "R2_lo": v["ci"][s]["R2"][0], "R2_hi": v["ci"][s]["R2"][1]})
pd.DataFrame(rows).to_csv(RD / "phaseC_cluster_bootstrap_metrics.csv", index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False))
print()
for k, v in out.items():
    for r, q in v["ratios"].items():
        print(f"{k:<28} MAE {r:<18} {q['point']:.3f}  95% CI [{q['ci95'][0]:.3f}, {q['ci95'][1]:.3f}]  (resampling unit: {q['unit']})")
