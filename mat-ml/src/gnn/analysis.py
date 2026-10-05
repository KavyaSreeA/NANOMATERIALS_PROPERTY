"""Phase A analysis: CGCNN vs LightGBM on identical folds (design: results/phaseA_design.md).   python -m src.gnn.analysis   (Task A environment)

Cluster bootstrap over structure families (paired: the same resampled families for both models and for both schemes in a ratio), 2,000 resamples.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from .. import data as D

cfg = D.load_config()
RD = D.ROOT / cfg["paths"]["results_dir"]
rng = np.random.default_rng(0)
NB = 2000

recs = [json.loads(l) for l in open(D.ROOT / "cache" / "phaseA_cgcnn_oof.jsonl")]
G = pd.DataFrame([{"uid": u, "scheme": r["scheme"], "fold": r["fold"], "y": y, "cgcnn": p} for r in recs for u, y, p in zip(r["uid"], r["y"], r["pred"])])
L = pd.read_csv(RD / "diagnostics" / "taskA_oof_predictions.csv")
L = L[L.target == "Y2D"][["uid", "scheme", "family", "pred"]].rename(columns={"pred": "lgbm"})
M = G.merge(L, on=["uid", "scheme"])
assert len(M) == len(G) == 3 * 7258, (len(M), len(G))
epochs = pd.DataFrame([{"scheme": r["scheme"], "fold": r["fold"], "epochs_used": r["epochs_used"], "test_mae": r["test_mae"]} for r in recs])
S = ["random", "chemsys", "family"]
by = {s: M[M.scheme == s].reset_index(drop=True) for s in S}
# align rows across schemes by uid so that numerator and denominator of a ratio use the same resampled rows
for s in S:
    by[s] = by[s].set_index("uid").loc[by["random"].uid].reset_index()
fam = by["random"].family.values
codes, uniq = pd.factorize(fam)
rows_of = [np.flatnonzero(codes == i) for i in range(len(uniq))]


def mae(d, col, idx=None):
    d = d if idx is None else d.iloc[idx]
    return float(np.abs(d[col] - d.y).mean())


point = {(s, m): mae(by[s], m) for s in S for m in ("lgbm", "cgcnn")}
draws = {k: [] for k in ("diff_family", "diff_random", "diff_chemsys", "ratio_lgbm", "ratio_cgcnn", "ratio_diff", "mae_cgcnn_family", "mae_cgcnn_random", "mae_lgbm_family", "mae_lgbm_random")}
for _ in range(NB):
    idx = np.concatenate([rows_of[i] for i in rng.integers(0, len(rows_of), len(rows_of))])
    m = {(s, c): mae(by[s], c, idx) for s in S for c in ("lgbm", "cgcnn")}
    draws["diff_family"].append(m["family", "lgbm"] - m["family", "cgcnn"])
    draws["diff_random"].append(m["random", "lgbm"] - m["random", "cgcnn"])
    draws["diff_chemsys"].append(m["chemsys", "lgbm"] - m["chemsys", "cgcnn"])
    draws["ratio_lgbm"].append(m["family", "lgbm"] / m["random", "lgbm"])
    draws["ratio_cgcnn"].append(m["family", "cgcnn"] / m["random", "cgcnn"])
    draws["ratio_diff"].append(draws["ratio_cgcnn"][-1] - draws["ratio_lgbm"][-1])
    for k, (s, c) in {"mae_cgcnn_family": ("family", "cgcnn"), "mae_cgcnn_random": ("random", "cgcnn"), "mae_lgbm_family": ("family", "lgbm"), "mae_lgbm_random": ("random", "lgbm")}.items():
        draws[k].append(m[s, c])
ci = lambda k: [float(np.percentile(draws[k], 2.5)), float(np.percentile(draws[k], 97.5))]
res = {"MAE_N_per_m": {f"{s}|{m}": point[s, m] for s in S for m in ("lgbm", "cgcnn")},
       "R2": {f"{s}|{m}": float(r2_score(by[s].y, by[s][m])) for s in S for m in ("lgbm", "cgcnn")},
       "ci95_MAE": {f"{s}|{m}": ci(f"mae_{m}_{s}") for s in ("random", "family") for m in ("lgbm", "cgcnn")},
       "ratio_family_over_random": {"lgbm": point["family", "lgbm"] / point["random", "lgbm"], "lgbm_ci95": ci("ratio_lgbm"),
                                    "cgcnn": point["family", "cgcnn"] / point["random", "cgcnn"], "cgcnn_ci95": ci("ratio_cgcnn"),
                                    "diff_cgcnn_minus_lgbm": point["family", "cgcnn"] / point["random", "cgcnn"] - point["family", "lgbm"] / point["random", "lgbm"], "diff_ci95": ci("ratio_diff")},
       "paired_MAE_lgbm_minus_cgcnn": {s: {"point": point[s, "lgbm"] - point[s, "cgcnn"], "ci95": ci(f"diff_{s}")} for s in S},
       "epochs_used_mean_by_scheme": epochs.groupby("scheme").epochs_used.mean().to_dict(),
       "per_fold_test_MAE_cgcnn": {s: epochs[epochs.scheme == s].sort_values("fold").test_mae.round(2).tolist() for s in S}}
# pre-registered hypotheses
f = res["paired_MAE_lgbm_minus_cgcnn"]["family"]
res["A1_cgcnn_does_not_beat_lgbm_on_families"] = bool(f["ci95"][0] <= 0 or f["point"] < 2.0)
rr = res["ratio_family_over_random"]
res["A2_ratio_within_0.15_of_lgbm"] = bool(abs(rr["diff_cgcnn_minus_lgbm"]) <= 0.15)
res["A3_cgcnn_narrows_gap"] = bool(rr["cgcnn"] < 1.25 and rr["cgcnn_ci95"][1] < rr["lgbm"])
json.dump(res, open(RD / "phaseA_cgcnn_vs_lgbm.json", "w"), indent=2)
print(json.dumps(res, indent=2))
