"""Phase A, multi-seed addendum: CGCNN initialisation/validation seeds 42, 43, 44 on the SAME folds (cache/phaseA_cgcnn_oof{,_s43,_s44}.jsonl).
Reports per-seed MAE and family/random ratio, and the paired cluster bootstrap (over families, 2,000 resamples) for the 3-seed ensemble (mean of seed predictions) vs LightGBM.
python -m src.gnn.analysis_seeds"""
import json

import numpy as np
import pandas as pd

from .. import data as D

cfg = D.load_config()
RD = D.ROOT / cfg["paths"]["results_dir"]
rng = np.random.default_rng(0)
NB = 2000
TAGS = {42: "", 43: "_s43", 44: "_s44"}
S = ["random", "chemsys", "family"]
L = pd.read_csv(RD / "diagnostics" / "taskA_oof_predictions.csv")
L = L[L.target == "Y2D"][["uid", "scheme", "family", "pred"]].rename(columns={"pred": "lgbm"})
frames = []
for sd, tg in TAGS.items():
    recs = [json.loads(l) for l in open(D.ROOT / "cache" / f"phaseA_cgcnn_oof{tg}.jsonl")]
    assert len(recs) == 15, (sd, len(recs))
    frames.append(pd.DataFrame([{"uid": u, "scheme": r["scheme"], "y": y, f"p{sd}": p} for r in recs for u, y, p in zip(r["uid"], r["y"], r["pred"])]))
M = frames[0]
for f in frames[1:]:
    M = M.merge(f.drop(columns="y"), on=["uid", "scheme"])
M = M.merge(L, on=["uid", "scheme"])
M["ens"] = M[[f"p{s}" for s in TAGS]].mean(axis=1)
by = {s: M[M.scheme == s].reset_index(drop=True) for s in S}
for s in S:
    by[s] = by[s].set_index("uid").loc[by["random"].uid].reset_index()
codes, uniq = pd.factorize(by["random"].family.values)
rows_of = [np.flatnonzero(codes == i) for i in range(len(uniq))]
mae = lambda d, c, idx=None: float(np.abs((d if idx is None else d.iloc[idx])[c] - (d if idx is None else d.iloc[idx]).y).mean())
cols = [f"p{s}" for s in TAGS] + ["ens", "lgbm"]
res = {"per_seed": {c: {"MAE": {s: mae(by[s], c) for s in S}, "ratio_family_over_random": mae(by["family"], c) / mae(by["random"], c)} for c in cols}}
dr = {"ratio_ens": [], "ratio_lgbm": [], "diff_ratio": [], "gap_family": [], "gap_random": []}
for _ in range(NB):
    idx = np.concatenate([rows_of[i] for i in rng.integers(0, len(rows_of), len(rows_of))])
    r_e = mae(by["family"], "ens", idx) / mae(by["random"], "ens", idx)
    r_l = mae(by["family"], "lgbm", idx) / mae(by["random"], "lgbm", idx)
    dr["ratio_ens"].append(r_e); dr["ratio_lgbm"].append(r_l); dr["diff_ratio"].append(r_e - r_l)
    dr["gap_family"].append(mae(by["family"], "lgbm", idx) - mae(by["family"], "ens", idx))
    dr["gap_random"].append(mae(by["random"], "lgbm", idx) - mae(by["random"], "ens", idx))
ci = lambda k: [float(np.percentile(dr[k], 2.5)), float(np.percentile(dr[k], 97.5))]
res["ensemble"] = {"ratio": res["per_seed"]["ens"]["ratio_family_over_random"], "ratio_ci95": ci("ratio_ens"), "lgbm_ratio_ci95": ci("ratio_lgbm"),
                   "diff_ratio_ens_minus_lgbm": res["per_seed"]["ens"]["ratio_family_over_random"] - res["per_seed"]["lgbm"]["ratio_family_over_random"], "diff_ci95": ci("diff_ratio"),
                   "paired_MAE_lgbm_minus_ens_family": {"point": res["per_seed"]["lgbm"]["MAE"]["family"] - res["per_seed"]["ens"]["MAE"]["family"], "ci95": ci("gap_family")},
                   "paired_MAE_lgbm_minus_ens_random": {"point": res["per_seed"]["lgbm"]["MAE"]["random"] - res["per_seed"]["ens"]["MAE"]["random"], "ci95": ci("gap_random")}}
sd_ratio = float(np.std([res["per_seed"][f"p{s}"]["ratio_family_over_random"] for s in TAGS]))
res["seed_sd_of_ratio"] = sd_ratio
json.dump(res, open(RD / "phaseA_cgcnn_seeds.json", "w"), indent=2)
print(json.dumps(res, indent=2))
