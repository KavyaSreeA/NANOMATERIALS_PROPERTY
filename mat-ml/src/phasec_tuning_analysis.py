"""Phase C2 analysis: tuned vs default LightGBM (nested, group-aware), cluster bootstrap over families (2000 resamples).   python -m src.phasec_tuning_analysis"""
import json

import numpy as np
import pandas as pd

from . import data as D

cfg = D.load_config()
RD = D.ROOT / cfg["paths"]["results_dir"]
fam = pd.read_pickle(D.ROOT / "cache" / "c2db_all.pkl").set_index("uid").family
rows = []
for r in map(json.loads, open(D.ROOT / "cache" / "phasec_tuning.jsonl")):
    for u, y, a, b in zip(r["uid"], r["y"], r["pred_tuned"], r["pred_default_refit"]):
        rows.append((r["scheme"], u, y, a, b))
d = pd.DataFrame(rows, columns=["scheme", "uid", "y", "tuned", "default"])
d["family"] = d.uid.map(fam)
d["et"], d["ed"] = (d.tuned - d.y).abs(), (d.default - d.y).abs()
rng = np.random.default_rng(0)
codes = {s: pd.factorize(g.family)[0] for s, g in d.groupby("scheme")}
S = {s: g.reset_index(drop=True) for s, g in d.groupby("scheme")}
rows_of = {s: [np.flatnonzero(codes[s] == i) for i in range(codes[s].max() + 1)] for s in S}
# families overlap between schemes (same materials) -> resample the same family set for both
allfam = sorted(set(d.family))
idx = {s: {f: np.flatnonzero(S[s].family.to_numpy() == f) for f in allfam if (S[s].family == f).any()} for s in S}
res = {"tuned": [], "default": [], "rt": [], "rd": [], "dt": [], "dd": []}
for _ in range(2000):
    pick = rng.choice(allfam, len(allfam))
    m = {}
    for s in S:
        ii = np.concatenate([idx[s][f] for f in pick])
        m[s] = (S[s].et.to_numpy()[ii].mean(), S[s].ed.to_numpy()[ii].mean())
    res["rt"].append(m["family"][0] / m["random"][0]); res["rd"].append(m["family"][1] / m["random"][1])
    res["dt"].append(m["random"][1] - m["random"][0]); res["dd"].append(m["family"][1] - m["family"][0])
ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
pt = {s: (float(S[s].et.mean()), float(S[s].ed.mean())) for s in S}
out = {"MAE_tuned_default": pt, "ratio_family_random_tuned": [pt["family"][0] / pt["random"][0], ci(res["rt"])],
       "ratio_family_random_default": [pt["family"][1] / pt["random"][1], ci(res["rd"])],
       "gain_random(default-tuned)": [pt["random"][1] - pt["random"][0], ci(res["dt"])],
       "gain_family(default-tuned)": [pt["family"][1] - pt["family"][0], ci(res["dd"])]}
json.dump(out, open(RD / "phaseC_tuning.json", "w"), indent=2)
print(json.dumps(out, indent=2))
