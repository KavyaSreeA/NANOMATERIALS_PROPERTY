"""Phase C2 analysis for Task B: tuned vs default LightGBM on BiDB binding energy; cluster bootstrap over monolayers (random/monolayer schemes) and over families for the family scheme
and for the family/random ratio (resampling families; random rows follow their family).  python -m src.phasec_tuning_taskb_analysis"""
import json

import numpy as np
import pandas as pd

from . import data as D

cfg = D.load_config()
RD = D.ROOT / cfg["paths"]["results_dir"]
rows = []
for r in map(json.loads, open(D.ROOT / "cache" / "phasec_tuning_taskb.jsonl")):
    for m, f, y, a, b in zip(r["monolayer"], r["family"], r["y"], r["pred_tuned"], r["pred_default_refit"]):
        rows.append((r["scheme"], m, f, y, abs(a - y), abs(b - y)))
d = pd.DataFrame(rows, columns=["scheme", "mono", "family", "y", "et", "ed"])
S = ["random", "monolayer", "family"]
point = {s: (float(d[d.scheme == s].et.mean()), float(d[d.scheme == s].ed.mean())) for s in S}
rng = np.random.default_rng(0)
fams = sorted(set(d.family))
g = {s: {f: x for f, x in d[d.scheme == s].groupby("family")[["et", "ed"]]} for s in S}
res = {k: [] for k in ("rt", "rd", "mt", "md", "gain_r", "gain_m", "gain_f")}
for _ in range(2000):
    pick = rng.choice(fams, len(fams))
    m = {}
    for s in S:
        x = pd.concat([g[s][f] for f in pick if f in g[s]])
        m[s] = (x.et.mean(), x.ed.mean())
    res["rt"].append(m["family"][0] / m["random"][0]); res["rd"].append(m["family"][1] / m["random"][1])
    res["mt"].append(m["monolayer"][0] / m["random"][0]); res["md"].append(m["monolayer"][1] / m["random"][1])
    res["gain_r"].append(m["random"][1] - m["random"][0]); res["gain_m"].append(m["monolayer"][1] - m["monolayer"][0]); res["gain_f"].append(m["family"][1] - m["family"][0])
ci = lambda k: [float(np.percentile(res[k], 2.5)), float(np.percentile(res[k], 97.5))]
out = {"target": "binding_energy_zscan, mono_stiffness features, LightGBM", "MAE_tuned_default": point,
       "ratio_family_random_tuned": [point["family"][0] / point["random"][0], ci("rt")], "ratio_family_random_default": [point["family"][1] / point["random"][1], ci("rd")],
       "ratio_monolayer_random_tuned": [point["monolayer"][0] / point["random"][0], ci("mt")], "ratio_monolayer_random_default": [point["monolayer"][1] / point["random"][1], ci("md")],
       "gain_default_minus_tuned": {"random": [point["random"][1] - point["random"][0], ci("gain_r")], "monolayer": [point["monolayer"][1] - point["monolayer"][0], ci("gain_m")],
                                    "family": [point["family"][1] - point["family"][0], ci("gain_f")]},
       "note": "resampling unit = structure family (monolayers nest within families, so this is the conservative choice)"}
json.dump(out, open(RD / "phaseC_tuning_taskb.json", "w"), indent=2)
print(json.dumps(out, indent=2))
