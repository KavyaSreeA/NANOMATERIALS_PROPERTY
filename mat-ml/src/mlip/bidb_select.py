"""Select the BiDB validation set (reference-side information only; no MACE result is used). Seed fixed.

Eligibility : monolayer non-magnetic, monolayer natoms <= 6, >= 5 bilayers with a valid reference.
Valid ref   : distance > 0 A, 0 < binding_energy_zscan <= U, where U = Q3 + 3*IQR of log10(binding_energy_zscan) over ALL non-magnetic BiDB
              bilayers (BiDB-only statistic, the documented log-IQR outlier rule of this project, k = 3).
Sample      : 30 monolayers uniformly at random (seed 42); ALL their bilayers are run (valid or not; invalid references are counted and
              excluded from reference-based metrics only).
Stiffness   : for each sampled monolayer, the valid bilayer with the highest and the one with the lowest BiDB binding_energy_zscan.
Runs in the MLIP environment (numpy/pandas).
"""
import argparse
import json

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--n-mono", type=int, default=30)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()
OUT = "results/mlip"
p = pd.read_csv("../Dataset/bidb_properties.csv", low_memory=False)
mono = p[p.number_of_layers == 1].set_index("monolayer_uid")
bi = p[p.number_of_layers == 2].copy()
nm = bi[(bi.magnetic == False)]  # noqa: E712
ls = np.log10(nm.binding_energy_zscan[nm.binding_energy_zscan > 0])
q1, q3 = ls.quantile(.25), ls.quantile(.75)
U = float(10 ** (q3 + 3 * (q3 - q1)))
bi["ref_valid"] = (bi.distance > 0) & (bi.binding_energy_zscan > 0) & (bi.binding_energy_zscan <= U)
info = {"bidb_bilayers": len(bi), "nonmagnetic_bilayers": len(nm), "zscan_upper_bound_meV_A2": U,
        "ref_valid_all_bilayers": int(bi.ref_valid.sum()), "ref_invalid_all_bilayers": int((~bi.ref_valid).sum())}
elig = []
for muid, g in bi.groupby("monolayer_uid"):
    m = mono.loc[muid]
    if bool(m.magnetic) or m.natoms > 6 or g.magnetic.any():
        continue
    if g.ref_valid.sum() >= 5:
        elig.append(muid)
info["eligible_monolayers"] = len(elig)
rng = np.random.default_rng(args.seed)
chosen = sorted(rng.choice(elig, size=min(args.n_mono, len(elig)), replace=False).tolist())
sel = bi[bi.monolayer_uid.isin(chosen)].copy()
sel["stiffness_pick"] = ""
for muid, g in sel.groupby("monolayer_uid"):
    v = g[g.ref_valid]
    sel.loc[v.binding_energy_zscan.idxmax(), "stiffness_pick"] = "best"
    sel.loc[v.binding_energy_zscan.idxmin(), "stiffness_pick"] = sel.loc[v.binding_energy_zscan.idxmin(), "stiffness_pick"] + "+worst" if sel.loc[v.binding_energy_zscan.idxmin(), "stiffness_pick"] else "worst"
info.update({"seed": args.seed, "selected_monolayers": len(chosen), "selected_bilayers": len(sel), "selected_ref_valid": int(sel.ref_valid.sum()),
             "selected_ref_invalid": int((~sel.ref_valid).sum()), "stiffness_bilayers": int((sel.stiffness_pick != "").sum())})
cols = ["uid", "monolayer_uid", "formula", "natoms", "binding_energy_zscan", "binding_energy_gs", "distance", "slide_stability", "ref_valid", "stiffness_pick"]
sel[cols].to_csv(f"{OUT}/bidb_validation_set.csv", index=False)
json.dump({"monolayers": chosen, "info": info}, open(f"{OUT}/bidb_selection.json", "w"), indent=2)
print(json.dumps(info, indent=2)); print(sel.groupby("monolayer_uid").agg(formula=("formula", "first"), n=("uid", "size"), n_valid=("ref_valid", "sum")).to_string())
