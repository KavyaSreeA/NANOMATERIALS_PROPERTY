"""Select the Stage 2 reference set from JARVIS exfoliation energies (no MACE results are used; seed fixed).

Pool rule : numeric exfoliation_energy, nat <= 10, non-magnetic (|magmom_oszicar| < 0.05), structure convertible to a slab.
Sample    : N materials drawn uniformly at random (seed 42). Nothing is removed afterwards.
Reference : Eb_ref [meV/A^2] = exfoliation_energy [meV/atom] * nat / A.  The meV/atom unit is a plausibility reading, not a documented definition:
            graphene 26.8, MoS2 26.2, WS2 25.9, h-BN 26.1 meV/A^2 are in the usual vdW range.
Caveats   : bulk-derived (energy to remove one layer from the bulk, OptB88vdW), stacking of the bulk is not stored, no interlayer distance.
Runs in the MLIP environment (numpy/pandas only).
"""
import argparse
import json

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()
OUT = "results/mlip"
raw = json.load(open("../Dataset/jarvis_dft_2d.json", encoding="utf-8"))
rows = []
for r in raw:
    at = r["atoms"]
    L = np.array(at["lattice_mat"], float)
    area = float(np.linalg.norm(np.cross(L[0], L[1])))
    coords = np.array(at["coords"], float)
    cart = coords if at.get("cartesian") else coords @ L
    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan
    rows.append({"id": r["jid"], "formula": r["formula"], "nat": len(at["elements"]), "exf": num(r.get("exfoliation_energy")),
                 "mag": num(r.get("magmom_oszicar")), "area": area, "symbols": list(at["elements"]), "cell": L.tolist(), "positions": cart.tolist()})
d = pd.DataFrame(rows)
steps = {"jarvis_rows": len(d), "numeric_exfoliation_energy": int(d.exf.notna().sum())}
p = d[d.exf.notna()]
p = p[p.nat <= 10]; steps["nat<=10"] = len(p)
nm = p[(p.mag.abs() < 0.05)]; steps["non_magnetic(|magmom|<0.05)"] = len(nm)
steps["magmom_missing_among_nat<=10"] = int(p.mag.isna().sum())
pool = nm.reset_index(drop=True)
rng = np.random.default_rng(args.seed)
idx = rng.choice(len(pool), size=min(args.n, len(pool)), replace=False)
sel = pool.iloc[sorted(idx)].reset_index(drop=True)
items = []
for r in sel.itertuples():
    items.append({"id": r.id, "source": "jarvis", "formula": r.formula, "nat": int(r.nat), "symbols": r.symbols, "numbers": None, "cell": r.cell,
                  "positions": r.positions, "pbc": [True, True, False],
                  "ref": {"exf_meV_atom": r.exf, "Eb_meV_A2": r.exf * r.nat / r.area}})
json.dump(items, open(f"{OUT}/interlayer_inputs.json", "w"))
sel.assign(Eb_ref=sel.exf * sel.nat / sel.area)[["id", "formula", "nat", "exf", "Eb_ref"]].to_csv(f"{OUT}/interlayer_reference_set.csv", index=False)
steps.update({"pool": len(pool), "selected": len(sel), "seed": args.seed,
              "Eb_ref_quantiles_selected": np.round((sel.exf * sel.nat / sel.area).quantile([0, .25, .5, .75, 1]).to_numpy(), 1).tolist(),
              "Eb_ref_quantiles_pool": np.round((pool.exf * pool.nat / pool.area).quantile([0, .25, .5, .75, 1]).to_numpy(), 1).tolist()})
json.dump(steps, open(f"{OUT}/interlayer_selection.json", "w"), indent=2)
print(json.dumps(steps, indent=2)); print(sel.assign(Eb_ref=sel.exf * sel.nat / sel.area)[["id", "formula", "nat", "exf", "Eb_ref"]].round(1).to_string(index=False))
