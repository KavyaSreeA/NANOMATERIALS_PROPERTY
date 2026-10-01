"""Select a reproducible monolayer benchmark and export structures + reference values.

Runs in the Task A environment (needs src.data). Output is self-contained JSON for the MLIP environment.

C2DB rule (documented, seed fixed):
  pool   = clean C2DB rows (tensor-stable) with is_magnetic == False, ehull <= 0.1 eV/atom, nat <= 12
  sample = draw N families uniformly at random from the pool's structure families (seed 42), then one random member
           of each. Family-uniform sampling avoids over-representing the few very large families.
JARVIS: every clean (stable xx,yy block) row with nat <= 24.

  python -m src.mlip.select_benchmark --n-c2db 150
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .. import data as D
from ..data import _nd, data_path
from ..external import match_materials


def c2db_structure(cfg, path):
    import os
    base = data_path(cfg, cfg["paths"]["c2db_dir"])
    with open(os.path.join(base, path, "structure.json"), encoding="utf-8") as f:
        st = json.load(f)["1"]
    cell = _nd(st["cell"]).astype(float).reshape(3, 3)
    return {"numbers": _nd(st["numbers"]).astype(int).tolist(), "cell": cell.tolist(),
            "positions": _nd(st["positions"]).astype(float).reshape(-1, 3).tolist(), "pbc": [True, True, False]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-c2db", type=int, default=150)
    ap.add_argument("--max-nat-jarvis", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = D.load_config(args.config)
    out_dir = D.ROOT / cfg["paths"]["results_dir"] / "mlip"
    out_dir.mkdir(parents=True, exist_ok=True)

    c2, _ = D.clean_c2db(D.load_c2db(cfg), cfg)
    pool = c2[(c2.is_magnetic == False) & (c2.ehull <= 0.1) & (c2.nat <= 12)].reset_index(drop=True)  # noqa: E712
    rng = np.random.default_rng(args.seed)
    fams = np.array(sorted(pool.family.unique()))
    chosen = rng.choice(fams, size=min(args.n_c2db, len(fams)), replace=False)
    sel = []
    for f in sorted(chosen):
        g = pool[pool.family == f]
        sel.append(g.iloc[rng.integers(len(g))])
    sel = pd.DataFrame(sel).reset_index(drop=True)

    jar, _ = D.clean_jarvis(D.load_jarvis(cfg))
    jar = jar[jar.nat <= args.max_nat_jarvis].reset_index(drop=True)

    items = []
    for r in sel.itertuples():
        items.append({"id": r.uid, "source": "c2db", "formula": r.reduced_formula, "family": r.family, "chemsys": r.chemsys,
                      "nat": int(r.nat), "ehull": float(r.ehull), "is_magnetic": bool(r.is_magnetic),
                      **c2db_structure(cfg, r.path),
                      "ref": {"c11": r.c11, "c12": r.c12, "c22": r.c22, "c66": r.shear_xy, "Y2D": r.Y2D, "poisson": r.poisson}})
    raw = {r["jid"]: r for r in json.load(open(data_path(cfg, cfg["paths"]["jarvis_json"]), encoding="utf-8"))}
    for r in jar.itertuples():
        at = raw[r.jid]["atoms"]
        lat = np.array(at["lattice_mat"], float)
        coords = np.array(at["coords"], float)
        cart = coords if at.get("cartesian") else coords @ lat
        items.append({"id": r.jid, "source": "jarvis", "formula": r.reduced_formula, "family": None, "chemsys": r.chemsys,
                      "nat": int(r.nat), "ehull": None, "is_magnetic": None,
                      "numbers": None, "symbols": list(at["elements"]), "cell": lat.tolist(), "positions": cart.tolist(),
                      "pbc": [True, True, False],
                      "ref": {"c11": r.c11, "c12": r.c12, "c22": r.c22, "c66": None, "Y2D": r.Y2D, "poisson": r.poisson}})
    json.dump(items, open(out_dir / "benchmark_inputs.json", "w"))

    # reference disagreement between the two databases on materials present in both (the gate's yardstick)
    mi = match_materials(jar, c2)
    m = (mi >= 0).to_numpy()
    dis = {"n_matched": int(m.sum())}
    for q in ("c11", "c12", "c22", "Y2D", "poisson"):
        a = jar.loc[m, q].to_numpy(float)
        b = c2.loc[mi[m], q].to_numpy(float)
        dis[q] = {"mae": float(np.mean(np.abs(a - b))), "median_ratio": float(np.median(a / b)) if q != "poisson" else None}
    json.dump(dis, open(out_dir / "db_disagreement.json", "w"), indent=2)

    cov = {"pool_rows": len(pool), "pool_families": int(pool.family.nunique()), "selected_c2db": len(sel),
           "selected_families": int(sel.family.nunique()), "jarvis_rows": len(jar), "seed": args.seed,
           "Y2D_quantiles_pool": pool.Y2D.quantile([.05, .25, .5, .75, .95]).round(1).tolist(),
           "Y2D_quantiles_selected": sel.Y2D.quantile([.05, .25, .5, .75, .95]).round(1).tolist(),
           "nat_median_pool_vs_selected": [float(pool.nat.median()), float(sel.nat.median())],
           "n_elements_pool": int(len(set("-".join(pool.chemsys).split("-")))), "n_elements_selected": int(len(set("-".join(sel.chemsys).split("-")))),
           "family_size_median_pool_vs_selected_pool_size": [float(pool.family.value_counts().median()),
                                                             float(pool.family.value_counts()[sel.family].median())]}
    json.dump(cov, open(out_dir / "benchmark_selection.json", "w"), indent=2)
    sel[["uid", "reduced_formula", "family", "chemsys", "nat", "ehull", "c11", "c12", "c22", "Y2D", "poisson"]].to_csv(
        out_dir / "benchmark_c2db_subset.csv", index=False)
    print(json.dumps(cov, indent=2)); print(json.dumps(dis, indent=2))


if __name__ == "__main__":
    main()
