"""Phase B-2: replicate the property panel on JARVIS-DFT 2D (design: results/panel_replication_design.md, written before any fit).

  python -m src.panel_jarvis            # ~10-15 min     then   python -m src.panel_analysis --oof panel_oof_jarvis --tag jarvis
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from . import data as D
from .features import build_features
from .panel import run_one

MIN_LABELS = 500
SKIP = {"kpoint_length_unit", "encut", "maxdiff_mesh", "maxdiff_bz", "density", "nat", "optb88vdw_total_energy", "ehull", "spg_number"}
# one representative per property family (rules in the design file)
FAMILY_REP = {"epsx": "dielectric", "epsy": "dielectric", "epsz": "dielectric", "mepsx": "dielectric_mbj", "mepsy": "dielectric_mbj", "mepsz": "dielectric_mbj",
              "avg_elec_mass": "mass", "avg_hole_mass": "mass", "n-Seebeck": "seebeck", "p-Seebeck": "seebeck", "n-powerfact": "powerfact", "p-powerfact": "powerfact",
              "ncond": "cond", "pcond": "cond", "nkappa": "kappa", "pkappa": "kappa", "magmom_oszicar": "magmom", "magmom_outcar": "magmom"}
KEEP = {"epsx", "avg_elec_mass", "n-Seebeck", "n-powerfact", "ncond", "nkappa", "magmom_oszicar"}


def load_jarvis_all(cfg):
    recs = json.load(open(D.data_path(cfg, cfg["paths"]["jarvis_json"]), encoding="utf-8"))
    rows = []
    for r in recs:
        at = r["atoms"]
        lat = np.array(at["lattice_mat"], float)
        co = np.array(at["coords"], float)
        frac = co @ np.linalg.inv(lat) if at.get("cartesian") else co
        row = {k: float(v) for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)}
        row.update({"uid": r["jid"], "nat": len(at["elements"]), "n_layers": 1, "spg_number": r["spg_number"]})
        row.update(D.composition_info(list(at["elements"])))
        row.update(D.slab_geometry(lat, frac))
        rows.append(row)
    df = pd.DataFrame(rows)
    df["family"] = df.anon_formula + "|spg" + df.spg_number.astype(int).astype(str)
    return df


def select_targets(df):
    inc, exc = [], {}
    for t in df.columns:
        if t in SKIP or t in ("uid", "spg_number", "n_layers", "a", "b", "gamma", "area", "cell_height", "z_extent", "n_elements") or df[t].dtype.kind not in "fi":
            continue
        s = df[t].dropna()
        if len(s) < MIN_LABELS:
            continue
        if t in FAMILY_REP and t not in KEEP:
            exc[t] = f"same property family as a kept target ({FAMILY_REP[t]})"
            continue
        mode = s.round(6).mode().iloc[0]
        if (np.abs(s - mode) < 1e-6).mean() > 0.5:
            exc[t] = "degenerate (>50% at the modal value)"
            continue
        log = bool((s > 0).all() and s.max() / s.min() > 100)
        inc.append({"target": t, "n": int(len(s)), "transform": "log" if log else "none", "n_families": int(df.loc[s.index, "family"].nunique()),
                    "n_chemsys": int(df.loc[s.index, "chemsys"].nunique())})
    return inc, exc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = ap.parse_args()
    cfg = D.load_config()
    t0 = time.time()
    df = load_jarvis_all(cfg)
    inc, exc = select_targets(df)
    json.dump({"included": inc, "excluded": exc, "n_materials": len(df), "family_definition": "anonymous formula + space-group number"},
              open(D.ROOT / cfg["paths"]["results_dir"] / "panel_targets_jarvis.json", "w"), indent=2)
    print(len(df), "materials; panel:", [(i["target"], i["n"], i["transform"], i["n_families"]) for i in inc], "\nexcluded:", exc, flush=True)
    X, _ = build_features(df, cfg)
    out = D.ROOT / "cache" / "panel_oof_jarvis"
    out.mkdir(parents=True, exist_ok=True)
    for i in inc:
        for seed in args.seeds:
            for sc in ("random", "chemsys", "family"):
                run_one(df, X, i["target"], i["transform"], sc, seed, cfg, out)
            print(f"  {i['target']:<22} seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
