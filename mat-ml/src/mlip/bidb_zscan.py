"""Reproduce BiDB's rigid z-scan with MACE-MPA-0 + D3 on the stored BiDB geometries. Runs in the MLIP environment. Resumable.

Procedure and criteria are fixed in results/mlip/bidb_preregistration.md.
  E_b [meV/A^2] = -(E_min - 2*E_mono)/A * 1000   (per interface, positive = bound);  d = gap at the minimum (A).
Geometries are NOT relaxed (BiDB's stored bilayers are rigid copies of the C2DB monolayer); the top layer is shifted vertically.

  python -m src.mlip.bidb_zscan            # the 307-bilayer validation set
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
import traceback

import numpy as np
import pandas as pd
import torch
from ase import Atoms

from .calculators import build, versions

OUT = "results/mlip"
VACUUM = 20.0
COARSE = np.arange(-1.0, 6.0 + 1e-9, 0.25)


@dataclasses.dataclass
class ZSettings:
    model: str = "mace_mpa0_medium"
    dispersion: bool = True
    vacuum: float = VACUUM
    coarse: tuple = (-1.0, 6.0, 0.25)
    fine_halfwidth: float = 0.25
    fine_step: float = 0.02


def atoms_from(entry: dict, n_bottom: int | None = None, gap: float | None = None, vacuum: float = VACUUM) -> Atoms:
    """Atoms from a BiDB entry; for a bilayer set the vertical gap (top layer shifted rigidly). Fully periodic cell, >= `vacuum` A of vacuum."""
    pos = np.array(entry["positions"], float)
    num = np.array(entry["numbers"], int)
    cell = np.array(entry["cell"], float)
    if gap is not None:
        g0 = pos[n_bottom:, 2].min() - pos[:n_bottom, 2].max()
        pos[n_bottom:, 2] += gap - g0
    z = pos[:, 2]
    ext = z.max() - z.min()
    pos[:, 2] = z - z.min() + vacuum / 2
    c = cell.copy()
    c[0, 2] = c[1, 2] = 0.0
    c[2] = [0.0, 0.0, ext + vacuum]
    return Atoms(num, positions=pos, cell=c, pbc=True)


def area(entry) -> float:
    c = np.array(entry["cell"])
    return float(np.linalg.norm(np.cross(c[0], c[1])))


def energy(entry, n, g, calc, vacuum):
    a = atoms_from(entry, n, g, vacuum)
    a.calc = calc
    return float(a.get_potential_energy())


def zscan(entry, n, calc, s: ZSettings):
    gaps_c = np.arange(s.coarse[0], s.coarse[1] + 1e-9, s.coarse[2])
    ec = np.array([energy(entry, n, g, calc, s.vacuum) for g in gaps_c])
    i = int(np.argmin(ec))
    lo, hi = gaps_c[i] - s.fine_halfwidth, gaps_c[i] + s.fine_halfwidth
    gaps_f = np.arange(lo, hi + 1e-9, s.fine_step)
    ef = np.array([energy(entry, n, g, calc, s.vacuum) for g in gaps_f])
    j = int(np.argmin(ef))
    return {"gap": float(gaps_f[j]), "e_min": float(ef[j]), "edge_high": bool(i == len(gaps_c) - 1), "edge_low": bool(i == 0),
            "coarse_gaps": gaps_c.tolist(), "coarse_e": ec.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--uids", nargs="+")
    ap.add_argument("--dispersion", choices=["on", "off"], default="on")
    args = ap.parse_args()
    s = ZSettings(dispersion=args.dispersion == "on")
    sel = pd.read_csv(f"{OUT}/bidb_validation_set.csv")
    if args.uids:
        sel = sel[sel.uid.isin(args.uids)]
    if args.limit:
        sel = sel.head(args.limit)
    S = {e["uid"]: e for e in json.load(open("../Dataset/bidb_structures.json"))}
    path = f"{OUT}/raw_bidb_zscan_{s.model}_disp{args.dispersion}.jsonl"
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["uid"] for l in open(path) if l.strip()}
    todo = sel[~sel.uid.isin(done)]
    print(f"BiDB z-scan {s.model} dispersion={args.dispersion}: {len(done)} done, {len(todo)} to do", flush=True)
    calc = build(s.model, dispersion=s.dispersion)
    ver = versions()
    emono = {}
    t_all = time.time()
    with open(path, "a") as f:
        for k, r in enumerate(todo.itertuples()):
            torch.cuda.reset_peak_memory_stats()
            rec = {"uid": r.uid, "monolayer_uid": r.monolayer_uid, "formula": r.formula, "model": s.model, "settings": dataclasses.asdict(s),
                   "versions": ver, "ref": {"distance": r.distance, "zscan": r.binding_energy_zscan, "gs": r.binding_energy_gs,
                                            "slide_stability": r.slide_stability, "ref_valid": bool(r.ref_valid)}}
            t0 = time.time()
            try:
                bi = S[r.uid]
                mono = S[r.monolayer_uid]
                n = len(mono["numbers"])
                assert len(bi["numbers"]) == 2 * n, "bilayer natoms != 2 x monolayer natoms"
                if r.monolayer_uid not in emono:
                    a = atoms_from(mono, vacuum=s.vacuum)
                    a.calc = calc
                    emono[r.monolayer_uid] = float(a.get_potential_energy())
                A = area(bi)
                z = zscan(bi, n, calc, s)
                e_far = energy(bi, n, 6.0, calc, s.vacuum)
                rec["result"] = {"gap": z["gap"], "E_b_meV_A2": -(z["e_min"] - 2 * emono[r.monolayer_uid]) / A * 1000, "e_min": z["e_min"],
                                 "e_mono": emono[r.monolayer_uid], "area": A, "edge_high": z["edge_high"], "edge_low": z["edge_low"],
                                 "sanity_E_at_6A_meV_A2": -(e_far - 2 * emono[r.monolayer_uid]) / A * 1000,
                                 "coarse_gaps": z["coarse_gaps"], "coarse_e_rel_meV_A2": [(e - 2 * emono[r.monolayer_uid]) / A * 1000 for e in z["coarse_e"]]}
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"
                rec["trace"] = traceback.format_exc()[-500:]
            rec["wall_s"] = time.time() - t0
            f.write(json.dumps(rec, default=float) + "\n")
            f.flush()
            if (k + 1) % 20 == 0:
                print(f"  {k + 1}/{len(todo)}  {time.time() - t_all:.0f}s", flush=True)
    print(f"finished in {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
