"""Stage 2: interlayer binding energy and distance of homobilayers with MACE-MPA-0 (+D3). Runs in the MLIP environment.

Conventions (fixed, see results/mlip/stage2_preregistration.md):
  E_b [meV/A^2] = -(E_bilayer - 2*E_monolayer) / A * 1000, per INTERFACE (a bilayer has exactly one), positive = bound.
  Not divided by the number of layers. The in-plane cell is the monolayer's own relaxed cell (layers stay commensurate).
  d_int  = distance between the z-centroids of the two layers [A];   gap = min z(layer 2) - max z(layer 1) [A].
Stacking search: 4 orientations (identity, 180-degree rotation about z, z-mirror, mirror+rotation) x 4x4 in-plane shifts x 5 gaps
(single points); the 3 lowest-energy configurations are relaxed (ions only, cell fixed) and the lowest relaxed energy is kept.

  python -m src.mlip.interlayer --dispersion on
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import os
import time
import traceback

import numpy as np
import torch
from ase import Atoms
from ase.filters import FrechetCellFilter
from ase.optimize import LBFGS

from .calculators import build, versions
from .stiffness import orient

OUT = "results/mlip"
VACUUM = 15.0
OPS = ("id", "rot180", "mirror", "mirror_rot")
SHIFTS = [0.0, 0.25, 0.5, 0.75]
GAPS = [2.6, 3.0, 3.4, 3.8, 4.4]


@dataclasses.dataclass
class S2Settings:
    model: str = "mace_mpa0_medium"
    dispersion: bool = True
    fmax_mono: float = 0.005
    fmax_bi: float = 0.01
    max_steps: int = 500
    n_relax: int = 3


def slab(mono: Atoms):
    z = mono.positions[:, 2]
    return z - z.min(), float(z.max() - z.min())


def stack(mono: Atoms, op: str, fx: float, fy: float, gap: float) -> Atoms:
    """Homobilayer from a periodic monolayer (a || x). Layer 1 at the bottom; layer 2 = transformed copy above it."""
    z0, ext = slab(mono)
    xy = mono.positions[:, :2]
    a, b = mono.cell[0, :2], mono.cell[1, :2]
    xy2, z2 = xy.copy(), z0.copy()
    if op in ("rot180", "mirror_rot"):
        xy2 = -xy2
    if op in ("mirror", "mirror_rot"):
        z2 = ext - z2
    xy2 = xy2 + fx * a + fy * b
    n = len(mono)
    L = 2 * ext + gap + VACUUM
    pos = np.vstack([np.column_stack([xy, z0]), np.column_stack([xy2, z2 + ext + gap])])
    cell = mono.cell.array.copy()
    cell[2] = [0, 0, L]
    return Atoms(np.r_[mono.numbers, mono.numbers], cell=cell, positions=pos, pbc=True), n, ext


def relax(atoms, calc, fmax, steps, cell=False):
    atoms.calc = calc
    obj = FrechetCellFilter(atoms, mask=[1, 1, 0, 0, 0, 1]) if cell else atoms
    opt = LBFGS(obj, logfile=None)
    conv = opt.run(fmax=fmax, steps=steps)
    return {"converged": bool(conv), "steps": int(opt.get_number_of_steps())}


def layer_metrics(atoms: Atoms, n: int):
    z = atoms.positions[:, 2]
    z1, z2 = z[:n], z[n:]
    return {"d_int": float(z2.mean() - z1.mean()), "gap": float(z2.min() - z1.max())}


def energy_stress_check_bilayer(atoms, calc, d=1e-3):
    """Same self-check as the monolayer work, on a bilayer with the dispersion calculator: stress == dE/d(strain)/V."""
    from .stiffness import strained
    a = strained(strained(atoms, 0, 0.02), 1, 0.01)
    a.calc = calc
    sig, V = a.get_stress(), a.get_volume()
    out = {}
    for name, v in (("xx", 0), ("yy", 1)):
        ep, em = strained(a, v, d), strained(a, v, -d)
        ep.calc = em.calc = calc
        fd = (ep.get_potential_energy() - em.get_potential_energy()) / (2 * d) / V
        out[name] = {"analytic": float(sig[v]), "finite_diff": float(fd), "ratio": float(sig[v] / fd) if abs(fd) > 1e-9 else None}
    return out


def run_material(item: dict, calc, s: S2Settings) -> dict:
    t0 = time.time()
    from .structures import load_item
    mono = load_item(item)
    area = float(np.linalg.norm(np.cross(mono.cell[0], mono.cell[1])))
    mono.calc = calc
    rm = relax(mono, calc, s.fmax_mono, s.max_steps, cell=True)         # monolayer: ions + in-plane cell
    e_mono = float(mono.get_potential_energy())
    a_ml = float(np.linalg.norm(mono.cell[0]))
    # --- single-point stacking scan
    scan = []
    for op, fx, fy in itertools.product(OPS, SHIFTS, SHIFTS):
        best = None
        for g in GAPS:
            bi, n, ext = stack(mono, op, fx, fy, g)
            bi.calc = calc
            e = float(bi.get_potential_energy())
            if best is None or e < best[0]:
                best = (e, g)
        scan.append({"op": op, "fx": fx, "fy": fy, "e": best[0], "gap0": best[1]})
    scan.sort(key=lambda r: r["e"])
    # --- relax the lowest-energy configurations
    results = []
    for r in scan[: s.n_relax]:
        bi, n, ext = stack(mono, r["op"], r["fx"], r["fy"], r["gap0"])
        rr = relax(bi, calc, s.fmax_bi, s.max_steps, cell=False)
        e_bi = float(bi.get_potential_energy())
        results.append({"op": r["op"], "fx": r["fx"], "fy": r["fy"], "E_b_meV_A2": -(e_bi - 2 * e_mono) / area * 1000, **rr,
                        **layer_metrics(bi, n), "e_bi": e_bi})
    best = max(results, key=lambda r: r["E_b_meV_A2"])
    # spread over single-point candidates as an indication of the stacking dependence (unrelaxed, at best gap)
    e_all = np.array([r["e"] for r in scan])
    return {"mono": {"e": e_mono, "a_relaxed": a_ml, "area": area, **rm}, "relaxed_candidates": results, "best": best,
            "scan_energy_range_meV_A2": float((e_all.max() - e_all.min()) / area * 1000), "n_scan": len(scan),
            "scan_best3": scan[:3], "seconds": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dispersion", choices=["on", "off"], default="on")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ids", nargs="+")
    ap.add_argument("--inputs", default=f"{OUT}/interlayer_inputs.json")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    s = S2Settings(dispersion=args.dispersion == "on")
    items = json.load(open(args.inputs))
    if args.ids:
        items = [i for i in items if i["id"] in set(args.ids)]
    if args.limit:
        items = items[: args.limit]
    path = f"{OUT}/raw_interlayer_{s.model}_disp{args.dispersion}{args.tag}.jsonl"
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["id"] for l in open(path) if l.strip()}
    todo = [i for i in items if i["id"] not in done]
    print(f"interlayer {s.model} dispersion={args.dispersion}: {len(done)} done, {len(todo)} to do", flush=True)
    calc = build(s.model, dispersion=s.dispersion)
    ver = versions()
    # self-check on a bilayer built from the first structure (dispersion calculator included)
    from .structures import load_item
    mono = load_item(items[0])
    bi, _, _ = stack(mono, "id", 0.0, 0.0, 3.4)
    chk = energy_stress_check_bilayer(bi, calc)
    json.dump({"settings": dataclasses.asdict(s), "check": chk, "versions": ver}, open(f"{OUT}/consistency_interlayer_disp{args.dispersion}.json", "w"), indent=2)
    for k, v in chk.items():
        if v["ratio"] is not None and not (0.97 <= v["ratio"] <= 1.03):
            raise SystemExit(f"energy/stress consistency FAILED (dispersion={args.dispersion}) {k}: {v}")
    print("consistency ok", {k: round(v["ratio"], 4) for k, v in chk.items()}, flush=True)
    with open(path, "a") as f:
        for n, it in enumerate(todo):
            torch.cuda.reset_peak_memory_stats()
            rec = {"id": it["id"], "source": it["source"], "formula": it["formula"], "nat": it["nat"], "model": s.model,
                   "settings": dataclasses.asdict(s), "versions": ver, "ref": it["ref"]}
            try:
                rec["result"] = run_material(it, calc, s)
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"
                rec["trace"] = traceback.format_exc()[-600:]
            rec["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / 1e6
            f.write(json.dumps(rec, default=float) + "\n")
            f.flush()
            if (n + 1) % 5 == 0:
                print(f"  {n + 1}/{len(todo)}", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
