"""In-plane stiffness of BiDB homobilayers vs their monolayers with MACE-MPA-0 + D3 (descriptive; gates nothing). Runs in the MLIP environment. Resumable.

  R = Y2D_total(bilayer) / Y2D(monolayer)   (~2 if the layers act independently; per-layer ratio R/2 ~ 1)
Both are relaxed (ions + in-plane cell, c fixed) with the SAME calculator and then strained +/-1% with the validated protocol (src/mlip/stiffness.py).
Set: the 60 bilayers marked in results/mlip/bidb_validation_set.csv (best and worst reference stacking of each of the 30 monolayers).
Total and per-layer values are stored in separate columns; nothing is divided by the number of layers except where named per_layer.

  python -m src.mlip.bidb_stiffness_ratio [--shard 0/2]
"""
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
from .stiffness import Settings, elastic_constants, energy_stress_check, orient

OUT = "results/mlip"


def to_atoms(e):
    return orient(Atoms(numbers=e["numbers"], positions=e["positions"], cell=e["cell"], pbc=[True, True, False]))


def layer_shift(a: Atoms, n: int):
    """In-plane fractional offset between the centres of mass of the two layers (mod 1)."""
    f = a.get_scaled_positions(wrap=False)[:, :2]
    d = f[n:].mean(0) - f[:n].mean(0)
    return d - np.round(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    k, nsh = map(int, args.shard.split("/"))
    sel = pd.read_csv(f"{OUT}/bidb_validation_set.csv")
    sel = sel[sel.stiffness_pick.notna() & (sel.stiffness_pick != "")]
    monos = sorted(sel.monolayer_uid.unique())
    mine = set(m for i, m in enumerate(monos) if i % nsh == k)
    sel = sel[sel.monolayer_uid.isin(mine)]
    if args.limit:
        sel = sel.head(args.limit)
    S = {e["uid"]: e for e in json.load(open("../Dataset/bidb_structures.json"))}
    tag = "" if nsh == 1 else f"_shard{k}of{nsh}"
    path = f"{OUT}/raw_bidb_stiffness_mace_mpa0_medium{tag}.jsonl"
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["key"] for l in open(path) if l.strip()}
    calc = build("mace_mpa0_medium", dispersion=True)
    ver = versions()
    st = Settings(variant="relaxed_cell")
    first = to_atoms(S[sel.uid.iloc[0]])
    chk = energy_stress_check(first, calc)
    for kk, v in chk.items():
        if v["ratio"] is not None and not (0.97 <= v["ratio"] <= 1.03):
            raise SystemExit(f"energy/stress consistency FAILED (D3 on) {kk}: {v}")
    json.dump({"check": chk, "versions": ver}, open(f"{OUT}/consistency_bidb_stiffness.json", "w"), indent=2)
    print("consistency ok", {kk: round(v["ratio"], 4) for kk, v in chk.items() if v["ratio"]}, flush=True)
    jobs = [("mono", m) for m in sorted(sel.monolayer_uid.unique())] + [("bi", u) for u in sel.uid]
    todo = [(t, u) for t, u in jobs if f"{t}:{u}" not in done]
    print(f"{len(done)} done, {len(todo)} to do", flush=True)
    t_all = time.time()
    with open(path, "a") as f:
        for j, (typ, uid) in enumerate(todo):
            rec = {"key": f"{typ}:{uid}", "type": typ, "uid": uid, "model": "mace_mpa0_medium+D3", "versions": ver, "settings": dataclasses.asdict(st)}
            t0 = time.time()
            try:
                e = S[uid]
                a0 = to_atoms(e)
                if typ == "bi":
                    row = sel[sel.uid == uid].iloc[0]
                    n = len(a0) // 2
                    rec.update({"monolayer_uid": row.monolayer_uid, "formula": row.formula, "pick": row.stiffness_pick,
                                "ref_zscan": row.binding_energy_zscan, "ref_distance": row.distance})
                    sh0 = layer_shift(a0, n)
                info, a = elastic_constants(a0, calc, st, return_atoms=True)
                if typ == "bi":
                    sh1 = layer_shift(a, n)
                    cell = a.cell.array[:2, :2]
                    move = float(np.linalg.norm((sh1 - sh0) @ cell))
                    z = a.positions[:, 2]
                    info["layer_shift_change_A"] = move
                    info["stacking_preserved"] = bool(move < 0.3)
                    info["gap_after_relax"] = float(z[n:].min() - z[:n].max())
                rec["result"] = info
            except Exception as ex:
                rec["error"] = f"{type(ex).__name__}: {ex}"
                rec["trace"] = traceback.format_exc()[-500:]
            rec["wall_s"] = time.time() - t0
            f.write(json.dumps(rec, default=float) + "\n")
            f.flush()
            if (j + 1) % 5 == 0:
                print(f"  {j + 1}/{len(todo)}  {time.time() - t_all:.0f}s", flush=True)
    print(f"finished in {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
