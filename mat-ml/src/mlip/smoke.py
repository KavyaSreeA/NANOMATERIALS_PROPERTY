"""Smoke test: build each calculator, run the energy/stress consistency check and one stiffness calculation on MoS2."""
import json
import sys
import time

import numpy as np
import torch
from ase import Atoms

from .calculators import MODELS, build
from .stiffness import Settings, elastic_constants, energy_stress_check, orient
from .structures import load_item


def load(item):
    nums = item["numbers"] if item.get("numbers") else None
    a = Atoms(numbers=nums, symbols=None if nums else item["symbols"], cell=item["cell"], positions=item["positions"], pbc=item["pbc"])
    return orient(a)


if __name__ == "__main__":
    items = json.load(open("results/mlip/benchmark_inputs.json"))
    pick = next((i for i in items if i["formula"] == "MoS2" and i["source"] == "jarvis"), items[0])
    atoms = load(pick)
    print("structure", pick["id"], pick["formula"], len(atoms), "atoms; ref", {k: round(v, 1) for k, v in pick["ref"].items() if v is not None})
    for name in (sys.argv[1:] or list(MODELS)):
        t = time.time()
        calc = build(name)
        print(f"\n== {name}: built in {time.time() - t:.1f}s")
        torch.cuda.reset_peak_memory_stats()
        chk = energy_stress_check(atoms, calc)
        print("energy/stress consistency", {k: {kk: round(vv, 5) if vv is not None else None for kk, vv in v.items()} for k, v in chk.items()})
        t = time.time()
        r = elastic_constants(atoms, calc, Settings(variant="relaxed_cell"))
        print(f"relaxed_cell {time.time() - t:.1f}s  peakGPU {torch.cuda.max_memory_allocated() / 1e6:.0f} MB")
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k in ("c11", "c12", "c22", "c66", "Y2D", "poisson", "asym_frac", "stable_tensor", "strain_relax_converged", "Lz", "a_b_gamma")})
        print("ref_relax", r["ref_relax"])
