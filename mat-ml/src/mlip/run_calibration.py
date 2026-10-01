"""Monolayer calibration run: one model x one variant over the benchmark; append-only JSONL, resumable.

  python -m src.mlip.run_calibration --model mace_mp0_medium --variant relaxed_cell
Provenance per record: structure id/source, model name+tag, library versions, Settings, reference relaxation info,
strain size, per-structure runtime and peak GPU memory. The energy/stress self-check must pass (ratio 0.97-1.03) or the run aborts.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
import traceback

import torch

from .calculators import MODELS, build, versions
from .stiffness import Settings, elastic_constants, energy_stress_check
from .structures import load_item

OUT = "results/mlip"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--variant", default="relaxed_cell", choices=["relaxed_cell", "dft_cell"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--retry-errors", action="store_true", help="recompute ids whose previous record is an error (old error lines are kept in the file; the report uses the last record per id)")
    ap.add_argument("--ids", nargs="+")
    ap.add_argument("--fmax", type=float, default=0.005)
    ap.add_argument("--strain", type=float, default=0.01)
    args = ap.parse_args()

    items = json.load(open(f"{OUT}/benchmark_inputs.json"))
    if args.ids:
        items = [i for i in items if i["id"] in set(args.ids)]
    if args.limit:
        items = items[: args.limit]
    path = f"{OUT}/raw_{args.model}_{args.variant}.jsonl"
    done = set()
    if os.path.exists(path):
        recs = [json.loads(l) for l in open(path) if l.strip()]
        done = {r["id"] for r in recs if not (args.retry_errors and "error" in r)}
    todo = [i for i in items if i["id"] not in done]
    print(f"{args.model}/{args.variant}: {len(done)} done, {len(todo)} to do", flush=True)

    calc = build(args.model)
    ver = versions()
    # numerical self-check on the first structure (stress must equal -dE/d(strain)/V)
    chk = energy_stress_check(load_item(items[0]), calc)
    json.dump({"model": args.model, "check": chk, "versions": ver}, open(f"{OUT}/consistency_{args.model}.json", "w"), indent=2)
    for k, v in chk.items():
        if v["ratio"] is not None and not (0.97 <= v["ratio"] <= 1.03):
            raise SystemExit(f"energy/stress consistency FAILED for {args.model} {k}: {v}")
    print("consistency ok", {k: round(v["ratio"], 4) if v["ratio"] else None for k, v in chk.items()}, flush=True)

    s = Settings(strain=args.strain, fmax=args.fmax, variant=args.variant)
    t_all = time.time()
    with open(path, "a") as f:
        for n, it in enumerate(todo):
            torch.cuda.reset_peak_memory_stats()
            rec = {"id": it["id"], "source": it["source"], "formula": it["formula"], "family": it["family"], "nat": it["nat"],
                   "model": args.model, "model_tag": MODELS[args.model][1], "versions": ver, "settings": dataclasses.asdict(s),
                   "ref": it["ref"]}
            t0 = time.time()
            try:
                rec["result"] = elastic_constants(load_item(it), calc, s)
            except Exception as e:  # keep going; failures are part of the reliability picture
                rec["error"] = f"{type(e).__name__}: {e}"
                rec["trace"] = traceback.format_exc()[-600:]
            rec["wall_s"] = time.time() - t0
            rec["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / 1e6
            f.write(json.dumps(rec, default=float) + "\n")
            f.flush()
            if (n + 1) % 20 == 0:
                print(f"  {n + 1}/{len(todo)}  {time.time() - t_all:.0f}s", flush=True)
    print(f"finished in {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
