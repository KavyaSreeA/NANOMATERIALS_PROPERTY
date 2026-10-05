"""Phase B: many C2DB properties x {random, chemical-system, family} CV with LightGBM and group-mean baselines; stores out-of-fold predictions.

  python -m src.panel                  # full panel (resumable; ~2 h)      python -m src.panel --quick   # smoke test
Design and hypotheses: results/panel_design.md (written before any panel model was fitted). Analysis: python -m src.panel_analysis.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.base import clone

from . import data as D
from .features import build_features
from .models import get_models
from .splits import make_splits

CANDIDATES = ["hform", "ehull", "gap", "gap_hse", "evac", "efermi", "vbm", "magmom", "alphax_el", "plasmafrequency_x", "emass_cbm", "Y2D", "poisson"]
EXCLUDED_BY_RULE = {"number": "identifier / symmetry integer", "lgnum": "identifier / symmetry integer", "thickness": "purely geometric",
                    "energy": "extensive total energy", "dipz": "degenerate (>50% at the modal value)", "minhessianeig": "degenerate (>50% at the modal value)"}
SCHEMES = ["random", "chemsys", "family"]
MIN_LABELS = 3000
CACHE = D.ROOT / "cache"


def load_c2db_all(cfg: dict, rebuild: bool = False) -> pd.DataFrame:
    """All C2DB materials: numeric properties + composition + geometry (+ Y2D / poisson where the Task A cleaning keeps them). Cached."""
    cp = CACHE / "c2db_all.pkl"
    if cp.exists() and not rebuild:
        return pd.read_pickle(cp)
    base = D.data_path(cfg, cfg["paths"]["c2db_dir"])
    rows = []
    for root, _d, files in os.walk(base):
        if "data.json" not in files:
            continue
        d = json.load(open(os.path.join(root, "data.json"), encoding="utf-8"))
        r = {k: float(v) for k, v in d.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and np.isfinite(v)}
        r.update({"uid": d["uid"], "path": os.path.relpath(root, base)})
        st = json.load(open(os.path.join(root, "structure.json"), encoding="utf-8"))["1"]
        numbers = D._nd(st["numbers"]).astype(int)
        lat = D._nd(st["cell"]).astype(float).reshape(3, 3)
        cart = D._nd(st["positions"]).astype(float).reshape(-1, 3)
        r.update(D.composition_info([D._symbol(z) for z in numbers]))
        r.update(D.slab_geometry(lat, cart @ np.linalg.inv(lat)))
        r["nat"] = len(numbers)
        r["n_layers"] = 1
        rows.append(r)
    df = pd.DataFrame(rows).sort_values("uid").reset_index(drop=True)
    df["family"] = df.anon_formula + "|lg" + df.lgnum.astype(int).astype(str)
    df["spg_number"] = df["number"]
    cc, _ = D.clean_c2db(D.load_c2db(cfg), cfg)
    cc = cc.set_index("uid")
    df["Y2D"] = df.uid.map(cc.Y2D)
    df["poisson"] = df.uid.map(cc.poisson.where(cc.poisson.abs() <= cfg["cleaning"]["poisson_abs_max"]))
    CACHE.mkdir(exist_ok=True)
    df.to_pickle(cp)
    return df


def select_targets(df: pd.DataFrame) -> tuple[list[dict], dict]:
    """Apply the pre-registered panel rules. Returns (included targets with metadata, excluded {name: reason})."""
    inc, exc = [], dict(EXCLUDED_BY_RULE)
    for t in CANDIDATES:
        s = df[t].dropna() if t in df else pd.Series(dtype=float)
        s = s[np.isfinite(s)]
        if len(s) < MIN_LABELS:
            exc[t] = f"fewer than {MIN_LABELS} labels ({len(s)})"
            continue
        mode = s.round(6).mode().iloc[0]
        if (np.abs(s - mode) < 1e-6).mean() > 0.5:
            exc[t] = "degenerate (>50% at the modal value)"
            continue
        log = bool((s > 0).all() and s.max() / s.min() > 100)
        inc.append({"target": t, "n": int(len(s)), "transform": "log" if log else "none", "n_families": int(df.loc[s.index, "family"].nunique()),
                    "n_chemsys": int(df.loc[s.index, "chemsys"].nunique())})
    return inc, exc


def group_mean_pred(y_tr, g_tr, g_te):
    m = pd.Series(y_tr).groupby(g_tr).mean()
    return pd.Series(g_te).map(m).fillna(y_tr.mean()).to_numpy()


def run_one(df, X, t, kind, scheme, seed, cfg, out, k=5):
    path = out / f"{t}__{scheme}__seed{seed}.npz"
    if path.exists():
        return
    ok = df[t].notna() & np.isfinite(df[t])
    d = df[ok].reset_index(drop=True)
    Xs = X[ok.to_numpy()].reset_index(drop=True)
    y = d[t].to_numpy(float)
    yt = np.log(y) if kind == "log" else y
    splits = make_splits(scheme, d, Xs, y, k, cfg["cv"]["n_clusters"], seed)
    model = get_models(cfg, seed, ["gradient_boosting"])["gradient_boosting"]
    P = {n: np.full(len(y), np.nan) for n in ("lgbm", "global", "chemsys", "family")}
    fold = np.full(len(y), -1)
    for f, (tr, te) in enumerate(splits):
        m = clone(model).fit(Xs.iloc[tr], yt[tr])
        p = m.predict(Xs.iloc[te])
        P["lgbm"][te] = np.exp(p) if kind == "log" else p
        P["global"][te] = y[tr].mean()
        P["chemsys"][te] = group_mean_pred(y[tr], d.chemsys.to_numpy()[tr], d.chemsys.to_numpy()[te])
        P["family"][te] = group_mean_pred(y[tr], d.family.to_numpy()[tr], d.family.to_numpy()[te])
        fold[te] = f
    np.savez_compressed(path, uid=d.uid.to_numpy(str), y=y, fold=fold, family=d.family.to_numpy(str), chemsys=d.chemsys.to_numpy(str), **{f"pred_{n}": v for n, v in P.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--targets", nargs="+")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    cfg = D.load_config()
    t0 = time.time()
    df = load_c2db_all(cfg, rebuild=args.rebuild)
    print(f"C2DB materials: {len(df)}  ({time.time() - t0:.0f}s)", flush=True)
    inc, exc = select_targets(df)
    json.dump({"included": inc, "excluded": exc, "rules": {"min_labels": MIN_LABELS, "degenerate_share": 0.5, "log_if_range_ratio_above": 100}},
              open(D.ROOT / cfg["paths"]["results_dir"] / "panel_targets.json", "w"), indent=2)
    print("panel:", [(i["target"], i["n"], i["transform"]) for i in inc], "\nexcluded:", exc, flush=True)
    X, _ = build_features(df, cfg)
    out = CACHE / ("panel_oof_quick" if args.quick else "panel_oof")
    out.mkdir(parents=True, exist_ok=True)
    cfgq = dict(cfg)
    if args.quick:
        cfgq["models"] = dict(cfg["models"], gbm_trees=60)
    seeds = args.seeds[:1] if args.quick else args.seeds
    for i in inc:
        if args.targets and i["target"] not in args.targets:
            continue
        for seed in seeds:
            for sc in SCHEMES:
                run_one(df, X, i["target"], i["transform"], sc, seed, cfgq, out)
            print(f"  {i['target']:<18} seed {seed} done  ({time.time() - t0:.0f}s)", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
