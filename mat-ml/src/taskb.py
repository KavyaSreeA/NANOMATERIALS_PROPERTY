"""Task B: predict BiDB DFT interlayer labels from MONOLAYER information, under random / monolayer-grouped / family-grouped CV.

  python -m src.taskb                    # seed-42 full grid + seeds 42-46 LightGBM runs + sensitivity   (python -m src.train --task B does the same)
  python -m src.taskb --quick            # smoke test

Design is fixed in results/taskB_design.md (written before any model was fitted). Outputs are prefixed taskB_.
Targets (BiDB, PBE-D3, rigid C2DB layers): binding_energy_zscan [meV/A^2, per interface, positive = bound] and distance [A, vertical gap];
secondary binding_energy_gs (definition undocumented). All targets are fitted on the ln scale; metrics are reported in original units.
"""
from __future__ import annotations

import argparse
import json
import re
import time

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import data as D
from .features import magpie_table
from .models import get_models
from .splits import make_splits

STACK_RX = re.compile(r"-2-(?P<m>-?\d+_-?\d+_-?\d+_-?\d+)(?:-(?P<iz>Iz))?-(?P<tx>-?[\d.]+)_(?P<ty>-?[\d.]+)$")
TARGETS_PRIMARY = ["binding_energy_zscan", "distance"]
TARGET_SECONDARY = "binding_energy_gs"
SCHEMES = ["random", "monolayer", "family"]
FEATURE_SETS = ["mono_basic", "mono_stiffness", "mono_stiffness_stacking"]
PRIMARY_FS = "mono_stiffness"


# ----------------------------------------------------------------------------- data
def load_table(cfg: dict) -> pd.DataFrame:
    p = pd.read_csv(D.data_path(cfg, cfg["paths"]["bidb_properties"]), low_memory=False)
    S = {e["uid"]: e for e in json.load(open(D.data_path(cfg, cfg["paths"]["bidb_structures"]), encoding="utf-8"))}
    mono = p[p.number_of_layers == 1].set_index("monolayer_uid")
    bi = p[p.number_of_layers == 2].copy().reset_index(drop=True)
    # --- monolayer geometry from the stored monolayer structure
    geo = {}
    for muid in mono.index:
        e = S[mono.loc[muid, "uid"]]
        lat = np.array(e["cell"], float)
        frac = np.array(e["positions"], float) @ np.linalg.inv(lat)
        g = D.slab_geometry(lat, frac)
        geo[muid] = {"m_a": g["a"], "m_b": g["b"], "m_gamma": g["gamma"], "m_area_per_atom": g["area"] / len(e["numbers"]),
                     "m_nat": len(e["numbers"]), "m_thickness": g["z_extent"], "m_layer_group": mono.loc[muid, "layer_group_number"],
                     "m_stoich": str(mono.loc[muid, "stoichiometry"])}
    G = pd.DataFrame(geo).T
    for c in G.columns:
        if c != "m_stoich":
            G[c] = pd.to_numeric(G[c])
    bi = bi.join(G, on="monolayer_uid")
    bi["family"] = bi.m_stoich + "|lg" + bi.m_layer_group.astype(int).astype(str)
    # --- composition identifiers (bilayer formula reduces to the monolayer's reduced formula)
    comps = [Composition(f) for f in bi.formula]
    bi["reduced_formula"] = [c.reduced_formula for c in comps]
    bi["chemsys"] = ["-".join(sorted(e.symbol for e in c.elements)) for c in comps]
    bi["n_elements"] = [len(c.elements) for c in comps]
    # --- stacking descriptors decoded from the uid (extension feature set only)
    st = []
    for u in bi.uid:
        m = STACK_RX.search(u)
        if m is None:
            st.append([np.nan] * 8)
            continue
        a, b, c_, d = (int(v) for v in m.group("m").split("_"))
        tx, ty = float(m.group("tx")), float(m.group("ty"))
        st.append([a, b, c_, d, 1.0 if m.group("iz") else 0.0, np.sin(2 * np.pi * tx), np.cos(2 * np.pi * tx), np.sin(2 * np.pi * ty)])
        st[-1].append(np.cos(2 * np.pi * ty))
    S_ = pd.DataFrame(st, columns=["st_p1", "st_p2", "st_p3", "st_p4", "st_iz", "st_sin_tx", "st_cos_tx", "st_sin_ty", "st_cos_ty"])
    bi = pd.concat([bi, S_], axis=1)
    bi["b_layer_group"] = pd.to_numeric(bi.layer_group_number, errors="coerce")
    bi["b_inversion"] = bi.has_inversion_symmetry.astype(float)
    # --- C2DB stiffness through the uid map (Task A cleaning: tensor stable, asymmetry <= 10%)
    mp = pd.read_csv(D.data_path(cfg, cfg["paths"]["bidb_uid_map"]))
    cc, _ = D.clean_c2db(D.load_c2db(cfg), cfg)
    cc = cc.set_index("uid")
    mp["has_stiffness"] = mp.c2db_uid.isin(cc.index)
    sti = mp.set_index("monolayer_uid")
    rows = {}
    for muid, r in sti.iterrows():
        if r.has_stiffness:
            c = cc.loc[r.c2db_uid]
            rows[muid] = {"ln_Y2D": np.log(c.Y2D), "ln_C11": np.log(c.c11) if c.c11 > 0 else np.nan, "ln_C22": np.log(c.c22) if c.c22 > 0 else np.nan,
                          "C12": c.c12, "shear": c.shear_xy, "poisson": c.poisson, "has_stiffness": 1.0}
        else:
            rows[muid] = {"ln_Y2D": np.nan, "ln_C11": np.nan, "ln_C22": np.nan, "C12": np.nan, "shear": np.nan, "poisson": np.nan, "has_stiffness": 0.0}
    bi = bi.join(pd.DataFrame(rows).T, on="monolayer_uid")
    return bi


def feature_matrices(bi: pd.DataFrame, cfg: dict) -> dict:
    mt = magpie_table(bi.reduced_formula.unique(), cfg)
    comp = mt.loc[bi.reduced_formula].reset_index(drop=True)
    comp.columns = [f"mp_{c}" for c in comp.columns]
    geom = bi[["m_a", "m_b", "m_gamma", "m_area_per_atom", "m_nat", "m_thickness", "m_layer_group", "n_elements"]].reset_index(drop=True).astype(float)
    basic = pd.concat([comp, geom], axis=1)
    stiff_cols = ["ln_Y2D", "ln_C11", "ln_C22", "C12", "shear", "poisson", "has_stiffness"]
    stiff = bi[stiff_cols].reset_index(drop=True).astype(float)
    stack_cols = ["st_p1", "st_p2", "st_p3", "st_p4", "st_iz", "st_sin_tx", "st_cos_tx", "st_sin_ty", "st_cos_ty", "b_layer_group", "b_inversion"]
    stack = bi[stack_cols].reset_index(drop=True).astype(float)
    return {"mono_basic": basic, "mono_stiffness": pd.concat([basic, stiff], axis=1),
            "mono_stiffness_stacking": pd.concat([basic, stiff, stack], axis=1)}


def valid_mask(s: pd.Series, rule: str) -> tuple[pd.Series, float]:
    """rule 'upper': > 0 and <= U (U = Q3 + 3 IQR of log10 over positive values); 'none': finite and > 0; 'two_sided': also >= lower bound."""
    pos = s.notna() & (s > 0)
    ls = np.log10(s[pos])
    q1, q3 = ls.quantile(.25), ls.quantile(.75)
    hi, lo = 10 ** (q3 + 3 * (q3 - q1)), 10 ** (q1 - 3 * (q3 - q1))
    if rule == "none":
        return pos, np.inf
    if rule == "upper":
        return pos & (s <= hi), float(hi)
    if rule == "two_sided":
        return pos & (s <= hi) & (s >= lo), float(hi)
    raise ValueError(rule)


# ----------------------------------------------------------------------------- evaluation
def score(y, p, yl, pl) -> dict:
    rho = spearmanr(y, p)[0] if np.ptp(p) > 0 else np.nan
    return {"mae": mean_absolute_error(y, p), "rmse": mean_squared_error(y, p) ** 0.5, "r2": r2_score(y, p), "r2_log": r2_score(yl, pl),
            "spearman": rho}


def leakage(splits, df) -> dict:
    out = {}
    for key, col in (("monolayer_in_train", "monolayer_uid"), ("family_in_train", "family"), ("formula_in_train", "reduced_formula"), ("chemsys_in_train", "chemsys")):
        out[key] = float(np.mean([df.iloc[te][col].isin(set(df.iloc[tr][col])).mean() for tr, te in splits]))
    return out


def cross_validate(df, X_all, target, rule, schemes, fsets, model_names, cfg, seed, n_splits, quick, baselines=True, tag="main"):
    ok, U = valid_mask(df[target], rule)
    d = df[ok].reset_index(drop=True)
    y = d[target].to_numpy(float)
    yl = np.log(y)
    rows, leaks = [], {}
    models = get_models(cfg, seed, model_names, quick)
    for scheme in schemes:
        Xs = X_all[PRIMARY_FS][ok].reset_index(drop=True)
        splits = make_splits(scheme, d, Xs, y, n_splits, cfg["cv"]["n_clusters"], seed)
        leaks[scheme] = leakage(splits, d)
        for f, (tr, te) in enumerate(splits):
            base = {"target": target, "rule": rule, "scheme": scheme, "seed": seed, "fold": f, "n_train": len(tr), "n_test": len(te), "tag": tag}
            if baselines:
                am = y[tr].mean()
                gm = np.exp(yl[tr].mean())
                for name, val in (("mean_arith", am), ("mean_geom", gm)):
                    p = np.full(len(te), val)
                    rows.append({**base, "features": "none", "model": name, **score(y[te], p, yl[te], np.log(p))})
            for fs in fsets:
                X = X_all[fs][ok].reset_index(drop=True)
                for mname, model in models.items():
                    m = clone(model).fit(X.iloc[tr], yl[tr])
                    pl = m.predict(X.iloc[te])
                    rows.append({**base, "features": fs, "model": mname, **score(y[te], np.exp(pl), yl[te], pl)})
        print(f"  [{tag}] {target:<22} {scheme:<10} seed {seed} rule {rule} done", flush=True)
    # ceiling for monolayer-only information: share of ln-variance between monolayers
    g = pd.Series(yl).groupby(d.monolayer_uid.to_numpy())
    ceiling = 1 - ((yl - g.transform("mean").to_numpy()) ** 2).sum() / ((yl - yl.mean()) ** 2).sum()
    return rows, leaks, {"n_valid": int(ok.sum()), "n_removed": int((~ok).sum()), "upper_bound": U, "ceiling_between_monolayer_share_ln": float(ceiling),
                         "n_monolayers": int(d.monolayer_uid.nunique()), "n_families": int(d.family.nunique())}


# ----------------------------------------------------------------------------- reporting
def fold_spread(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["target", "rule", "scheme", "features", "model"], sort=False)
    agg = g.agg(n_folds=("fold", "count"), mae_mean=("mae", "mean"), mae_std=("mae", "std"), mae_min=("mae", "min"), mae_max=("mae", "max"),
                rmse_mean=("rmse", "mean"), r2_mean=("r2", "mean"), r2_std=("r2", "std"), r2_min=("r2", "min"), r2_max=("r2", "max"),
                r2_log_mean=("r2_log", "mean"), spearman_mean=("spearman", "mean"), spearman_std=("spearman", "std")).reset_index()
    base = df[df.model == "mean_arith"][["target", "rule", "scheme", "seed", "fold", "mae"]].rename(columns={"mae": "mae_base"})
    m = df.merge(base, on=["target", "rule", "scheme", "seed", "fold"])
    m["win"] = m.mae < m.mae_base
    w = m.groupby(["target", "rule", "scheme", "features", "model"]).agg(folds_better_than_mean=("win", "sum"), mean_mae_gain=("mae", lambda s: 0.0)).reset_index()
    gain = m.assign(d=m.mae_base - m.mae).groupby(["target", "rule", "scheme", "features", "model"]).d.mean().rename("mae_gain_vs_mean").reset_index()
    return agg.merge(w[["target", "rule", "scheme", "features", "model", "folds_better_than_mean"]], on=["target", "rule", "scheme", "features", "model"]).merge(
        gain, on=["target", "rule", "scheme", "features", "model"])


def seed_spread(df: pd.DataFrame) -> pd.DataFrame:
    per = df.groupby(["target", "rule", "scheme", "features", "model", "seed"]).agg(mae=("mae", "mean"), r2=("r2", "mean"), r2_log=("r2_log", "mean"),
                                                                                   spearman=("spearman", "mean")).reset_index()
    return per.groupby(["target", "rule", "scheme", "features", "model"]).agg(n_seeds=("seed", "nunique"), mae_mean=("mae", "mean"), mae_sd=("mae", "std"),
                                                                            r2_mean=("r2", "mean"), r2_sd=("r2", "std"), r2_log_mean=("r2_log", "mean"),
                                                                            spearman_mean=("spearman", "mean")).reset_index()


def md(df, floatfmt=3):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(f"{v:.{floatfmt}f}" if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-secondary", action="store_true")
    ap.add_argument("--no-sensitivity", action="store_true")
    ap.add_argument("--n-splits", type=int)
    ap.add_argument("--config")
    args = ap.parse_args(argv)
    cfg = D.load_config(args.config)
    t0 = time.time()
    k = args.n_splits or cfg["cv"]["n_splits"]
    bi = load_table(cfg)
    X_all = feature_matrices(bi, cfg)
    info = {"n_bilayers": len(bi), "n_monolayers": int(bi.monolayer_uid.nunique()), "n_families": int(bi.family.nunique()),
            "stacking_uid_parsed": int(bi.st_p1.notna().sum()), "bilayers_with_clean_c2db_stiffness": int((bi.has_stiffness == 1).sum()),
            "monolayers_with_clean_c2db_stiffness": int(bi.drop_duplicates("monolayer_uid").has_stiffness.sum()),
            "n_features": {k_: v.shape[1] for k_, v in X_all.items()}, "design": "results/taskB_design.md"}
    rows, leaks, per_target = [], {}, {}
    seed0 = args.seeds[0]
    # --- A: full grid, first seed
    targets = TARGETS_PRIMARY + ([] if args.no_secondary else [TARGET_SECONDARY])
    for t in targets:
        sec = t == TARGET_SECONDARY
        r, lk, pt = cross_validate(bi, X_all, t, "upper", SCHEMES, [PRIMARY_FS] if sec else FEATURE_SETS, ["ridge", "random_forest", "gradient_boosting"],
                                   cfg, seed0, k, args.quick, tag="main")
        rows += r; leaks[t] = lk; per_target[t] = pt
    # --- B: more seeds, LightGBM + baselines, primary targets, two feature sets
    if not args.quick:
        for sd in args.seeds[1:]:
            for t in TARGETS_PRIMARY:
                r, _, _ = cross_validate(bi, X_all, t, "upper", SCHEMES, ["mono_basic", PRIMARY_FS], ["gradient_boosting"], cfg, sd, k, False, tag="seeds")
                rows += r
    # --- C: sensitivity to the outlier rule (primary model, seed 0)
    if not args.quick and not args.no_sensitivity:
        for rule in ("none", "two_sided"):
            for t in TARGETS_PRIMARY:
                r, _, pt = cross_validate(bi, X_all, t, rule, SCHEMES, [PRIMARY_FS], ["gradient_boosting"], cfg, seed0, k, False, tag=f"sens_{rule}")
                rows += r
                per_target[f"{t}|{rule}"] = pt
    R = pd.DataFrame(rows)
    rd = D.ROOT / cfg["paths"]["results_dir"]
    R.to_csv(rd / "taskB_fold_metrics.csv", index=False)
    main_ = R[(R.tag == "main")]
    S = fold_spread(main_)
    S.to_csv(rd / "taskB_summary_folds.csv", index=False)
    seeds_df = R[R.tag.isin(["main", "seeds"]) & R.target.isin(TARGETS_PRIMARY) & (R.rule == "upper") & R.features.isin(["none", "mono_basic", PRIMARY_FS])]
    seeds_df = seeds_df[(seeds_df.model.isin(["gradient_boosting", "mean_arith", "mean_geom"]))]
    SS = seed_spread(seeds_df)
    SS.to_csv(rd / "taskB_summary_seeds.csv", index=False)
    sens = fold_spread(R[R.tag.str.startswith("sens")]) if (R.tag.str.startswith("sens")).any() else pd.DataFrame()
    if len(sens):
        sens.to_csv(rd / "taskB_summary_sensitivity.csv", index=False)
    info.update({"per_target": per_target, "leakage": leaks, "versions": {"pandas": pd.__version__, "numpy": np.__version__}, "runtime_s": round(time.time() - t0, 1),
                 "seeds": args.seeds, "quick": args.quick})
    (rd / "taskB_run_info.json").write_text(json.dumps(info, indent=2, default=float), encoding="utf-8")
    # --- comparison markdown
    lines = ["# Task B results (auto-generated)", "", f"Design: results/taskB_design.md. Seed {seed0} full grid, {k}-fold; MAE in original units (meV/A^2 for binding energy, A for distance).",
             "Per-fold spread = mean +/- std (min-max) over the folds.", ""]
    for t in targets:
        s = S[(S.target == t) & (S.rule == "upper")].copy()
        s["MAE"] = s.apply(lambda r: f"{r.mae_mean:.2f} ± {r.mae_std:.2f} ({r.mae_min:.2f}-{r.mae_max:.2f})", axis=1)
        s["R2"] = s.apply(lambda r: f"{r.r2_mean:.3f} ± {r.r2_std:.3f}", axis=1)
        s["folds_better"] = np.where(s.model == "mean_arith", "-", s.folds_better_than_mean.astype(int).astype(str) + f"/{k}")
        lines += [f"## {t}", "", f"valid {per_target[t]['n_valid']} (removed {per_target[t]['n_removed']}), monolayers {per_target[t]['n_monolayers']}, families {per_target[t]['n_families']}, "
                  f"ceiling (between-monolayer share of ln-variance) {per_target[t]['ceiling_between_monolayer_share_ln']:.3f}", "",
                  md(s[["scheme", "features", "model", "MAE", "R2", "r2_log_mean", "spearman_mean", "folds_better"]]), ""]
    lines += ["## Leakage (fraction of test bilayers whose key also occurs in training; seed %d)" % seed0, ""]
    lk = pd.DataFrame({f"{t}|{s_}": v for t in leaks for s_, v in leaks[t].items()}).T.reset_index().rename(columns={"index": "target|scheme"})
    lines += [md(lk), ""]
    if len(SS):
        lines += ["## Five-seed robustness (LightGBM and baselines; mean and std over seeds of the fold means)", "", md(SS), ""]
    if len(sens):
        lines += ["## Sensitivity to the outlier rule (LightGBM, mono_stiffness)", "", md(sens[["target", "rule", "scheme", "features", "model", "mae_mean", "mae_std", "r2_mean", "r2_std"]]), ""]
    (rd / "taskB_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:40])); print(f"\nWrote Task B results to {rd}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
