"""Phase D2: which inputs does the Task A model use?  Design and hypotheses: results/attribution_design.md (written before any number).

  python -m src.attribution                 # full run (seeds 42-44, three schemes); needs the project data and cache
  python -m src.attribution --quick         # one seed, one repeat (smoke test)

Method 1 permutation importance (group-wise, joint permutation within the held-out fold) and Method 2 TreeSHAP (LightGBM pred_contrib), both on the held-out rows of
5-fold cross-validation under random / chemical-system / family splits. Groups: the 22 Magpie element properties (six statistics each) and the individual
structural inputs. Outputs: results/attribution_fold.csv, attribution_summary.csv, attribution_hypotheses.json, attribution_report.md, figures/fig_attribution.png.
This describes how the model predicts; it does not support causal statements.
"""
from __future__ import annotations

import argparse
import json
import re
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone

STRUCT_GROUPS = ["a", "b", "gamma", "area_per_atom", "nat", "z_extent", "spg_number", "n_elements"]   # n_layers is constant and excluded
GEOMETRY = ["a", "b", "gamma", "area_per_atom", "nat", "z_extent"]
MAGPIE = re.compile(r"^mp_MagpieData (?:minimum|maximum|range|mean|avg_dev|mode) (.+)$")
SCHEMES = ["random", "chemsys", "family"]


def group_columns(columns) -> dict:
    """{group name: [columns]}; Magpie properties are grouped over their six statistics."""
    groups: dict[str, list[str]] = {}
    for c in columns:
        m = MAGPIE.match(c)
        if m:
            groups.setdefault(f"magpie:{m.group(1)}", []).append(c)
        elif c in STRUCT_GROUPS:
            groups[c] = [c]
    return groups


def fold_importance(model, Xtr, ztr, Xte, yte, groups, rng, repeats=5, inv=np.exp) -> pd.DataFrame:
    """Fit on the training rows; for each group return the permutation importance (MAE increase in original units) and the TreeSHAP importance (mean |contribution|, ln space)."""
    m = clone(model).fit(Xtr, ztr)
    base = float(np.abs(inv(m.predict(Xte)) - yte).mean())
    cols = list(Xte.columns)
    # TreeSHAP through the LightGBM booster inside the pipeline (imputer first)
    imp, booster = m[0], m[-1].booster_
    contrib = booster.predict(imp.transform(Xte), pred_contrib=True)[:, :-1]
    assert contrib.shape[1] == len(cols), "imputer dropped a column; SHAP columns no longer align"
    rows = []
    for g, gc in groups.items():
        idx = [cols.index(c) for c in gc]
        deltas = []
        for _ in range(repeats):
            perm = rng.permutation(len(Xte))
            Xp = Xte.copy()
            Xp[gc] = Xte[gc].to_numpy()[perm]            # one permutation shared by all columns of the group
            deltas.append(float(np.abs(inv(m.predict(Xp)) - yte).mean()) - base)
        rows.append({"group": g, "n_cols": len(gc), "perm_dMAE": float(np.mean(deltas)), "shap_abs": float(np.abs(contrib[:, idx]).sum(axis=1).mean()), "base_MAE": base})
    return pd.DataFrame(rows)


def shares(x: pd.Series) -> pd.Series:
    pos = x.clip(lower=0)
    return pos / pos.sum()


def summarise(fold: pd.DataFrame) -> pd.DataFrame:
    g = fold.groupby(["scheme", "group"]).agg(perm_dMAE=("perm_dMAE", "mean"), perm_sd=("perm_dMAE", "std"), shap_abs=("shap_abs", "mean"), n_cols=("n_cols", "first")).reset_index()
    g["perm_share"] = g.groupby("scheme").perm_dMAE.transform(lambda s: shares(s))
    g["shap_share"] = g.groupby("scheme").shap_abs.transform(lambda s: s / s.sum())
    return g


def hypotheses(summ: pd.DataFrame) -> dict:
    p = summ.pivot(index="group", columns="scheme", values="perm_dMAE")
    s = summ.pivot(index="group", columns="scheme", values="shap_abs")
    ps = summ.pivot(index="group", columns="scheme", values="perm_share")
    a1 = float(spearmanr(p["random"], p["family"])[0])
    top = ps["family"].idxmax()
    a2 = {"top_group": top, "share": float(ps["family"].max())}
    a3 = {sc: float(ps.loc["spg_number", sc]) for sc in SCHEMES}
    a4 = {sc: float(spearmanr(p[sc], s[sc])[0]) for sc in SCHEMES}
    geo = {sc: float(ps.loc[[g for g in GEOMETRY if g in ps.index], sc].sum()) for sc in SCHEMES}
    return {"A1_rank_agreement_random_vs_family": {"rho": a1, "pass": bool(a1 >= 0.70)},
            "A2_max_group_share_family": {**a2, "pass": bool(a2["share"] <= 0.40)},
            "A3_spg_share": {**a3, "pass": bool(all(v < 0.05 for v in a3.values()))},
            "A4_perm_vs_shap_rank": {**a4, "pass": bool(all(v >= 0.70 for v in a4.values()))},
            "A5_geometry_share_exploratory": {**geo, "family_gt_random": bool(geo["family"] > geo["random"])}}


def report(summ: pd.DataFrame, H: dict, path) -> None:
    L = ["# D2: feature-group attribution of the Task A LightGBM (auto-generated; design in `attribution_design.md`)", "",
         "Permutation importance = increase in held-out MAE (N/m) when a group is permuted jointly; share = importance / sum of positive importances. TreeSHAP share = mean |contribution| (ln space). Mean over 15 fold-fits per scheme.", ""]
    for sc in SCHEMES:
        t = summ[summ.scheme == sc].sort_values("perm_dMAE", ascending=False).head(12)
        L += [f"## {sc} split, top 12 groups", "", "| group | columns | permutation dMAE (N/m) | sd | perm. share | SHAP share |", "|---|---|---|---|---|---|"]
        for r in t.itertuples():
            L.append(f"| {r.group} | {r.n_cols} | {r.perm_dMAE:.2f} | {r.perm_sd:.2f} | {r.perm_share:.1%} | {r.shap_share:.1%} |")
        L.append("")
    L += ["## Pre-specified tests", "", "| test | outcome | passed |", "|---|---|---|"]
    a1, a2, a3, a4, a5 = (H[k] for k in ("A1_rank_agreement_random_vs_family", "A2_max_group_share_family", "A3_spg_share", "A4_perm_vs_shap_rank", "A5_geometry_share_exploratory"))
    L.append(f"| A1 Spearman(random, family importance) >= 0.70 | {a1['rho']:.2f} | {a1['pass']} |")
    L.append(f"| A2 top family-scheme group share <= 40% | {a2['top_group']}: {a2['share']:.1%} | {a2['pass']} |")
    L.append("| A3 spg_number share < 5% in every scheme | " + ", ".join(f"{k} {a3[k]:.1%}" for k in SCHEMES) + f" | {a3['pass']} |")
    L.append("| A4 permutation vs SHAP rank >= 0.70 | " + ", ".join(f"{k} {a4[k]:.2f}" for k in SCHEMES) + f" | {a4['pass']} |")
    L.append("| A5 (exploratory) geometry share | " + ", ".join(f"{k} {a5[k]:.1%}" for k in SCHEMES) + f"; family > random: {a5['family_gt_random']} | - |")
    path.write_text("\n".join(L) + "\n")


def figure(summ: pd.DataFrame, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = summ.pivot(index="group", columns="scheme", values="perm_share")
    top = p.sort_values("family", ascending=False).head(12)[::-1]
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    y = np.arange(len(top))
    ax.barh(y + 0.2, top["random"], 0.38, color="#2a78d6", label="random split")
    ax.barh(y - 0.2, top["family"], 0.38, color="#1baf7a", hatch="//", edgecolor="white", label="family split")
    ax.set_yticks(y)
    ax.set_yticklabels([g.replace("magpie:", "") for g in top.index], fontsize=7)
    ax.set_xlabel("share of permutation importance", fontsize=8)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def main():
    from . import data as D
    from .features import build_features
    from .models import get_models
    from .splits import make_splits
    from .train import fwd, inv

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    seeds, repeats = ([42], 1) if a.quick else (a.seeds, a.repeats)
    cfg = D.load_config()
    out = D.ROOT / cfg["paths"]["results_dir"]
    df, _ = D.clean_c2db(D.load_c2db(cfg), cfg)
    X, comp_cols = build_features(df, cfg)
    y = df["Y2D"].to_numpy(float)
    kind = cfg["targets"]["taskA"]["Y2D"]
    z = fwd(y, kind)
    groups = group_columns(X.columns)
    print(f"{len(groups)} groups over {sum(len(v) for v in groups.values())} columns", flush=True)
    recs, t0 = [], time.time()
    for scheme in SCHEMES:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            splits = make_splits(scheme, df, X[comp_cols], y, cfg["cv"]["n_splits"], cfg["cv"]["n_clusters"], seed)
            model = get_models(cfg, seed, ["gradient_boosting"], quick=a.quick)["gradient_boosting"]
            for f, (tr, te) in enumerate(splits):
                r = fold_importance(model, X.iloc[tr], z[tr], X.iloc[te], y[te], groups, rng, repeats, inv=lambda v: inv(v, kind))
                r.insert(0, "fold", f); r.insert(0, "seed", seed); r.insert(0, "scheme", scheme)
                recs.append(r)
            print(f"{scheme} seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    fold = pd.concat(recs, ignore_index=True)
    fold.to_csv(out / "attribution_fold.csv", index=False)
    summ = summarise(fold)
    summ.to_csv(out / "attribution_summary.csv", index=False)
    H = hypotheses(summ)
    (out / "attribution_hypotheses.json").write_text(json.dumps(H, indent=1))
    report(summ, H, out / "attribution_report.md")
    (out / "figures").mkdir(exist_ok=True)
    figure(summ, out / "figures" / "fig_attribution.png")
    print((out / "attribution_report.md").read_text())


if __name__ == "__main__":
    main()
