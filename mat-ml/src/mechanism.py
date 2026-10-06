"""Phase D1: where does the family penalty come from?  Design and hypotheses: results/mechanism_design.md (written before any number).

  python -m src.mechanism              # Part A: dose-response on the saved out-of-fold predictions (pandas + numpy + scipy only)
  python -m src.mechanism --distance   # Part B: nearest-training-row distance in feature space (needs the project data and cache)

Outputs: results/mechanism_partA.csv, mechanism_hypotheses.json, mechanism_report.md, figures/fig_mechanism.png (and mechanism_partB.json with --distance).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OOF = ROOT / "results" / "diagnostics" / "taskA_oof_predictions.csv"
OUT = ROOT / "results"
NB = 2000
FAM_BINS = [(1, 1, "1"), (2, 3, "2-3"), (4, 9, "4-9"), (10, 49, "10-49"), (50, 10**9, ">=50")]
CHEM_BINS = [(1, 1, "1"), (2, 3, "2-3"), (4, 9, "4-9"), (10, 10**9, ">=10")]


def load_errors(path=OOF) -> pd.DataFrame:
    """One row per material with absolute errors (N/m) under each scheme and the family / chemical-system sizes."""
    d = pd.read_csv(path)
    d = d[d.target == "Y2D"].copy()
    d["e"] = (d.pred - d.y).abs()
    w = d.pivot(index="uid", columns="scheme", values="e").rename(columns=lambda c: f"e_{c}")
    meta = d.drop_duplicates("uid").set_index("uid")[["family", "chemsys"]]
    out = meta.join(w).reset_index()
    out["n_fam"] = out.groupby("family").uid.transform("size")
    out["n_chem"] = out.groupby("chemsys").uid.transform("size")
    return out


def assign_bin(n: pd.Series, bins) -> pd.Series:
    lab = pd.Series(index=n.index, dtype=object)
    for lo, hi, name in bins:
        lab[(n >= lo) & (n <= hi)] = name
    return lab


def cluster_table(df: pd.DataFrame, cluster: str, size: str, bins, e_group: str) -> pd.DataFrame:
    """Per cluster: rows and the sums of the grouped and the random error, split by size bin."""
    df = df.assign(bin=assign_bin(df[size], bins))
    g = df.groupby([cluster, "bin"]).agg(rows=("uid", "size"), s_g=(e_group, "sum"), s_r=("e_random", "sum")).reset_index()
    return g


def boot_ratio(g: pd.DataFrame, cluster: str, bin_names, rng) -> dict:
    """Point estimate and cluster-bootstrap 95% CI of sum(s_g)/sum(s_r) per bin; also returns the bootstrap draws (for differences)."""
    clusters = g[cluster].unique()
    idx = {c: i for i, c in enumerate(clusters)}
    ci = g[cluster].map(idx).to_numpy()
    out, draws = {}, {}
    for b in bin_names:
        m = (g.bin == b).to_numpy()
        sg = np.bincount(ci[m], weights=g.s_g.to_numpy()[m], minlength=len(clusters))
        sr = np.bincount(ci[m], weights=g.s_r.to_numpy()[m], minlength=len(clusters))
        rows = np.bincount(ci[m], weights=g.rows.to_numpy()[m], minlength=len(clusters))
        out[b] = {"rows": int(rows.sum()), "clusters": int((rows > 0).sum()), "Q": float(sg.sum() / sr.sum()) if sr.sum() > 0 else np.nan}
        draws[b] = (sg, sr)
    W = rng.multinomial(len(clusters), np.ones(len(clusters)) / len(clusters), size=NB).astype(float)
    qs = {}
    for b in bin_names:
        sg, sr = draws[b]
        with np.errstate(invalid="ignore", divide="ignore"):
            qs[b] = (W @ sg) / (W @ sr)
        lo, hi = np.nanpercentile(qs[b], [2.5, 97.5]) if np.isfinite(qs[b]).any() else (np.nan, np.nan)
        out[b].update({"lo": float(lo), "hi": float(hi)})
    return out, qs, W, draws


def combined_ratio(draws, W, names):
    sg = sum(draws[n][0] for n in names)
    sr = sum(draws[n][1] for n in names)
    return float(sg.sum() / sr.sum()), (W @ sg) / (W @ sr)


def per_family_trend(df: pd.DataFrame, rng, min_n=3) -> dict:
    f = df.groupby("family").agg(n=("uid", "size"), sg=("e_family", "sum"), sr=("e_random", "sum")).reset_index()
    f = f[f.n >= min_n].copy()
    f["Q"] = f.sg / f.sr
    rho = spearmanr(np.log(f.n), f.Q)[0]
    boots = []
    for _ in range(NB):
        s = f.sample(len(f), replace=True, random_state=int(rng.integers(1 << 31)))
        boots.append(spearmanr(np.log(s.n), s.Q)[0])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {"families": int(len(f)), "rho": float(rho), "ci95": [float(lo), float(hi)]}


def part_a() -> dict:
    rng = np.random.default_rng(0)
    df = load_errors()
    res = {"rows": int(len(df)), "families": int(df.family.nunique()), "chemsys": int(df.chemsys.nunique()),
           "MAE": {s: float(df[f"e_{s}"].mean()) for s in ("random", "chemsys", "family")}}
    # family axis
    gf = cluster_table(df, "family", "n_fam", FAM_BINS, "e_family")
    names_f = [b[2] for b in FAM_BINS]
    fam, qf, Wf, df_draws = boot_ratio(gf, "family", names_f, rng)
    small, small_draw = combined_ratio(df_draws, Wf, ["1", "2-3"])
    large, large_draw = combined_ratio(df_draws, Wf, ["10-49", ">=50"])
    diff = large - small
    diff_draw = large_draw - small_draw
    diff_ci = [float(np.nanpercentile(diff_draw, 2.5)), float(np.nanpercentile(diff_draw, 97.5))]
    # chemistry axis
    gc = cluster_table(df, "chemsys", "n_chem", CHEM_BINS, "e_chemsys")
    names_c = [b[2] for b in CHEM_BINS]
    chem, _, _, _ = boot_ratio(gc, "chemsys", names_c, rng)
    # shares
    tot_pen = float((df.e_family - df.e_random).sum())
    big = df.n_fam >= 10
    shares = {"rows_share_n_fam>=10": float(big.mean()),
              "penalty_share_n_fam>=10": float((df.e_family - df.e_random)[big].sum() / tot_pen)}
    trend = per_family_trend(df, rng)
    s = fam["1"]
    H = {
        "M1_singletons": {"Q": s["Q"], "ci95": [s["lo"], s["hi"]], "rows": s["rows"],
                          "pass": bool(s["lo"] <= 1 <= s["hi"] and 0.90 <= s["Q"] <= 1.10)},
        "M2_dose_response": {"Q_small(n<=3)": small, "Q_large(n>=10)": large, "difference": diff, "ci95": diff_ci,
                             "pass": bool(diff >= 0.15 and diff_ci[0] > 0)},
        "M3_per_family_trend": {**trend, "pass": bool(trend["rho"] > 0.2 and trend["ci95"][0] > 0)},
        "M4_chemistry_control": {"bins": {b: chem[b] for b in names_c},
                                 "pass": bool(all(0.90 <= chem[b]["Q"] <= 1.15 for b in names_c if chem[b]["rows"] >= 100))},
        "M5_shares": shares,
    }
    res.update({"family_bins": fam, "chemsys_bins": chem, "hypotheses": H})
    return res


def write_part_a(res: dict) -> None:
    rows = []
    for axis, key in (("family", "family_bins"), ("chemsys", "chemsys_bins")):
        for b, v in res[key].items():
            rows.append({"axis": axis, "bin": b, **v})
    pd.DataFrame(rows).to_csv(OUT / "mechanism_partA.csv", index=False)
    (OUT / "mechanism_hypotheses.json").write_text(json.dumps(res, indent=1))
    H = res["hypotheses"]
    L = ["# D1 Part A: dose-response of the family penalty (auto-generated; design in `mechanism_design.md`)", "",
         f"Rows {res['rows']}; families {res['families']}; chemical systems {res['chemsys']}. Seed-42 out-of-fold MAE (N/m): "
         + ", ".join(f"{k} {v:.2f}" for k, v in res["MAE"].items()) + ".", "",
         "## Penalty ratio Q = sum(grouped error) / sum(random error), by size of the material's group",
         "", "| axis | group size | rows | groups | Q [95% CI] |", "|---|---|---|---|---|"]
    for axis, key in (("family", "family_bins"), ("chemical system", "chemsys_bins")):
        for b, v in res[key].items():
            L.append(f"| {axis} | {b} | {v['rows']} | {v['clusters']} | {v['Q']:.2f} [{v['lo']:.2f}, {v['hi']:.2f}] |")
    L += ["", "## Pre-specified tests", "", "| test | outcome | passed |", "|---|---|---|"]
    m1, m2, m3, m4, m5 = (H[k] for k in ("M1_singletons", "M2_dose_response", "M3_per_family_trend", "M4_chemistry_control", "M5_shares"))
    L.append(f"| M1 singletons Q in [0.90, 1.10], CI contains 1 | Q = {m1['Q']:.2f} [{m1['ci95'][0]:.2f}, {m1['ci95'][1]:.2f}] | {m1['pass']} |")
    L.append(f"| M2 Q(n>=10) - Q(n<=3) >= 0.15, CI above 0 | {m2['Q_large(n>=10)']:.2f} - {m2['Q_small(n<=3)']:.2f} = {m2['difference']:.2f} [{m2['ci95'][0]:.2f}, {m2['ci95'][1]:.2f}] | {m2['pass']} |")
    L.append(f"| M3 Spearman(log n, per-family Q) > 0.2, CI above 0 | rho = {m3['rho']:.2f} [{m3['ci95'][0]:.2f}, {m3['ci95'][1]:.2f}] ({m3['families']} families) | {m3['pass']} |")
    L.append(f"| M4 chemistry Q in [0.90, 1.15] in every bin with >= 100 rows | see table | {m4['pass']} |")
    L.append(f"| M5 (descriptive) | families with n >= 10 hold {m5['rows_share_n_fam>=10']:.0%} of rows and {m5['penalty_share_n_fam>=10']:.0%} of the total penalty | - |")
    (OUT / "mechanism_report.md").write_text("\n".join(L) + "\n")


def figure(res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    for ax, key, ttl, col in ((a, "family_bins", "(a) Family held out vs random, by family size", "#1baf7a"),
                              (b, "chemsys_bins", "(b) Chemical system held out vs random, by system size", "#eb6834")):
        v = res[key]
        xs = np.arange(len(v))
        q = [v[k]["Q"] for k in v]
        lo = [v[k]["Q"] - v[k]["lo"] for k in v]
        hi = [v[k]["hi"] - v[k]["Q"] for k in v]
        ax.errorbar(xs, q, yerr=[lo, hi], fmt="o", color=col, capsize=3, lw=1.3)
        ax.axhline(1, color="k", lw=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{k}\n(n={v[k]['rows']})" for k in v], fontsize=7)
        ax.set_xlabel("number of materials in the group", fontsize=8)
        ax.set_title(ttl, fontsize=8, loc="left")
        ax.grid(axis="x", visible=False)
    a.set_ylabel("MAE ratio, grouped / random (95% CI)", fontsize=8)
    fig.tight_layout()
    (OUT / "figures").mkdir(exist_ok=True)
    fig.savefig(OUT / "figures" / "fig_mechanism.png", dpi=200)


def part_b() -> dict:
    """Nearest-training-row distance (standardised 141 features), random vs family split, seed 42 (same folds as the diagnostics)."""
    from sklearn.impute import SimpleImputer
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    from . import data as D
    from .features import build_features
    from .splits import make_splits

    cfg = D.load_config()
    df, _ = D.clean_c2db(D.load_c2db(cfg), cfg)
    X, comp_cols = build_features(df, cfg)
    y = df["Y2D"].to_numpy(float)
    dist = {}
    for sc in ("random", "family"):
        splits = make_splits(sc, df, X[comp_cols], y, 5, cfg["cv"]["n_clusters"], 42)
        d = np.full(len(df), np.nan)
        for tr, te in splits:
            imp = SimpleImputer(strategy="median").fit(X.iloc[tr])
            sca = StandardScaler().fit(imp.transform(X.iloc[tr]))
            A, B = sca.transform(imp.transform(X.iloc[tr])), sca.transform(imp.transform(X.iloc[te]))
            nn = NearestNeighbors(n_neighbors=1).fit(A)
            d[te] = nn.kneighbors(B)[0][:, 0]
        dist[sc] = d
    e = load_errors().set_index("uid")
    tab = pd.DataFrame({"uid": df.uid, "family": df.family, "d_random": dist["random"], "d_family": dist["family"]}).set_index("uid").join(e[["e_random", "e_family"]])
    ratio = float(np.median(tab.d_family) / np.median(tab.d_random))
    f = tab.groupby("family").agg(n=("d_random", "size"), dr=("d_random", "median"), df_=("d_family", "median"), sr=("e_random", "sum"), sg=("e_family", "sum"))
    f = f[f.n >= 3]
    f["dd"] = f.df_ / f.dr
    f["Q"] = f.sg / f.sr
    rho = spearmanr(f.dd, f.Q)[0]
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(NB):
        s = f.sample(len(f), replace=True, random_state=int(rng.integers(1 << 31)))
        boots.append(spearmanr(s.dd, s.Q)[0])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    res = {"median_distance_random": float(np.median(tab.d_random)), "median_distance_family": float(np.median(tab.d_family)),
           "distance_ratio_family_over_random": ratio, "families": int(len(f)), "rho_distance_increase_vs_penalty": float(rho), "ci95": [float(lo), float(hi)],
           "M6_pass": bool(ratio >= 1.30 and rho > 0.3 and lo > 0)}
    (OUT / "mechanism_partB.json").write_text(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance", action="store_true", help="Part B (needs the project data and cache)")
    a = ap.parse_args()
    if a.distance:
        print(json.dumps(part_b(), indent=1))
        return
    res = part_a()
    write_part_a(res)
    figure(res)
    print(open(OUT / "mechanism_report.md").read())


if __name__ == "__main__":
    main()
