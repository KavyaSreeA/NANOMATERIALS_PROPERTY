import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from src.attribution import fold_importance, group_columns, hypotheses, summarise
from src.mechanism import assign_bin, boot_ratio, cluster_table, FAM_BINS


def _synthetic(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "mp_MagpieData mean Number": rng.normal(size=n), "mp_MagpieData mode Number": rng.normal(size=n),
        "mp_MagpieData mean Row": rng.normal(size=n), "mp_MagpieData range Row": rng.normal(size=n),
        "a": rng.normal(size=n), "spg_number": rng.integers(1, 80, n).astype(float), "n_layers": 1.0, "n_elements": rng.integers(1, 4, n).astype(float)})
    X["mp_MagpieData mode Number"] = X["mp_MagpieData mean Number"] * 0.8 + 0.2 * rng.normal(size=n)   # correlated member of the same group
    z = 2.0 * X["mp_MagpieData mean Number"] + 0.7 * X["a"] + 0.05 * rng.normal(size=n)
    return X, z.to_numpy()


def test_group_columns_groups_magpie_statistics_and_drops_constant_n_layers():
    X, _ = _synthetic(10)
    g = group_columns(X.columns)
    assert g["magpie:Number"] == ["mp_MagpieData mean Number", "mp_MagpieData mode Number"]
    assert g["magpie:Row"] and "n_layers" not in g and "a" in g and "spg_number" in g


def test_permutation_and_shap_rank_the_informative_group_first_and_noise_near_zero():
    import lightgbm as lgb
    X, z = _synthetic()
    model = make_pipeline(SimpleImputer(strategy="median"), lgb.LGBMRegressor(n_estimators=200, learning_rate=0.1, verbose=-1, random_state=0))
    tr, te = np.arange(1000), np.arange(1000, 1500)
    r = fold_importance(model, X.iloc[tr], z[tr], X.iloc[te], z[te], group_columns(X.columns), np.random.default_rng(0), repeats=3, inv=lambda v: v)
    r = r.set_index("group")
    assert r.perm_dMAE.idxmax() == "magpie:Number" and r.shap_abs.idxmax() == "magpie:Number"
    assert r.loc["magpie:Row", "perm_dMAE"] < 0.05 * r.loc["magpie:Number", "perm_dMAE"]
    assert r.loc["a", "perm_dMAE"] > r.loc["spg_number", "perm_dMAE"]


def test_summary_and_hypotheses_on_a_fake_fold_table():
    rows = []
    for sc in ("random", "chemsys", "family"):
        for f in range(3):
            for g, v in (("magpie:Number", 5.0), ("a", 2.0), ("spg_number", 0.01), ("area_per_atom", 1.0), ("nat", 0.5)):
                rows.append({"scheme": sc, "seed": 1, "fold": f, "group": g, "n_cols": 1, "perm_dMAE": v + 0.1 * f, "shap_abs": v / 10, "base_MAE": 1.0})
    s = summarise(pd.DataFrame(rows))
    assert abs(s[s.scheme == "family"].perm_share.sum() - 1) < 1e-9
    H = hypotheses(s)
    assert H["A1_rank_agreement_random_vs_family"]["pass"] and H["A3_spg_share"]["pass"] and not H["A2_max_group_share_family"]["pass"]


def test_dose_response_tools_recover_a_planted_effect():
    rng = np.random.default_rng(0)
    fam = np.repeat(np.arange(300), rng.integers(1, 30, 300))
    n = pd.Series(fam).map(pd.Series(fam).value_counts())
    df = pd.DataFrame({"uid": np.arange(len(fam)), "family": fam.astype(str), "n_fam": n.to_numpy()})
    df["e_random"] = 1.0 + 0.0 * n
    df["e_family"] = 1.0 + 0.02 * np.minimum(n, 25)          # penalty grows with family size, none for singletons beyond 0.02
    g = cluster_table(df, "family", "n_fam", FAM_BINS, "e_family")
    out, *_ = boot_ratio(g, "family", [b[2] for b in FAM_BINS], np.random.default_rng(1))
    assert out["1"]["Q"] < 1.05 and out["10-49"]["Q"] > 1.25 and out["10-49"]["lo"] < out["10-49"]["Q"] < out["10-49"]["hi"]
    assert assign_bin(pd.Series([1, 3, 12, 80]), FAM_BINS).tolist() == ["1", "2-3", "10-49", ">=50"]
