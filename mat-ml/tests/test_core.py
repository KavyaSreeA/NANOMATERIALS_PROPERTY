"""Fast tests (no large data). Run:  pytest -q   from mat-ml/   (pip install pytest)"""
import numpy as np
import pandas as pd
import pytest

from src import data as D
from src.splits import leakage_stats, make_splits
from src.taskb import STACK_RX, valid_mask


def test_tensor_stability_positive_definite():
    ok = D.tensor_stability(np.array([[100, 30, 0], [30, 100, 0], [0, 0, 40.0]]))
    bad = D.tensor_stability(np.array([[10, 50, 0], [50, 10, 0], [0, 0, 5.0]]))
    nan = D.tensor_stability(np.array([[np.nan, 1, 0], [1, 2, 0], [0, 0, 1.0]]))
    assert ok["stable_tensor"] and not bad["stable_tensor"] and not nan["stable_tensor"]
    assert ok["asym_frac"] == 0.0


def test_y2d_poisson_formula():
    df = pd.DataFrame({"c11": [134.1], "c12": [34.4], "c22": [134.1], "n_layers": [1]})
    out = D.add_elastic_targets(df)
    assert out.Y2D.iloc[0] == pytest.approx((134.1 * 134.1 - 34.4**2) / 134.1)       # 125.28 N/m (MoS2, JARVIS)
    assert out.poisson.iloc[0] == pytest.approx(34.4 / 134.1)
    assert out.Y2D_per_layer.iloc[0] == pytest.approx(out.Y2D.iloc[0])               # n_layers = 1


def test_slab_thickness_handles_periodic_wrap():
    lat = np.array([[3.0, 0, 0], [0, 3.0, 0], [0, 0, 20.0]])
    frac = np.array([[0, 0, 0.02], [0.5, 0.5, 0.98]])        # slab straddles the cell boundary
    g = D.slab_geometry(lat, frac)
    assert g["z_extent"] == pytest.approx(0.8) and g["a"] == pytest.approx(3.0) and g["gamma"] == pytest.approx(90.0)


@pytest.mark.parametrize("uid,m,iz,tx,ty", [
    ("C2-a6735a4a3797-2-1_0_0_1--0.67_-0.33", "1_0_0_1", None, "-0.67", "-0.33"),
    ("C2H2O2Ti3-b2d22251cb52-2--1_0_0_-1-Iz--0.67_-0.33", "-1_0_0_-1", "Iz", "-0.67", "-0.33"),
    ("Re4S8-627997a105e1-Re4S8-2--1_0_0_-1-0.06_0.76", "-1_0_0_-1", None, "0.06", "0.76"),     # doubled prefix seen in BiDB
    ("X-aaaaaaaaaaaa-2-1_-1_1_0-0_0", "1_-1_1_0", None, "0", "0"),
])
def test_stacking_uid_parser(uid, m, iz, tx, ty):
    r = STACK_RX.search(uid)
    assert r is not None and r.group("m") == m and r.group("iz") == iz and r.group("tx") == tx and r.group("ty") == ty


def test_validity_rule_upper_tail_only():
    s = pd.Series([-5.0, 0.0, 0.5, 10, 12, 15, 18, 20, 22, 25, 7970.0])
    ok, U = valid_mask(s, "upper")
    assert not ok.iloc[0] and not ok.iloc[1] and ok.iloc[2] and not ok.iloc[-1] and U < 7970       # weak positive binders kept, extreme value and non-positive removed
    assert valid_mask(s, "none")[0].sum() == 9


def _toy(n=300, n_groups=40, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, n_groups, n)
    df = pd.DataFrame({"chemsys": [f"c{i}" for i in g], "family": [f"f{i // 4}" for i in g], "monolayer_uid": [f"m{i}" for i in g],
                       "reduced_formula": [f"r{i}" for i in g]})
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
    return df, X, rng.lognormal(size=n)


@pytest.mark.parametrize("scheme,col", [("chemsys", "chemsys"), ("family", "family"), ("monolayer", "monolayer_uid")])
def test_grouped_splits_never_share_groups(scheme, col):
    df, X, y = _toy()
    splits = make_splits(scheme, df, X, y, 5, 5, 42)
    all_test = np.concatenate([te for _, te in splits])
    assert sorted(all_test) == list(range(len(df)))                                   # every row tested exactly once
    for tr, te in splits:
        assert set(df.iloc[tr][col]).isdisjoint(set(df.iloc[te][col]))
    key = {"chemsys": "chemsys_in_train", "family": "family_in_train", "monolayer_uid": None}[col]
    if key:
        assert leakage_stats(splits, df)[key] == 0.0


def test_random_split_partitions_rows_and_leaks_groups():
    df, X, y = _toy()
    splits = make_splits("random", df, X, y, 5, 5, 42)
    assert sorted(np.concatenate([te for _, te in splits])) == list(range(len(df)))
    assert leakage_stats(splits, df)["chemsys_in_train"] > 0.9                        # random splits do leak groups (the reason for the grouped schemes)


def test_log_iqr_filter():
    s = pd.Series([1.0, 2, 3, 4, 5, 6, 7, 8, 1e6, -1])
    keep = D.iqr_filter_log(s, 3.0)
    assert not keep.iloc[-1] and not keep.iloc[-2] and keep.iloc[:8].all()
