import numpy as np
import pandas as pd

from src.panel import group_mean_pred, select_targets
from src.panel_analysis import icc1, mae
from src.panel_jarvis import select_targets as jarvis_select


def test_group_mean_pred_falls_back_to_global_mean_for_unseen_group():
    p = group_mean_pred(np.array([1.0, 3.0, 10.0]), np.array(["a", "a", "b"]), np.array(["a", "b", "zzz"]))
    assert p.tolist() == [2.0, 10.0, 14 / 3]


def test_icc1_is_high_for_group_determined_values_and_near_zero_for_noise():
    g = np.repeat(np.arange(50), 4)
    assert icc1(g.astype(float) + 0.01 * np.random.default_rng(0).normal(size=len(g)), g) > 0.95
    assert abs(icc1(np.random.default_rng(1).normal(size=len(g)), g)) < 0.2
    assert mae(np.array([1.0, 2.0]), np.array([2.0, 4.0])) == 1.5


def test_panel_selection_rules_drop_degenerate_small_and_identifier_targets():
    n = 3200
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"hform": rng.normal(size=n), "ehull": np.where(rng.random(n) < 0.7, 0.0, 1.0), "gap_hse": rng.normal(size=n),
                       "family": rng.integers(0, 50, n).astype(str), "chemsys": rng.integers(0, 80, n).astype(str)})
    df.loc[3000:, "gap_hse"] = np.nan                      # 3000 labels: still included (>= 3000)
    df.loc[2900:, "gap_hse"] = np.nan                      # 2900 labels: excluded
    inc, exc = select_targets(df.assign(**{c: np.nan for c in ("gap", "evac", "efermi", "vbm", "magmom", "alphax_el", "plasmafrequency_x", "emass_cbm", "Y2D", "poisson")}))
    assert [i["target"] for i in inc] == ["hform"]
    assert "degenerate" in exc["ehull"] and "fewer than" in exc["gap_hse"]


def test_jarvis_selection_keeps_one_representative_per_family():
    n = 600
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"formation_energy_peratom": rng.normal(size=n), "epsx": rng.random(n) + 1, "epsy": rng.random(n) + 1, "density": rng.random(n),
                       "family": "x", "chemsys": "y"})
    inc, exc = jarvis_select(df)
    names = [i["target"] for i in inc]
    assert "epsx" in names and "epsy" not in names and "density" not in names
