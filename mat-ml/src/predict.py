"""Use a saved model.

  python -m src.predict            # demo: Task A Y2D for MoS2 (an IN-SAMPLE material: the model was fitted on it)

  from src.predict import load, predict_table, featurize_task_a
  X = featurize_task_a(pd.DataFrame([{"formula": "MoS2", "a": 3.19, "b": 3.19, "gamma": 120.0, "area_per_atom": 2.93, "nat": 3, "thickness": 3.13, "spg_number": 187}]))
  y2d = predict_table("taskA_Y2D", X)        # N/m

Expect the cross-validated accuracy in models/model_cards.json for NEW materials (about 14 N/m MAE for known structure families, about 20 N/m for unseen families), not the in-sample fit.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from pymatgen.core import Composition

from . import data as D
from .features import magpie_table


def load(name: str) -> dict:
    return joblib.load(D.ROOT / "models" / f"{name}.joblib")


def predict_table(name: str, X: pd.DataFrame) -> np.ndarray:
    m = load(name)
    missing = [c for c in m["features"] if c not in X.columns]
    if missing:
        raise KeyError(f"missing feature columns: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    p = m["pipeline"].predict(X[m["features"]])
    return np.exp(p) if m["transform"] == "log" else p


def featurize_task_a(rows: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Build the 141 Task A features from: formula, a, b, gamma (A, A, deg), area_per_atom (A^2), nat, thickness (A), spg_number."""
    cfg = cfg or D.load_config()
    red = [Composition(f).reduced_formula for f in rows.formula]
    mt = magpie_table(sorted(set(red)), cfg)
    comp = mt.loc[red].reset_index(drop=True)
    comp.columns = [f"mp_{c}" for c in comp.columns]
    geo = pd.DataFrame({"a": rows.a.values, "b": rows.b.values, "gamma": rows.gamma.values, "area_per_atom": rows.area_per_atom.values, "nat": rows.nat.values,
                        "z_extent": rows.thickness.values, "n_layers": 1, "spg_number": rows.spg_number.values,
                        "n_elements": [len(Composition(f).elements) for f in rows.formula]}).astype(float)
    return pd.concat([comp, geo], axis=1)


if __name__ == "__main__":
    cfg = D.load_config()
    c2, _ = D.clean_c2db(D.load_c2db(cfg), cfg)
    r = c2[c2.uid == "1MoS2-1"].iloc[0]
    rows = pd.DataFrame([{"formula": r.reduced_formula, "a": r.a, "b": r.b, "gamma": r.gamma, "area_per_atom": r.area / r.nat, "nat": r.nat, "thickness": r.z_extent,
                          "spg_number": r.number}])
    print(f"{r.uid}: C2DB Y2D = {r.Y2D:.1f} N/m; model (in-sample) = {predict_table('taskA_Y2D', featurize_task_a(rows, cfg))[0]:.1f} N/m")
