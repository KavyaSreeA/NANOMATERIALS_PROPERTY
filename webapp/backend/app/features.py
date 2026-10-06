"""Build the 141 Task A features from a formula and a few geometry numbers, exactly as mat-ml/src/features.py does.

132 Magpie composition statistics (computed on the REDUCED formula) + a, b, gamma, area_per_atom, nat, z_extent (thickness),
n_layers (= 1), spg_number, n_elements.  The column order comes from the saved model, never from this file.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import pandas as pd
from pymatgen.core import Composition

_EP = None


def _featurizer():
    global _EP
    if _EP is None:
        from matminer.featurizers.composition import ElementProperty
        _EP = ElementProperty.from_preset("magpie", impute_nan=True)
    return _EP


@lru_cache(maxsize=4096)
def magpie_row(reduced_formula: str) -> tuple:
    ep = _featurizer()
    try:
        return tuple(float(v) for v in ep.featurize(Composition(reduced_formula)))
    except Exception as exc:                       # unknown element etc.
        raise ValueError(f"cannot featurize composition '{reduced_formula}': {exc}") from exc


def magpie_labels() -> list[str]:
    return [f"mp_{c}" for c in _featurizer().feature_labels()]


def parse_formula(formula: str) -> Composition:
    try:
        comp = Composition(formula)
    except Exception as exc:
        raise ValueError(f"cannot read formula '{formula}'") from exc
    if len(comp.elements) == 0 or any(getattr(e, "Z", 0) < 1 for e in comp.elements):
        raise ValueError(f"'{formula}' contains an unknown element or is not a valid chemical formula")
    return comp


def area_per_atom(a: float, b: float, gamma_deg: float, nat: int) -> float:
    return a * b * math.sin(math.radians(gamma_deg)) / nat


def featurize(rows: list[dict], columns: list[str]) -> pd.DataFrame:
    """rows: dicts with formula, a, b, gamma, nat, thickness, spg_number (and optionally area_per_atom). Returns a DataFrame with `columns` in the model's order."""
    labels = magpie_labels()
    out = []
    for r in rows:
        comp = parse_formula(r["formula"])
        red = comp.reduced_formula
        feat = dict(zip(labels, magpie_row(red)))
        apa = r.get("area_per_atom")
        if apa is None:
            apa = area_per_atom(r["a"], r["b"], r["gamma"], r["nat"])
        feat.update({"a": r["a"], "b": r["b"], "gamma": r["gamma"], "area_per_atom": apa, "nat": r["nat"], "z_extent": r["thickness"],
                     "n_layers": 1.0, "spg_number": r["spg_number"], "n_elements": float(len(Composition(red).elements))})
        out.append(feat)
    X = pd.DataFrame(out)
    missing = [c for c in columns if c not in X.columns]
    if missing:
        raise KeyError(f"feature columns missing: {missing[:5]}")
    return X[columns].astype(float)
