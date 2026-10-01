"""Composition (matminer Magpie) + simple structural features.

Inputs are composition and geometry only. DFT outputs (ehull, hform, dynamic stability, band gap)
are excluded unless features.use_dft_descriptors is true, because they are not available for a new
material and would blur what the model has actually learned.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pymatgen.core import Composition

from .data import cache_path

STRUCT_COLS = ["a", "b", "gamma", "area_per_atom", "nat", "z_extent", "n_layers", "spg_number"]


def magpie_table(formulas, cfg: dict) -> pd.DataFrame:
    """Magpie features indexed by reduced formula; only formulas missing from the cache are computed."""
    from matminer.featurizers.composition import ElementProperty

    cp = cache_path(cfg, "magpie.pkl")
    have = pd.read_pickle(cp) if cp.exists() else pd.DataFrame()
    todo = sorted(set(formulas) - set(have.index))
    if todo:
        f = ElementProperty.from_preset("magpie", impute_nan=True)
        labels = f.feature_labels()
        out = []
        for s in todo:
            try:
                out.append(f.featurize(Composition(s)))
            except Exception:
                out.append([np.nan] * len(labels))
        have = pd.concat([have, pd.DataFrame(out, index=todo, columns=labels)])
        have.to_pickle(cp)
    return have.loc[sorted(set(formulas))]


def build_features(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    """Return (X, composition_columns). Row order matches df."""
    mt = magpie_table(df.reduced_formula.unique(), cfg)
    comp = mt.loc[df.reduced_formula].reset_index(drop=True)
    comp.columns = [f"mp_{c}" for c in comp.columns]
    struct = pd.DataFrame({
        "a": df.a.values, "b": df.b.values, "gamma": df.gamma.values,
        "area_per_atom": (df.area / df.nat).values, "nat": df.nat.values,
        "z_extent": df.z_extent.values, "n_layers": df.n_layers.values,
        "spg_number": pd.to_numeric(df.get("spg_number", df.get("number")), errors="coerce").values,
    })
    struct["n_elements"] = df.n_elements.values
    X = pd.concat([comp, struct], axis=1)
    if cfg["features"].get("use_dft_descriptors"):
        for c in ("ehull", "hform"):
            if c in df:
                X[c] = df[c].values
    return X, list(comp.columns)


def prototype_features(df: pd.DataFrame, min_count: int = 10) -> pd.DataFrame:
    """Structure-prototype descriptors for C2DB: layer group and anonymous formula.

    lgnum              layer-group number (numeric)
    lg_<n>             one-hot of the layer group; groups with fewer than min_count rows -> lg_other
    anon_<A2B...>      one-hot of the anonymous formula; rare ones -> anon_other
    The category lists use row counts over the whole table (no labels), so no target information is used.
    Not available for JARVIS (it has no layer group), so these cannot enter the external check.
    """
    out = {"lgnum": pd.to_numeric(df["lgnum"], errors="coerce").to_numpy()}
    blocks = [pd.DataFrame(out, index=range(len(df)))]
    for col, prefix in (("lgnum", "lg"), ("anon_formula", "anon")):
        v = df[col].astype(str).reset_index(drop=True)
        keep = v.value_counts()
        keep = set(keep[keep >= min_count].index)
        v = v.where(v.isin(keep), "other")
        blocks.append(pd.get_dummies(v, prefix=prefix, dtype=float))
    return pd.concat(blocks, axis=1)
