"""Cross-validation schemes.

random   KFold on rows. Comparison only: leaks chemistry (same chemical system in train and test).
cluster  K-means on standardised composition features, then StratifiedKFold on the cluster labels.
         Every cluster appears in train and test, as in Fronzi et al.: an in-domain split.
chemsys  GroupKFold-style by chemical system (e.g. 'Mo-S'): test systems never appear in training.
family   Same, grouped by structure family = anonymous formula + layer group (e.g. 'AB2|lg72').
monolayer (Task B) group by monolayer_uid so all stackings of one monolayer stay together.

Grouped schemes use StratifiedGroupKFold on 10 quantile bins of the target. This only balances the
target distribution across folds; groups stay disjoint.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

SCHEMES = ("random", "cluster", "chemsys", "family", "monolayer")


def cluster_labels(X_comp: pd.DataFrame, n_clusters: int, seed: int) -> np.ndarray:
    Z = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X_comp))
    return KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(Z)


def make_splits(scheme: str, df: pd.DataFrame, X_comp: pd.DataFrame, y: np.ndarray,
                n_splits: int, n_clusters: int, seed: int):
    idx = np.arange(len(df))
    if scheme == "random":
        return list(KFold(n_splits, shuffle=True, random_state=seed).split(idx))
    if scheme == "cluster":
        lab = cluster_labels(X_comp, n_clusters, seed)
        return list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(idx, lab))
    group_col = {"chemsys": "chemsys", "family": "family", "monolayer": "monolayer_uid"}.get(scheme)
    if group_col is None or group_col not in df:
        raise ValueError(f"scheme '{scheme}' not available for this dataset")
    bins = pd.qcut(pd.Series(y), 10, labels=False, duplicates="drop").to_numpy()
    sgkf = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
    return list(sgkf.split(idx, bins, df[group_col].to_numpy()))


def leakage_stats(splits, df: pd.DataFrame) -> dict:
    """Mean over folds of: fraction of test rows whose chemical system / reduced formula / family also
    occurs in the training rows. Grouped schemes drive their own column to exactly 0."""
    out = {"chemsys_in_train": [], "formula_in_train": [], "family_in_train": []}
    for tr, te in splits:
        for key, col in (("chemsys_in_train", "chemsys"), ("formula_in_train", "reduced_formula"),
                         ("family_in_train", "family")):
            if col in df:
                out[key].append(df.iloc[te][col].isin(set(df.iloc[tr][col])).mean())
    return {k: float(np.mean(v)) for k, v in out.items() if v}
