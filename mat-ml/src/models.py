"""Baseline regressors (scikit-learn; LightGBM for gradient boosting when installed)."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = ("ridge", "random_forest", "gradient_boosting")


def get_models(cfg: dict, seed: int, names=MODEL_NAMES, quick: bool = False) -> dict:
    rf_n = 60 if quick else cfg["models"]["rf_trees"]
    gb_n = 100 if quick else cfg["models"]["gbm_trees"]
    models = {}
    if "ridge" in names:
        models["ridge"] = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                        RidgeCV(alphas=np.logspace(-3, 3, 13)))
    if "random_forest" in names:
        models["random_forest"] = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(n_estimators=rf_n, max_features=0.33, n_jobs=-1, random_state=seed))
    if "gradient_boosting" in names:
        gbm = None
        if cfg["models"]["gbm"] == "lightgbm":
            try:
                from lightgbm import LGBMRegressor
                gbm = LGBMRegressor(n_estimators=gb_n, learning_rate=0.05, num_leaves=31, subsample=0.8,
                                    subsample_freq=1, colsample_bytree=0.5, random_state=seed, verbose=-1,
                                    n_jobs=-1)
            except ImportError:
                pass
        if gbm is None:
            gbm = GradientBoostingRegressor(n_estimators=min(gb_n, 300), learning_rate=0.05, max_depth=4,
                                            subsample=0.8, random_state=seed)
        models["gradient_boosting"] = make_pipeline(SimpleImputer(strategy="median"), gbm)
    return models
