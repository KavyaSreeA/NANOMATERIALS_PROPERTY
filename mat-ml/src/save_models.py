"""Train and save the final primary models on ALL valid data, with feature lists and a model card.

  python -m src.save_models

These are the models behind the headline results (LightGBM; same settings as the cross-validated runs, seed 42). The saved files are for reuse and inspection.
They are NOT an estimate of generalisation: their in-sample error (recorded in the model card) is optimistic by construction. The honest accuracy figures are the
grouped cross-validation results copied into the card. Each file is a joblib dict {pipeline, features, target, transform, seed}; predictions are exp(pipeline.predict) for log targets.
"""
from __future__ import annotations

import json
import platform

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error, r2_score

from . import data as D
from . import taskb as T
from .features import build_features
from .models import get_models

SEED = 42
OUT = D.ROOT / "models"


def fit_one(name, X, y, kind, cfg):
    model = get_models(cfg, SEED, ["gradient_boosting"])["gradient_boosting"]
    yt = np.log(y) if kind == "log" else y
    model.fit(X, yt)
    pred = model.predict(X)
    pred = np.exp(pred) if kind == "log" else pred
    obj = {"pipeline": model, "features": list(X.columns), "target": name, "transform": kind, "seed": SEED}
    path = OUT / f"{name}.joblib"
    joblib.dump(obj, path, compress=3)
    # round trip: the file must reproduce the in-memory predictions exactly
    back = joblib.load(path)
    p2 = back["pipeline"].predict(X[back["features"]])
    p2 = np.exp(p2) if kind == "log" else p2
    assert np.allclose(pred, p2), f"round-trip mismatch for {name}"
    return {"file": f"models/{name}.joblib", "size_kb": round(path.stat().st_size / 1024), "n_train": int(len(y)), "n_features": X.shape[1],
            "target_transform": kind, "in_sample_MAE_optimistic": float(mean_absolute_error(y, pred)), "in_sample_R2_optimistic": float(r2_score(y, pred)),
            "round_trip_ok": True}


def main():
    OUT.mkdir(exist_ok=True)
    cfg = D.load_config()
    rd = D.ROOT / cfg["paths"]["results_dir"]
    cards = {}
    # ---- Task A
    c2, _ = D.clean_c2db(D.load_c2db(cfg), cfg)
    for target, kind, unit in (("Y2D", "log", "N/m"), ("poisson", "none", "")):
        df = D.restrict_for_target(c2, target, cfg)
        X, _ = build_features(df, cfg)
        card = fit_one(f"taskA_{target}", X, df[target].to_numpy(float), kind, cfg)
        card.update({"task": "A: monolayer stiffness", "unit": unit, "training_data": "C2DB, mechanically stable monolayers (tensor positive definite, asymmetry <= 10%)",
                     "feature_set": "Magpie (132) + n_elements + a, b, gamma, area_per_atom, nat, z_extent, n_layers, spg_number"})
        cards[f"taskA_{target}"] = card
    rob = pd.read_csv(rd / "robustness_summary.csv")
    g = rob[(rob.model == "gradient_boosting") & (rob.features == "all")].set_index("scheme")
    cards["taskA_Y2D"]["cross_validated_MAE_N_per_m_5seeds"] = {s: {"mean": float(g.loc[s, "mae_mean"]), "seed_sd": float(g.loc[s, "mae_sd"]), "ln_R2": float(g.loc[s, "r2_log_mean"])}
                                                                 for s in ("random", "chemsys", "family")}
    ta = pd.read_csv(rd / "task_A_summary.csv")
    gp = ta[(ta.target == "poisson") & (ta.model == "gradient_boosting")].set_index("scheme")
    cards["taskA_poisson"]["cross_validated_seed42"] = {s: {"MAE": float(gp.loc[s, "mae_mean"]), "R2": float(gp.loc[s, "r2_mean"])} for s in ("random", "chemsys", "family")}
    # ---- Task B
    bi = T.load_table(cfg)
    X_all = T.feature_matrices(bi, cfg)
    ts = pd.read_csv(rd / "taskB_summary_seeds.csv")
    for target, unit in (("binding_energy_zscan", "meV/A^2"), ("distance", "A")):
        ok, U = T.valid_mask(bi[target], "upper")
        X = X_all[T.PRIMARY_FS][ok].reset_index(drop=True)
        y = bi.loc[ok, target].to_numpy(float)
        card = fit_one(f"taskB_{target}", X, y, "log", cfg)
        g = ts[(ts.target == target) & (ts.model == "gradient_boosting") & (ts.features == T.PRIMARY_FS)].set_index("scheme")
        card.update({"task": "B: BiDB interlayer label from monolayer information", "unit": unit, "valid_range": f"> 0 and <= {U:.4g}",
                     "training_data": "BiDB homobilayers (PBE-D3, rigid C2DB layers); all stackings of each monolayer",
                     "feature_set": "mono_stiffness: Magpie + monolayer geometry + layer group + C2DB stiffness via uid map (147). Identical for all stackings of a monolayer.",
                     "cross_validated_MAE_5seeds": {s: {"mean": float(g.loc[s, "mae_mean"]), "seed_sd": float(g.loc[s, "mae_sd"]), "ln_R2": float(g.loc[s, "r2_log_mean"])}
                                                    for s in ("random", "monolayer", "family")}})
        cards[f"taskB_{target}"] = card
    meta = {"hyperparameters": {"model": "LightGBM (via sklearn pipeline: median imputer + LGBMRegressor)", "n_estimators": cfg["models"]["gbm_trees"], "learning_rate": 0.05,
                                "num_leaves": 31, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.5, "random_state": SEED},
            "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "lightgbm": lightgbm.__version__, "numpy": np.__version__, "pandas": pd.__version__},
            "how_to_read": ("Use the cross-validated numbers for expected accuracy on NEW data: random splits are optimistic; for new structure families (Task A) or new monolayers/families "
                            "(Task B) use the 'family' / 'monolayer' entries. In-sample numbers only show that the file loads and fits."),
            "domain": "Task A: mechanically stable, non-filtered C2DB-like monolayers. Task B: BiDB homobilayers of C2DB monolayers; magnetic layers were not excluded.",
            "models": cards}
    (OUT / "model_cards.json").write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")
    for k, v in cards.items():
        (OUT / f"{k}_features.json").write_text(json.dumps(joblib.load(OUT / f"{k}.joblib")["features"], indent=1), encoding="utf-8")
        print(k, {kk: v[kk] for kk in ("size_kb", "n_train", "n_features", "in_sample_MAE_optimistic", "round_trip_ok")})


if __name__ == "__main__":
    main()
