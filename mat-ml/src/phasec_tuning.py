"""Phase C2: nested, group-aware LightGBM tuning for Task A Y2D (design: results/phaseC_design.md).   python -m src.phasec_tuning   (~1.5-2.5 h, resumable per outer fold)

Outer folds = the Task A folds (seed 42, schemes random and family). Inner CV on each outer-training set uses the SAME grouping rule as the outer split
(KFold for random; GroupKFold by structure family for family) so that tuning never sees test-family information.
16 random configurations + the default configuration; chosen by inner MAE (original units); refit on the outer-training set; predict the outer-test set.
"""
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline

from . import data as D
from .features import build_features
from .splits import make_splits

cfg = D.load_config()
SEED = 42
RD = D.ROOT / cfg["paths"]["results_dir"]
OUT = D.ROOT / "cache" / "phasec_tuning.jsonl"
SPACE = {"num_leaves": [15, 31, 63, 127], "learning_rate": [0.02, 0.05, 0.1], "n_estimators": [300, 600, 1000], "min_child_samples": [5, 10, 20, 40],
         "subsample": [0.6, 0.8, 1.0], "colsample_bytree": [0.3, 0.5, 0.8], "reg_lambda": [0.0, 1.0, 10.0]}
DEFAULT = {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 500, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.5, "reg_lambda": 0.0}


def model(p):
    return make_pipeline(SimpleImputer(strategy="median"), lgb.LGBMRegressor(subsample_freq=1, random_state=SEED, verbose=-1, n_jobs=-1, **p))


def main():
    c2, _ = D.clean_c2db(D.load_c2db(cfg), cfg)
    X, comp = build_features(c2, cfg)
    y = c2.Y2D.to_numpy(float)
    ly = np.log(y)
    rng = np.random.default_rng(SEED)
    configs = [DEFAULT] + [{k: v[rng.integers(len(v))] for k, v in SPACE.items()} for _ in range(16)]
    done = set()
    if OUT.exists():
        done = {(r["scheme"], r["fold"]) for r in map(json.loads, open(OUT))}
    t0 = time.time()
    for scheme in ("random", "family"):
        splits = make_splits(scheme, c2, X[comp], y, 5, cfg["cv"]["n_clusters"], SEED)
        for f, (tr, te) in enumerate(splits):
            if (scheme, f) in done:
                continue
            inner = KFold(3, shuffle=True, random_state=SEED).split(tr) if scheme == "random" else GroupKFold(3).split(tr, groups=c2.family.to_numpy()[tr])
            inner = list(inner)
            scores = []
            for cfg_i in configs:
                errs = []
                for a, b in inner:
                    m = model(cfg_i).fit(X.iloc[tr[a]], ly[tr[a]])
                    errs.append(np.abs(np.exp(m.predict(X.iloc[tr[b]])) - y[tr[b]]).mean())
                scores.append(float(np.mean(errs)))
            best = int(np.argmin(scores))
            m = model(configs[best]).fit(X.iloc[tr], ly[tr])
            pred = np.exp(m.predict(X.iloc[te]))
            md = model(DEFAULT).fit(X.iloc[tr], ly[tr])
            pred_d = np.exp(md.predict(X.iloc[te]))
            rec = {"scheme": scheme, "fold": f, "best_config": configs[best], "inner_mae_best": scores[best], "inner_mae_default": scores[0], "chosen_is_default": best == 0,
                   "uid": c2.uid.to_numpy()[te].tolist(), "y": y[te].tolist(), "pred_tuned": pred.tolist(), "pred_default_refit": pred_d.tolist(),
                   "test_mae_tuned": float(np.abs(pred - y[te]).mean()), "test_mae_default": float(np.abs(pred_d - y[te]).mean())}
            open(OUT, "a").write(json.dumps(rec) + "\n")
            print(f"  {scheme:<7} fold {f}: tuned {rec['test_mae_tuned']:.2f}  default {rec['test_mae_default']:.2f}  (inner {scores[best]:.2f} vs {scores[0]:.2f}; chose {'default' if best == 0 else 'config ' + str(best)})  [{time.time() - t0:.0f}s]", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
