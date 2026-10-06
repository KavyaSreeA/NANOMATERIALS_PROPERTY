"""Phase C2 for Task B: nested group-aware LightGBM tuning of the BiDB binding energy (binding_energy_zscan, rule 'upper', feature set mono_stiffness, seed 42).
Same protocol as src/phasec_tuning.py (design: results/phaseC_design.md): 12 random configs + default, inner 3-fold with the SAME grouping as the outer split.
python -m src.phasec_tuning_taskb   (resumable; output cache/phasec_tuning_taskb.jsonl)"""
import json
import time

import lightgbm as lgb
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline

from . import data as D
from .splits import make_splits
from .taskb import PRIMARY_FS, feature_matrices, load_table, valid_mask
from .phasec_tuning import DEFAULT, SPACE

cfg = D.load_config()
SEED, TARGET = 42, "binding_energy_zscan"
OUT = D.ROOT / "cache" / "phasec_tuning_taskb.jsonl"
GROUP = {"random": None, "monolayer": "monolayer_uid", "family": "family"}


def model(p):
    return make_pipeline(SimpleImputer(strategy="median"), lgb.LGBMRegressor(subsample_freq=1, random_state=SEED, verbose=-1, n_jobs=6, **p))


def main():
    bi = load_table(cfg)
    X_all = feature_matrices(bi, cfg)
    ok, _ = valid_mask(bi[TARGET], "upper")
    d = bi[ok].reset_index(drop=True)
    X = X_all[PRIMARY_FS][ok].reset_index(drop=True)
    y = d[TARGET].to_numpy(float); ly = np.log(y)
    rng = np.random.default_rng(SEED)
    configs = [DEFAULT] + [{k: v[rng.integers(len(v))] for k, v in SPACE.items()} for _ in range(12)]
    done = set()
    if OUT.exists():
        done = {(r["scheme"], r["fold"]) for r in map(json.loads, open(OUT))}
    t0 = time.time()
    for scheme, gcol in GROUP.items():
        splits = make_splits(scheme, d, X, y, 5, cfg["cv"]["n_clusters"], SEED)
        for f, (tr, te) in enumerate(splits):
            if (scheme, f) in done:
                continue
            inner = list(KFold(3, shuffle=True, random_state=SEED).split(tr)) if gcol is None else list(GroupKFold(3).split(tr, groups=d[gcol].to_numpy()[tr]))
            scores = []
            for c in configs:
                e = []
                for a, b in inner:
                    m = model(c).fit(X.iloc[tr[a]], ly[tr[a]])
                    e.append(np.abs(np.exp(m.predict(X.iloc[tr[b]])) - y[tr[b]]).mean())
                scores.append(float(np.mean(e)))
            best = int(np.argmin(scores))
            pt = np.exp(model(configs[best]).fit(X.iloc[tr], ly[tr]).predict(X.iloc[te]))
            pd_ = np.exp(model(DEFAULT).fit(X.iloc[tr], ly[tr]).predict(X.iloc[te]))
            rec = {"scheme": scheme, "fold": f, "best_config": configs[best], "chosen_is_default": best == 0, "uid": d.bilayer_uid.to_numpy()[te].tolist() if "bilayer_uid" in d else te.tolist(),
                   "monolayer": d.monolayer_uid.to_numpy()[te].tolist(), "family": d.family.to_numpy()[te].tolist(), "y": y[te].tolist(),
                   "pred_tuned": pt.tolist(), "pred_default_refit": pd_.tolist(), "test_mae_tuned": float(np.abs(pt - y[te]).mean()), "test_mae_default": float(np.abs(pd_ - y[te]).mean())}
            open(OUT, "a").write(json.dumps(rec) + "\n")
            print(f"  {scheme:<9} fold {f}: tuned {rec['test_mae_tuned']:.2f}  default {rec['test_mae_default']:.2f}  ({'default' if best == 0 else 'config ' + str(best)})  [{time.time() - t0:.0f}s]", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
