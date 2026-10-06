"""Build models/taskA_reference.npz for the out-of-domain check.   Run from mat-ml/ in the project environment (needs Dataset/ and the cache):

    cd mat-ml
    python ../webapp/scripts/build_reference.py

It stores the standardised 141 training features (float32, about 4 MB), the imputer medians and the scaler mean/scale.
This file is DERIVED FROM C2DB: it is not tracked by git by default. Check C2DB's licence terms before publishing it; to deploy, copy it next to the models.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
from sklearn.impute import SimpleImputer          # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src import data as D                         # noqa: E402
from src.features import build_features           # noqa: E402

cfg = D.load_config()
df, _ = D.clean_c2db(D.load_c2db(cfg), cfg)
X, _ = build_features(df, cfg)
model_cols = __import__("joblib").load(D.ROOT / "models" / "taskA_Y2D.joblib")["features"]
X = X[model_cols]
imp = SimpleImputer(strategy="median").fit(X)
sca = StandardScaler().fit(imp.transform(X))
Z = sca.transform(imp.transform(X)).astype("float32")
out = D.ROOT / "models" / "taskA_reference.npz"
np.savez_compressed(out, Z=Z, median=imp.statistics_, mean=sca.mean_, scale=sca.scale_, columns=np.array(model_cols))
print(f"wrote {out}  rows={len(Z)}  columns={len(model_cols)}  size={out.stat().st_size/1e6:.1f} MB")
