"""Export the exact Task A folds (seed 42) and structure paths for the graph-network run.   python -m src.gnn.make_folds   (Task A environment)"""
import json

import numpy as np

from .. import data as D
from ..features import build_features
from ..splits import make_splits

cfg = D.load_config()
c2, _ = D.clean_c2db(D.load_c2db(cfg), cfg)
X, comp = build_features(c2, cfg)
y = c2.Y2D.to_numpy(float)
out = {"uid": c2.uid.tolist(), "path": c2.path.tolist(), "y": y.tolist(), "family": c2.family.tolist(), "chemsys": c2.chemsys.tolist(), "folds": {}}
for sc in ("random", "chemsys", "family"):
    splits = make_splits(sc, c2, X[comp], y, 5, cfg["cv"]["n_clusters"], 42)
    out["folds"][sc] = [{"train": tr.tolist(), "test": te.tolist()} for tr, te in splits]
json.dump(out, open(D.ROOT / "cache" / "phaseA_folds.json", "w"))
print("folds exported for", len(c2), "materials;", {s: [len(f["test"]) for f in v] for s, v in out["folds"].items()})
