"""Paths and constants. Every number below is copied from a saved project result (file named in the comment), not invented."""
from __future__ import annotations

import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MODELS_DIR = Path(os.environ.get("MODELS_DIR", REPO / "mat-ml" / "models"))
REFERENCE_PATH = Path(os.environ.get("REFERENCE_PATH", MODELS_DIR / "taskA_reference.npz"))
FRONTEND_DIST = Path(os.environ.get("FRONTEND_DIST", REPO / "webapp" / "frontend" / "dist"))
RESULTS_DIR = REPO / "mat-ml" / "results"


def _json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


CARDS = _json(MODELS_DIR / "model_cards.json", {})

# Part B of results/mechanism_design.md: median distance to the nearest training row (standardised 141 features), seed 42.
_partB = _json(RESULTS_DIR / "mechanism_partB.json", {})
DIST_TYPICAL = float(_partB.get("median_distance_random", 2.444))      # median under a random split
DIST_UNUSUAL = float(_partB.get("median_distance_family", 3.215))      # median under a family-held-out split

# Cross-validated MAE (models/model_cards.json): random split ~ "known family", family split ~ "new family".
_y = CARDS.get("models", {}).get("taskA_Y2D", {}).get("cross_validated_MAE_N_per_m_5seeds", {})
_p = CARDS.get("models", {}).get("taskA_poisson", {}).get("cross_validated_seed42", {})
ERR_Y2D = {"known": float(_y.get("random", {}).get("mean", 13.8)), "new": float(_y.get("family", {}).get("mean", 19.9))}
ERR_POISSON = {"known": float(_p.get("random", {}).get("MAE", 0.111)), "new": float(_p.get("family", {}).get("MAE", 0.140))}
R2_POISSON_NEW = float(_p.get("family", {}).get("R2", 0.139))

# Distribution of Y2D in the clean C2DB training set (results/data_audit.json), N/m.
_audit = _json(RESULTS_DIR / "data_audit.json", {}).get("c2db", {}).get("Y2D_clean", {})
Y2D_QUANTILES = {"p25": float(_audit.get("25%", 24.08)), "median": float(_audit.get("50%", 47.93)), "p75": float(_audit.get("75%", 82.87)),
                 "min": float(_audit.get("min", 0.09)), "max": float(_audit.get("max", 713.58))}
