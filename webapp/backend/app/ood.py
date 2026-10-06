"""Out-of-domain check: distance from the new material to its nearest training material in the standardised 141-feature space.

The reference (models/taskA_reference.npz) is built from the C2DB training features by scripts/build_reference.py; it is NOT committed by default
(it is derived from C2DB). Without it the API still works and reports that the check is unavailable.
Thresholds come from results/mechanism_partB.json: median nearest-neighbour distance when a random-split test row is predicted (typical) and when its whole family is held out (unusual).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


class Reference:
    def __init__(self, path=config.REFERENCE_PATH):
        z = np.load(path, allow_pickle=False)
        self.Z = z["Z"].astype(np.float32)
        self.median, self.mean, self.scale = z["median"], z["mean"], z["scale"]
        self.columns = [str(c) for c in z["columns"]]
        self.scale = np.where(self.scale == 0, 1.0, self.scale)

    def distance(self, X: pd.DataFrame) -> np.ndarray:
        A = X[self.columns].to_numpy(float)
        A = np.where(np.isnan(A), self.median, A)
        A = ((A - self.mean) / self.scale).astype(np.float32)
        d2 = (A ** 2).sum(1)[:, None] - 2 * A @ self.Z.T + (self.Z ** 2).sum(1)[None, :]
        return np.sqrt(np.clip(d2.min(axis=1), 0, None))


def load_reference():
    try:
        return Reference()
    except Exception:
        return None


def assess(distance: float | None) -> dict:
    if distance is None:
        return {"available": False, "level": "unknown",
                "message": "The check of how close your material is to the training data is not enabled on this server, so expect an error somewhere between the two values above."}
    if distance <= config.DIST_TYPICAL:
        level, msg = "typical", "Similar to materials the model was trained on; expect an error near the 'known family' value."
    elif distance < config.DIST_UNUSUAL:
        level, msg = "somewhat_unusual", "Farther from the training materials than most test cases; the error may be larger than the 'known family' value."
    else:
        level, msg = "unusual", "As far from the training materials as a held-out structure family; expect an error near the 'new family' value, or worse."
    return {"available": True, "level": level, "distance": float(distance), "typical_distance": config.DIST_TYPICAL, "unusual_distance": config.DIST_UNUSUAL, "message": msg}
