"""FastAPI backend: predicts monolayer stiffness (Y2D) and Poisson ratio from a formula + geometry or a CIF/POSCAR, with honest error bands.

  uvicorn app.main:app --reload --port 8000     (from webapp/backend)

Endpoints: GET /api/health, GET /api/models, POST /api/predict, POST /api/predict/structure, POST /api/predict/batch.
The model is a screening aid trained on C2DB monolayers; it is not a replacement for DFT and was not validated for bilayers.
"""
from __future__ import annotations

import csv
import io
import math

import joblib
import numpy as np
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, ood
from .features import area_per_atom, featurize, parse_formula
from .structure import describe, read_structure

MAX_UPLOAD = 2_000_000
MAX_BATCH = 500


class State:
    y = p = None
    columns: list[str] = []
    reference = None


S = State()


def load_models():
    S.y = joblib.load(config.MODELS_DIR / "taskA_Y2D.joblib")
    S.p = joblib.load(config.MODELS_DIR / "taskA_poisson.joblib")
    if S.y["features"] != S.p["features"]:
        raise RuntimeError("Y2D and Poisson models use different feature lists")
    S.columns = list(S.y["features"])
    S.reference = ood.load_reference()


@asynccontextmanager
async def lifespan(_app):
    load_models()
    yield


app = FastAPI(title="2D-material stiffness predictor", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    formula: str = Field(..., min_length=1, max_length=60, examples=["MoS2"])
    nat: int = Field(..., ge=1, le=200, description="atoms in the (primitive) cell")
    a: float = Field(..., gt=1.0, lt=30.0, description="in-plane lattice length a (Å)")
    b: float = Field(..., gt=1.0, lt=30.0, description="in-plane lattice length b (Å)")
    gamma: float = Field(..., gt=30.0, lt=150.0, description="angle between a and b (degrees)")
    thickness: float = Field(..., ge=0.0, le=20.0, description="slab thickness along the surface normal (Å); 0 for a single atomic plane")
    spg_number: int = Field(..., ge=1, le=230, description="space-group number of the slab")


def _band(v: float) -> dict:
    q = config.Y2D_QUANTILES
    lo, hi = math.log(q["min"]), math.log(q["max"])
    pos = lambda x: float(min(1.0, max(0.0, (math.log(max(x, q["min"])) - lo) / (hi - lo))))
    if v < q["p25"]:
        name = "bottom quarter of C2DB monolayers"
    elif v < q["median"]:
        name = "second quarter (below the median)"
    elif v < q["p75"]:
        name = "third quarter (above the median)"
    else:
        name = "top quarter of C2DB monolayers"
    return {"label": name, "position": pos(v), "quartiles": {"p25": pos(q["p25"]), "median": pos(q["median"]), "p75": pos(q["p75"])},
            "values": {"p25": q["p25"], "median": q["median"], "p75": q["p75"]}}


def _expected(level: str, known: float, new: float):
    if level == "typical":
        return known, "known-family error"
    if level == "unusual":
        return new, "new-family error"
    if level == "somewhat_unusual":
        return (known + new) / 2, "between the known- and new-family errors"
    return None, "unknown (distance check unavailable)"


def run(rows: list[dict]) -> list[dict]:
    if S.y is None:
        load_models()
    X = featurize(rows, S.columns)
    y = np.exp(S.y["pipeline"].predict(X))
    p = np.clip(S.p["pipeline"].predict(X), -1.0, 1.0)
    d = S.reference.distance(X) if S.reference is not None else [None] * len(rows)
    out = []
    for r, yi, pi, di in zip(rows, y, p, d):
        dom = ood.assess(None if di is None else float(di))
        e_y, e_y_note = _expected(dom["level"], config.ERR_Y2D["known"], config.ERR_Y2D["new"])
        e_p, _ = _expected(dom["level"], config.ERR_POISSON["known"], config.ERR_POISSON["new"])
        out.append({
            "input": {k: r[k] for k in ("formula", "nat", "a", "b", "gamma", "thickness", "spg_number")} | {"area_per_atom": r.get("area_per_atom") or area_per_atom(r["a"], r["b"], r["gamma"], r["nat"])},
            "stiffness": {"value": float(yi), "unit": "N/m", "expected_abs_error": e_y, "error_basis": e_y_note,
                          "error_known_family": config.ERR_Y2D["known"], "error_new_family": config.ERR_Y2D["new"], "position": _band(float(yi))},
            "poisson": {"value": float(pi), "unit": "", "expected_abs_error": e_p, "error_known_family": config.ERR_POISSON["known"], "error_new_family": config.ERR_POISSON["new"],
                        "confidence": "low", "note": f"The Poisson-ratio model explains little variance on new families (R² {config.R2_POISSON_NEW:.2f}); treat it as a rough guide only."},
            "domain": dom,
            "notes": ["Screening aid trained on C2DB monolayers (GPAW/PBE). Good for ranking candidates; not a replacement for DFT.",
                      "Not validated for bilayers or for materials unlike C2DB (magnetic and unstable cases were not specially treated)."],
        })
    return out


@app.get("/api/health")
def health():
    return {"status": "ok", "models_loaded": S.y is not None, "distance_check": S.reference is not None}


@app.get("/api/models")
def models():
    c = config.CARDS.get("models", {})
    return {"versions": config.CARDS.get("versions", {}), "domain": config.CARDS.get("domain"), "how_to_read": config.CARDS.get("how_to_read"),
            "models": {k: c[k] for k in ("taskA_Y2D", "taskA_poisson") if k in c}, "distance_check_available": S.reference is not None,
            "distance_thresholds": {"typical": config.DIST_TYPICAL, "unusual": config.DIST_UNUSUAL}}


def _guard(fn):
    try:
        return fn()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/api/predict")
def predict(req: PredictRequest):
    def go():
        parse_formula(req.formula)
        return run([req.model_dump()])[0]
    return _guard(go)


@app.post("/api/predict/structure")
async def predict_structure(file: UploadFile = File(...)):
    content = await file.read(MAX_UPLOAD + 1)
    if len(content) > MAX_UPLOAD:
        raise HTTPException(status_code=413, detail="file larger than 2 MB")

    def go():
        inputs, warns = describe(read_structure(content, file.filename or ""))
        res = run([inputs])[0]
        res["derived_from_structure"] = True
        res["warnings"] = warns
        return res
    return _guard(go)


REQUIRED = ["formula", "nat", "a", "b", "gamma", "thickness", "spg_number"]


@app.post("/api/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    content = await file.read(MAX_UPLOAD + 1)
    if len(content) > MAX_UPLOAD:
        raise HTTPException(status_code=413, detail="file larger than 2 MB")
    reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig", errors="replace")))
    missing = [c for c in REQUIRED if c not in (reader.fieldnames or [])]
    if missing:
        raise HTTPException(status_code=422, detail=f"CSV is missing columns: {', '.join(missing)} (required: {', '.join(REQUIRED)})")
    rows, errors = [], []
    for i, rec in enumerate(reader, start=2):
        if len(rows) + len(errors) >= MAX_BATCH:
            errors.append({"line": i, "error": f"batch limited to {MAX_BATCH} rows; remaining rows ignored"})
            break
        try:
            req = PredictRequest(formula=rec["formula"].strip(), nat=int(rec["nat"]), a=float(rec["a"]), b=float(rec["b"]), gamma=float(rec["gamma"]),
                                 thickness=float(rec["thickness"]), spg_number=int(rec["spg_number"]))
            parse_formula(req.formula)
            rows.append((i, req.model_dump()))
        except Exception as exc:
            errors.append({"line": i, "error": str(exc).splitlines()[0][:200]})
    results = run([r for _, r in rows]) if rows else []
    return {"n_ok": len(results), "n_errors": len(errors), "errors": errors,
            "results": [{"line": ln, **{k: res[k] for k in ("input", "stiffness", "poisson", "domain")}} for (ln, _), res in zip(rows, results)]}


if config.FRONTEND_DIST.exists():                       # single-container deployment: serve the built React app
    app.mount("/", StaticFiles(directory=config.FRONTEND_DIST, html=True), name="frontend")
