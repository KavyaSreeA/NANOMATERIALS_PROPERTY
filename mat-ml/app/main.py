"""Simple results viewer + predictor.   cd mat-ml ; python -m uvicorn app.main:app --reload     then open http://127.0.0.1:8000
Read-only: it serves files from results/ and models/ and runs the saved Task A model. It never trains or writes anything."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
RES, MODELS, STATIC = ROOT / "results", ROOT / "models", Path(__file__).parent / "static"
app = FastAPI(title="2D-materials ML results", version="1.0",
              description="Read-only API over the saved results of the study (random vs family-grouped evaluation) and the saved Task A model.")


def _json(name: str):
    p = RES / name
    if not p.exists():
        raise HTTPException(404, f"{name} not found")
    return json.loads(p.read_text(encoding="utf-8"))


def _csv(name: str):
    p = RES / name
    if not p.exists():
        raise HTTPException(404, f"{name} not found")
    return json.loads(pd.read_csv(p).to_json(orient="records"))


@app.get("/api/health")
def health():
    return {"status": "ok", "models": sorted(p.stem for p in MODELS.glob("*.joblib")), "results_files": len(list(RES.glob("*")))}


@app.get("/api/headline")
def headline():
    """Cluster-bootstrap metrics (Y2D, Poisson, BiDB binding energy / gap) per split scheme with 95% CIs."""
    return _csv("phaseC_cluster_bootstrap_metrics.csv")


@app.get("/api/panel/{db}")
def panel(db: str):
    """Per-property family/random MAE ratio, skill retention and diagnostics. db = c2db | jarvis."""
    if db not in ("c2db", "jarvis"):
        raise HTTPException(404, "db must be c2db or jarvis")
    return _csv("panel_summary.csv" if db == "c2db" else "panel_summary_jarvis.csv")


@app.get("/api/hypotheses")
def hypotheses():
    return {"c2db": _json("panel_hypotheses.json"), "jarvis": _json("panel_hypotheses_jarvis.json"), "replication_checks": _json("panel_replication_checks.json")}


@app.get("/api/robustness")
def robustness():
    """Graph network (3 seeds) and nested tuning results."""
    return {"gnn_vs_lgbm": _json("phaseA_cgcnn_vs_lgbm.json"), "gnn_seeds": _json("phaseA_cgcnn_seeds.json"),
            "tuning_taskA": _json("phaseC_tuning.json"), "tuning_taskB": _json("phaseC_tuning_taskb.json")}


@app.get("/api/model-cards")
def model_cards():
    p = MODELS / "model_cards.json"
    if not p.exists():
        raise HTTPException(404, "model_cards.json not found")
    return json.loads(p.read_text(encoding="utf-8"))


@app.get("/api/figures")
def figures():
    return [{"name": p.name, "url": f"/figures/{p.name}"} for p in sorted((RES / "figures").glob("*.png"))]


class Material(BaseModel):
    formula: str = Field(..., examples=["MoS2"], description="reduced chemical formula")
    a: float = Field(..., gt=0, description="in-plane lattice length a (A)")
    b: float = Field(..., gt=0, description="in-plane lattice length b (A)")
    gamma: float = Field(..., gt=0, lt=180, description="in-plane angle (deg)")
    area_per_atom: float = Field(..., gt=0, description="A^2 per atom")
    nat: int = Field(..., gt=0)
    thickness: float = Field(..., ge=0, description="slab thickness (A)")
    spg_number: int = Field(..., ge=1, le=230)


@app.post("/api/predict/y2d")
def predict_y2d(m: Material):
    """2D Young's modulus (N/m) from the saved LightGBM model. Expect about 14 N/m MAE for known structure families and about 20 N/m for unseen ones."""
    if not re.fullmatch(r"([A-Z][a-z]?\d*(\.\d+)?)+", m.formula):
        raise HTTPException(422, "formula must look like MoS2 or Bi2Se3")
    try:
        from src.predict import featurize_task_a, predict_table      # heavy imports only when needed
        X = featurize_task_a(pd.DataFrame([m.model_dump()]))
        y = float(predict_table("taskA_Y2D", X)[0])
    except Exception as e:                                             # unknown element, missing model, ...
        raise HTTPException(422, f"could not predict: {e}")
    return {"Y2D_N_per_m": y, "note": "Point estimate, no uncertainty interval. Typical error is about 14 N/m (known structure family) to 20 N/m (unseen family); not DFT-validated for this input."}


app.mount("/figures", StaticFiles(directory=RES / "figures"), name="figures")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
