# Stiffness predictor (web app)

A screening aid for 2D-material stiffness. Enter a formula and cell geometry (or upload a CIF/POSCAR) and get the 2D Young's modulus (N/m) and Poisson ratio, with a typical error and a warning when the material is unlike the training data.

- **Backend:** FastAPI (`webapp/backend/app`) serving the saved Task A LightGBM models from `mat-ml/models`.
- **Frontend:** React + Vite (`webapp/frontend`).
- **Scope:** monolayers only (Task A). Not validated for bilayers. Poisson ratio is low-confidence (new-family R² about 0.14).

## Run locally (PowerShell)

```powershell
# 1. backend (use the project environment; pin the versions the models were trained with)
pip install -r webapp/backend/requirements.txt
cd webapp/backend
uvicorn app.main:app --port 8000

# 2. frontend (second terminal)
cd webapp/frontend
npm install
npm run dev          # http://localhost:5173 (proxies /api to port 8000)
```

Single-server mode: `npm run build` in `webapp/frontend`, then open http://127.0.0.1:8000 (the backend serves `dist`).

If `pymatgen`/`matminer` fail to install because of `bibtexparser`, run
`pip install "setuptools<70" wheel` then `pip install --no-build-isolation "bibtexparser<2"` first.

## Enable the "how close is this to the training data" check

The check needs a reference file built from the C2DB training features, which are not in the repository. From `mat-ml` (with `Dataset/` and the feature cache present):

```powershell
cd mat-ml
python ../webapp/scripts/build_reference.py
```

This writes `mat-ml/models/taskA_reference.npz` (git-ignored because it derives from C2DB). Without it the app still works; it shows both error values and reports the check as unavailable.

## Tests

```powershell
cd webapp/backend
pytest tests
```

## Docker

```powershell
docker compose -f webapp/docker-compose.yml up --build
```

## Licences and credits

Models are trained on C2DB; JARVIS-DFT and BiDB were used for evaluation. Confirm each dataset's licence before any public deployment (the About page lists them with the items still to confirm).

## Roadmap

1. **Next iteration: shear effects.** Needs a shear-modulus target (C66). The MLIP shear convention is not yet verified and DFT/GPU checks are required, so nothing is built yet.
2. Bilayer predictions, after DFT validation of the bilayer/monolayer stiffness ratio.
