# NANOMATERIALS_PROPERTY

Machine learning for the **mechanical properties of 2D (van der Waals) materials**, with an emphasis on honest evaluation: do the models still work on materials unlike the ones they were trained on?
The code, results and documentation are in [`mat-ml/`](mat-ml/). Start with [`mat-ml/README.md`](mat-ml/README.md) and [`mat-ml/docs/methods_and_models.md`](mat-ml/docs/methods_and_models.md).

## What this repository contains
| Part | What | Where |
|---|---|---|
| **Task A** | Predict monolayer 2D Young's modulus (N/m) and Poisson ratio from composition and geometry (C2DB, 7,258 stable monolayers; external check on 186 JARVIS materials) | `mat-ml/src/`, `mat-ml/results/`, [`results_summary.md`](mat-ml/docs/results_summary.md) |
| **Task B** | Predict BiDB bilayer binding energy and interlayer gap from monolayer information (incl. C2DB stiffness via a uid map) | `mat-ml/src/taskb.py`, [`taskb_results.md`](mat-ml/docs/taskb_results.md) |
| **ML-potential validation** | MACE-MP-0, MACE-MPA-0, CHGNet checked against C2DB/JARVIS stiffness and BiDB binding energies (a separate environment; closed) | `mat-ml/src/mlip/`, [`mlip_calibration.md`](mat-ml/docs/mlip_calibration.md) |
| Literature | notes and a cautious comparison with the seven papers in `research papers/` | [`literature_notes.md`](mat-ml/docs/literature_notes.md), [`literature_comparison.md`](mat-ml/docs/literature_comparison.md) |
| Draft results section | Part 1 (monolayer ML), written from the saved outputs | [`draft_results_part1.md`](mat-ml/docs/draft_results_part1.md) |

## Main findings (details and caveats in the docs)
- **Random cross-validation overestimates accuracy, and the reason is structure-level overlap.** LightGBM predicts Y2D with MAE 13.8 N/m (R2 0.87) under random CV and 19.9 N/m (R2 0.70) when whole structure families are held out (1.44 +/- 0.02 over five seeds). Holding out chemical systems makes no difference (1.00x).
- **For bilayer binding energy the inflation is larger:** MAE 2.4 meV/A^2 (random) vs 9.1 (new monolayers) vs 13.6 (new families), against 18.5 for the mean predictor, because every monolayer appears about ten times with identical monolayer features.
- **External check:** on 186 JARVIS materials the ranking transfers (Spearman 0.82, 95% CI 0.72-0.90); on the 55 with chemistries absent from the training set MAE is 29 N/m (95% CI 20-40).
- **ML potentials:** MACE-MPA-0 passes a pre-registered stiffness gate on C2DB but not on JARVIS or combined; for BiDB it reproduces binding-energy ranking and magnitude (Spearman 0.90, median ratio 1.10) but fails the interlayer-gap criterion (median error 0.18 A vs 0.10 A allowed). Bilayer in-plane stiffness is consistent with additive layers in this model (median ratio 2.008, CI [1.95, 2.03]; 60 bilayers; not DFT-validated).
- **Not established:** DFT-validated bilayer stiffness; behaviour for magnetic or metallic layers; whether a graph neural network would narrow the family gap.

## Reproduce
```
cd mat-ml
pip install -r requirements.txt
python -m pytest tests -q
python -m src.train --task A       # Task A
python -m src.train --task B       # Task B
python -m src.diagnostics          # parity and residual plots, reproduction check
```
Final models and model cards are in [`mat-ml/models/`](mat-ml/models/); every number in the docs comes from files in `mat-ml/results/`.

## Data (not stored in this repository)
| Dataset | Used for | Source |
|---|---|---|
| C2DB | main training data (monolayer stiffness tensors) | <https://c2db.fysik.dtu.dk/> |
| JARVIS-DFT 2D | external check, ML-potential references | <https://jarvis-tools.readthedocs.io/en/master/databases.html> (`get_jarvis_2d.py`) |
| BiDB (van der Waals Bilayer Database) | Task B labels, ML-potential interlayer validation | <https://2dhub.org/bidb/bidb.html>; Pakdel et al., Nat. Commun. 15, 932 (2024) |
Place the files under `Dataset/` (paths in `mat-ml/config.yaml`; see `mat-ml/data/README.md`). `Dataset/` is deliberately not tracked (the C2DB tree is about 2.3 GB).

## Status and limits
One training database and one external set; no hyper-parameter tuning; tabular models only; no separate held-out test set (generalisation is measured by grouped cross-validation and the external set). Released under the MIT License (see [`LICENSE.MD`](LICENSE.MD)).
