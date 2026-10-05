# Methods and models: one-page manifest

Everything that was trained or evaluated, in one place. Numbers are in `results/headline_metrics.md` (generated from the result CSVs); interpretation is in `results_summary.md`, `taskb_results.md` and `mlip_calibration.md`.

## 1. What is predicted
| Task | Target | Definition | Unit | Fitted on | Data |
|---|---|---|---|---|---|
| A | Y2D (2D Young's modulus) | (C11 C22 - C12^2) / C22 from the C2DB elastic tensor | N/m | ln scale | 7,258 stable C2DB monolayers |
| A | Poisson ratio | C12 / C22, \|nu\| <= 1 | - | raw | 7,229 |
| B | binding energy (z-scan) | -(E_bilayer - 2 E_monolayer)/A x 1000, per interface, positive = bound (BiDB, PBE-D3, rigid layers) | meV/A^2 | ln scale | 9,993 bilayers (961 monolayers) |
| B | interlayer gap | min z(top) - max z(bottom) (BiDB `distance`) | A | ln scale | 10,189 bilayers (992 monolayers) |
| B (secondary) | `binding_energy_gs` | definition undocumented | meV/A^2 | ln scale | 9,740 |
Not predicted: band gap, formation energy or any electronic property (they are deliberately excluded as inputs and were not targets).

## 2. Models actually trained
| Model | Where | Settings |
|---|---|---|
| Ridge | A, B | median imputer, StandardScaler, RidgeCV (alpha 1e-3 ... 1e3, 13 values) |
| Random forest | A, B | median imputer, 300 trees, max_features 0.33 |
| **LightGBM (primary)** | A, B | median imputer, 500 trees, learning rate 0.05, 31 leaves, subsample 0.8, colsample 0.5 |
| Mean predictor (arithmetic; geometric also run) | A, B | training mean; no features |
| Family-mean and chemical-system-mean predictors | A | mean of ln Y2D within the training group, global mean for unseen groups |
| Not trained | - | graph neural networks, hyper-parameter search, ensembles |
Saved final models (`models/`, joblib; LightGBM, trained on all valid data): `taskA_Y2D`, `taskA_poisson`, `taskB_binding_energy_zscan`, `taskB_distance`, each with a feature list and `models/model_cards.json`. Their in-sample error is optimistic by construction; use the cross-validated numbers.

## 3. Features
Task A (141): 132 Magpie statistics (22 elemental properties x 6 statistics, matminer, on the reduced formula) + a, b, gamma, area per atom, number of atoms, slab thickness, layer count (constant), space-group number, number of elements.
Task B: `mono_basic` 140 (Magpie + n_elements + monolayer geometry + layer group); `mono_stiffness` 147 (+ C2DB stiffness through the uid map: ln Y2D, ln C11, ln C22, C12, shear, Poisson ratio, indicator); `mono_stiffness_stacking` 158 (+ stacking descriptors from the uid; extension).
Excluded deliberately: DFT outputs (hull energy, formation energy, band gap, stability and magnetic flags) and `slide_stability` (label leakage). **No automated feature selection** was run (no filter, wrapper or L1 selection, no importance analysis); signal was assessed by feature-group ablation (see `results_summary.md`).

## 4. Data preparation
Stability from the tensor (positive definite, asymmetry <= 10%); corrupt JARVIS tensors removed; BiDB targets kept if > 0 and below the upper log-IQR bound (Q3 + 3 IQR of log10); median imputation inside each training fold; no deduplication (polymorphs kept). Counts in `results/data_audit.json`, `results/taskB_run_info.json`.

## 5. Evaluation protocol
| Item | Setting |
|---|---|
| Resampling | 5-fold cross-validation; grouped folds use StratifiedGroupKFold on ten target quantile bins (target balance only; groups stay disjoint) |
| Schemes | A: random, cluster-stratified, grouped by chemical system, grouped by structure family. B: random, grouped by monolayer, grouped by family |
| Seeds | split and model seed 42 for the full grids; seeds 42-46 for the LightGBM robustness runs |
| Validation set | **none**: no hyper-parameters were tuned, so no validation split was needed |
| Held-out test set | **none separate**: generalisation is measured by grouped CV plus an external set (186 JARVIS materials; train on all of C2DB), reported with bootstrap intervals |
| Leakage checks | fraction of test materials whose chemical system, formula, family or monolayer also occurs in training, per scheme (`task_A_comparison.md`, `taskB_comparison.md`); unit tests that grouped folds never share groups (`tests/test_core.py`) |
| Baselines | mean predictor, family-mean, chemical-system-mean; Task B also reports the between-monolayer variance share as a ceiling |
| Uncertainty | std over folds (min-max), std over seeds, bootstrap 95% intervals (over JARVIS rows; cluster bootstrap over monolayers for BiDB-based ML-potential results) |
| Metrics | MAE, RMSE, R2 (original units), ln R2, Spearman, slope through the origin, median ratio, bias; formulas in section 5b |
| Diagnostics | out-of-fold parity and residual plots: `results/diagnostics/` (`python -m src.diagnostics`) |

## 5b. Metric formulas (y = reference, p = prediction, n = number of test materials)
| Metric | Formula | Notes |
|---|---|---|
| MAE | (1/n) sum \|p - y\| | original units |
| RMSE | sqrt[(1/n) sum (p - y)^2] | original units |
| R2 | 1 - sum (y - p)^2 / sum (y - mean(y))^2 | negative when worse than the mean predictor |
| ln R2 | R2 computed on ln(y) and ln(p) | stable under extrapolation (back-transformed R2 can blow up) |
| Spearman rho | rank correlation of p and y | ranking quality |
| Slope through origin | sum(p y) / sum(y^2) | 0.89 = predictions 11% too small on average |
| Median ratio | median(p / y) | typical relative bias |
| Bias | mean(p - y) | systematic offset |
| Grouped / random ratio | MAE(grouped split) / MAE(random split) | size of the random-split inflation |
| Error reduction | 1 - MAE / MAE(mean predictor) | skill over the baseline |
| Between-monolayer share (Task B ceiling) | 1 - sum (ln y - group mean)^2 / sum (ln y - mean)^2 | upper bound for monolayer-only features |

## 5c. Reading the diagnostic plots (`results/diagnostics/`)
- Every point is an **out-of-fold** prediction (the model never saw that material or its group). Splits and seed are the headline ones; the recomputed per-fold MAE equals the reported value for all 60 folds (max difference 3.6e-15, `reproduction_check.json`).
- Panel titles show **pooled** metrics (all out-of-fold predictions together); the tables show **fold means**. MAE agrees; R2 differs slightly because it is not linear (Task A, family split: 0.740 pooled vs 0.703 fold mean).
- Task A: residuals grow with stiffness and the stiffest materials are under-predicted (mean residual about -4 N/m for the random and chemical-system splits, -8 N/m for unseen families).
- Task B: under the grouped splits the horizontal streaks are monolayer-level predictions (all stackings of a monolayer receive one value), so a streak's height is the reference's stacking spread. Residuals are heavy-tailed: the median residual is about 0, but a minority of materials are missed by 50-130 meV/A^2 in both directions, which MAE and R2 alone do not show.

## 6. Recorded run times (wall clock, this machine: i7-14700HX, 15.7 GB RAM)
| Run | Time |
|---|---|
| Task A, seed 42, full grid + JARVIS check | 997 s |
| Task A robustness, 5 seeds, 3 feature sets (+ RF) | 3,173 s |
| Task A spg ablation / prototype features | 4,631 s / 398 s |
| Task A external check with intervals | 335 s |
| Task B, full grid + 5 seeds + sensitivity | 3,562 s |
| Saved models, tests | about 1 min / 2 s |
The ML-potential work used a separate environment (`.venv-mlip`, GPU); its run times and settings are in `mlip_calibration.md`.

## 7. Reproduce
```
pip install -r requirements.txt            # (+ pytest for the tests)
python -m pytest tests -q                  # 13 fast tests
python -m src.data --audit                 # cleaning counts
python -m src.train --task A               # Task A grid (~17 min); python -m src.robustness ...  for seeds / ablations
python -m src.train --task B               # Task B (~1 h)
python -m src.diagnostics                  # parity / residual plots, reproduction check
python -m src.save_models                  # final models + model cards
python -m src.make_tables                  # results/headline_metrics.md
```
Raw data are not in the repository (sources in `data/README.md`).
