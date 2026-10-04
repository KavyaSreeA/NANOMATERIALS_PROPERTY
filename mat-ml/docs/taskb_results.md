# Task B: predicting BiDB interlayer labels from monolayer information

Design fixed beforehand in `results/taskB_design.md` (with a dated addendum); code `src/taskb.py`; outputs `results/taskB_*`. Run: `python -m src.taskb` (or `python -m src.train --task B`), 3,562 s.
No pass/fail thresholds were defined. All metrics are in original units (binding energy in meV/A^2, distance in A); "ln R2" is the R2 of the log-transformed target.

## 1. Data, labels and features
- **Labels (DFT, PBE-D3, rigid C2DB layers):** `binding_energy_zscan` (meV/A^2 per interface, positive = bound) and `distance` (A; vertical gap min z(top) - max z(bottom)): primary. `binding_energy_gs` (definition not documented) is secondary and never used for physical conclusions.
  How these definitions were established from the files is in `results/mlip/bidb_preregistration.md`.
- **Rows:** 10,192 homobilayers of 992 monolayers in 109 structure families (anonymous stoichiometry + monolayer layer group; 50 families have one monolayer; largest 1,148 bilayers). All 10,192 stacking uids were decoded.
- **Validity rule (fixed beforehand):** target > 0 and <= U (upper tail of log10 target, Q3 + 3 IQR). Kept: zscan 9,993 (U = 277 meV/A^2; 26 non-positive and 173 above U removed), distance 10,189 (3 non-positive removed), gs 9,740.
- **Features.** `mono_basic` (140): Magpie composition (132) + n_elements + monolayer geometry (a, b, gamma, area per atom, atoms, thickness) + monolayer layer group.
  `mono_stiffness` (147, requested primary set): + C2DB monolayer stiffness through the uid map (ln Y2D, ln C11, ln C22, C12, shear, Poisson ratio, indicator); 977 of 992 monolayers (95% of bilayers) have a clean C2DB tensor (Task A cleaning), the rest are missing + flagged.
  `mono_stiffness_stacking` (158, extension): + stacking descriptors from the uid (rotation matrix, z-flip, sin/cos of the shifts) + bilayer layer group and inversion symmetry. `slide_stability` and other DFT outputs were excluded (label leakage / out of scope).
- **Because the primary feature sets describe only the monolayer, every stacking of a monolayer gets the same prediction.** The share of ln-variance that lies between monolayers is 0.955 (zscan), 0.771 (distance) and 0.856 (gs): this is the ceiling for any monolayer-only model.

## 2. Evaluation
Five-fold CV with three schemes: random rows; grouped by monolayer (all stackings together); grouped by family (StratifiedGroupKFold on target deciles; groups disjoint). Models: Ridge, random forest (300), LightGBM (500), fitted on ln(target). Baseline: mean predictor (arithmetic mean of the training targets; the geometric mean is also in the files).
Seed 42: full grid. Seeds 42-46: LightGBM and baselines on `mono_basic` and `mono_stiffness`. "Spread" = std and min-max over the five folds; "seed sd" = std over seeds of the fold means.

**Leakage (fraction of test bilayers whose key also occurs in training, zscan, seed 42).**

| Scheme | monolayer | family | formula | chemical system |
|---|---|---|---|---|
| random | **0.993** | 1.000 | 0.995 | 0.997 |
| grouped by monolayer | 0.000 | 0.888 | 0.209 | 0.371 |
| grouped by family | 0.000 | 0.000 | 0.107 | 0.296 |

## 3. Results (primary: LightGBM on `mono_stiffness`)

**Binding energy (zscan), MAE in meV/A^2.**

| Scheme | MAE, 5-seed mean (seed sd) | folds, seed 42: mean +/- sd (min-max) | R2 (seed sd) | ln R2 | Spearman | Mean predictor MAE | MAE reduction vs mean predictor |
|---|---|---|---|---|---|---|---|
| random | **2.42** (0.01) | 2.43 +/- 0.06 (2.36-2.52) | 0.985 (0.003) | 0.940 | 0.944 | 18.51 | 87% |
| grouped by monolayer | **9.07** (0.42) | 9.37 +/- 1.71 (6.69-11.10) | 0.483 (0.052) | 0.643 | 0.833 | 18.51 | 51% |
| grouped by family | **13.59** (0.30) | 13.85 +/- 4.67 (10.64-22.06) | 0.200 (0.019) | 0.351 | 0.707 | 18.54 | 27% |

Per-fold MAE (seed 42): random 2.42, 2.45, 2.52, 2.36, 2.40; monolayer-grouped 9.38, 9.07, 6.69, 11.10, 10.59; family-grouped 12.64, 12.61, 11.29, 10.64, **22.06** (mean predictor on that fold: 21.69, so LightGBM is worse than the mean predictor there; it is better in the other four folds).
The family folds contain very different families (the largest has 1,148 bilayers), which explains the spread.

**Interlayer gap (distance), MAE in A.**

| Scheme | MAE, 5-seed mean (seed sd) | folds, seed 42 (min-max) | R2 | Spearman | Mean predictor MAE | MAE reduction |
|---|---|---|---|---|---|---|
| random | 0.248 (0.001) | 0.24-0.25 | 0.660 | 0.764 | 0.415 | 40% |
| grouped by monolayer | 0.278 (0.005) | 0.25-0.28 | 0.513 | 0.751 | 0.415 | 33% |
| grouped by family | 0.316 (0.002) | 0.29-0.34 | 0.391 | 0.684 | 0.415 | 24% |

**Secondary (gs; seed 42; LightGBM on `mono_stiffness`).** MAE 2.34 / 2.97 / 3.59 meV/A^2 for random / monolayer / family splits (R2 0.881 / 0.694 / 0.575; mean predictor 6.2). Reported for continuity with the original brief only.

**Random-versus-grouped inflation.** For binding energy the grouped MAE is 3.7x (monolayer) and 5.6x (family) the random MAE; for the gap it is 1.12x and 1.27x. The random-split ln R2 (0.940) is essentially the between-monolayer ceiling (0.955): the model has learned to recognise monolayers, which random splits allow because 99.3% of test bilayers have their monolayer in the training data.

## 4. What the features contribute
- **C2DB stiffness: no detectable benefit.** `mono_basic` vs `mono_stiffness`, LightGBM, five seeds, MAE: binding energy random 2.443 vs 2.422, monolayer-grouped 9.064 vs 9.069, family-grouped 13.539 vs 13.589 (seed sd 0.01-0.42); gap 0.247 vs 0.248, 0.279 vs 0.278, 0.317 vs 0.316.
  In the seed-42 paired folds, stiffness was better in 5/5 folds only for random binding energy (by 0.04 meV/A^2) and in 2/5 and 1/5 folds for grouped binding energy. The random forest shows the same picture.
- **Stacking descriptors (extension; seed 42 only):** large gain for the gap, small or none for binding energy. Gap, LightGBM MAE: random 0.25 -> 0.15 (R2 0.66 -> 0.85), monolayer-grouped 0.27 -> 0.20 (0.57 -> 0.70), family-grouped 0.32 -> 0.27 (0.38 -> 0.49). Binding energy: random 2.43 -> 2.02, monolayer-grouped 9.37 -> 8.80, family-grouped 13.85 -> 13.92 (no change).
- **Models.** Random forest and LightGBM are close (within about 0.5 meV/A^2 under grouped splits). Ridge is much worse and, under family splits, worse than the mean predictor in 3 of 5 folds (MAE 20.7 vs 18.5).

## 5. Sensitivity to the outlier rule (LightGBM, `mono_stiffness`, seed 42)
- **No filtering:** the mean predictor's MAE is 34.4 meV/A^2 and LightGBM reaches R2 0.85 / 0.20 / 0.06 (random / monolayer / family) with MAE 4.2 / 19.0 / 21.9: the extreme z-scan values (up to 7,970 meV/A^2) dominate, so the filter is necessary for binding energy.
- **Two-sided rule:** identical to the upper-tail rule for binding energy (no positive value lies below the lower bound; the reasoning in the design text about weak binders did not hold for this dataset, see the addendum in `taskB_design.md`).
  For the gap it also removes 109 gaps below 1.49 A (10,080 kept) and the gap results improve: family-grouped MAE 0.278 (R2 0.489) vs 0.320 (0.375); monolayer-grouped 0.251 vs 0.271. Small gaps (nested terminations) are hard to predict, so the gap numbers above depend on the rule.

## 6. Reading and limits
- A random split is not a meaningful generalisation estimate here: each monolayer appears about ten times with identical monolayer features. The honest estimates are the grouped ones: binding energy to new monolayers about 9 meV/A^2 (R2 0.48, half the mean-predictor error), to new structure families about 13.6 meV/A^2 (R2 0.20, 27% below the mean predictor, with folds from 10.6 to 22.1).
- The gap is only moderately predictable from the monolayer (24-40% below the mean predictor); knowing the stacking helps substantially.
- Limits: labels come from one workflow (PBE-D3 on rigid layers; homobilayers only); the uid map is verified at formula level; family sizes are very uneven; standard deviations over folds and seeds do not include the uncertainty from the finite number of monolayers (961-992);
  `gs` is undocumented; thickness, layer-group and composition features were not tuned and no hyper-parameter search was done; only tabular models were tried.
- Not done: graph neural networks; hyper-parameter tuning; leave-one-chemical-system-out; predicting stacking-resolved energies from stacking descriptors for unseen monolayers beyond the extension above.
