# Phase B design: can split-induced inflation be predicted from labels alone?  (written BEFORE any panel model was fitted)

Date written: 2026-10-04. Earlier Task A / Task B / MLIP results are not modified; outputs of this phase are prefixed `panel_`.

## 1. Why this phase (and what is NOT new)
Random cross-validation overestimating materials-ML performance is established: Meredig et al. 2018 (leave-one-cluster-out), Li et al. (MD-HIT redundancy control; Nat. Commun. 2023, npj Comput. Mater. 2024),
MatFold (standardised leave-group-out protocols, which also reports that properties differ in sensitivity), kernelised LOCO-CV (Liverpool), and for BiDB the BiMat-ML paper already splits at the
monolayer level. **None of that is claimed here.** What this phase tests is narrower and falsifiable:
> *The amount by which a random split inflates a property's apparent skill can be forecast from labels alone, before any model is trained,
> by how well a pure group-identity predictor (structure family or chemical system) already does under random splits.*
If true this is a cheap pre-modelling diagnostic for leakage-prone property/dataset pairs. If false, that is reported as a negative result.

## 2. Panel (fixed by rule, not by looking at results)
Source: all 16,905 C2DB materials (data.json + structure.json). A numeric field enters the panel if ALL hold:
(i) >= 3,000 labelled materials; (ii) not an identifier, symmetry integer or purely geometric quantity (`number`, `lgnum`, `thickness`, `energy`);
(iii) not degenerate: fewer than 50% of labelled values within 1e-6 of the modal value; (iv) finite (inf removed); (v) ONE representative per property family
(near-duplicates such as gap_dir, vbm/cbm derivatives, alphay_el, plasmafrequency_y, *_hse variants of the same quantity are not added as separate targets, except the PBE and HSE gaps, which are distinct calculations).
Candidates and families, in priority order: hform; ehull; gap (PBE); gap_hse; evac; efermi; vbm; magmom; alphax_el; plasmafrequency_x; emass_cbm; Y2D; poisson (Task A cleaning).
Excluded by the rules and listed in the output: dipz, minhessianeig and any other degenerate field, fields with < 3,000 labels. Transform: ln if all values > 0 and max/min > 100, else raw (decided per target by this rule only).
BiDB (binding energy, gap; grouped by monolayer) is analysed separately as an out-of-panel dataset.

## 3. Features, models, splits (held constant across targets)
Features: the 141 Task A features (Magpie composition + geometry + space-group number); no DFT outputs. Model: LightGBM exactly as in Task A (500 trees, lr 0.05, 31 leaves, subsample 0.8, colsample 0.5).
Splits (5-fold, seeds 42, 43, 44): random; grouped by chemical system; grouped by structure family (anonymous formula + layer group). Out-of-fold predictions are stored.
Baselines per fold: global training mean, chemical-system training mean, family training mean (unseen group -> global mean).

## 4. Quantities
Per target t and scheme s (pooled out-of-fold, seed-averaged): MAE_s(t). Mean-predictor MAE_0(t) (same folds).
- Skill S_s = 1 - MAE_s / MAE_0.   **Skill retention** R_s = S_s / S_random (1 = nothing lost; 0 = all skill lost). Primary outcome: R_family. Secondary: log(MAE_family / MAE_random); R_chemsys.
- Label-only diagnostics (no model, no features): 
  D_family = 1 - MAE(family-mean predictor, random 5-fold) / MAE(global mean, same folds)   (out-of-fold group-identity skill);  D_chem likewise for chemical systems;
  ICC_family = ANOVA ICC(1) of the (transformed) label over families; covariates: n labelled, number of families, kurtosis of the label.
- Competing explanation (feature-space shift): median distance of a test material to its nearest training material in standardised feature space, ratio grouped/random. Because features are identical across targets this cannot explain
  target-to-target variation; it is computed to show that, and for the later cross-dataset step.

## 5. Hypotheses (fixed now)
- **H1 (descriptive).** For each target, R_family < 1 and MAE_family > MAE_random. Report the share of targets with MAE_family/MAE_random > 1.05.
- **H2 (primary).** Across the panel targets, Spearman rho(D_family, R_family) <= -0.6, and the 95% bootstrap interval of rho (resampling targets, section 6) lying entirely below 0. Interpretation: the more skill a pure family label gives, the less skill survives a family-grouped split.
- **H3.** A leave-one-target-out linear fit R_family ~ D_family has smaller mean absolute error than the intercept-only fit (ratio of LOTO errors < 0.8).
- **H4 (axis).** Same test for the chemical-system axis: rho(D_chem, R_chemsys) <= -0.6. Report which axis (family or chemistry) explains each target's loss; classify targets descriptively.
- **H5 (competitors).** D_family should be at least as strongly associated with R_family as ICC_family; covariates (n, number of families, kurtosis) alone should not match it. Reported either way.
Falsification: if rho(D_family, R_family) > -0.3 or the LOTO ratio >= 1, the diagnostic is reported as NOT predictive.

## 6. Uncertainty and power (stated honestly)
n ~ 12-13 panel targets, so correlations are imprecise. Confidence intervals: (a) bootstrap over families inside each target for MAE ratios and R values (resample families, recompute both schemes' pooled MAEs from the stored out-of-fold predictions);
(b) for rho and the LOTO error: bootstrap over targets (with the per-target values from (a) resampled as well). Results are labelled exploratory. A prospective test on datasets not yet run (JARVIS-2D, Matbench) is planned as Phase B-2 and its
predictions will be written down before those models are run.

## 7. Reproducibility
`python -m src.panel` (builds `cache/c2db_all.pkl`, runs the grid, stores out-of-fold predictions in `cache/panel_oof/`), `python -m src.panel_analysis` (tables, bootstrap, figures in `results/figures/`).
