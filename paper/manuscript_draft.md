# Random splits overstate accuracy for 2D-materials property prediction because of structure-family overlap — across twelve properties, model classes and tuning

DRAFT v0.1 (2026-10-05). Numbers come from saved outputs (file names in brackets). Citations are from `positioning_and_related_work.md` and **must be verified** before submission. Figure/table numbers are placeholders. Nothing here has been peer reviewed.

## Abstract
Machine-learning models of materials properties are usually scored with random cross-validation. For 2D materials, whose databases contain many near-copies of the same structure prototype with different elements, we ask what random splits actually leak. Using the C2DB database (16,905 monolayers), we compare random, chemical-system-grouped and structure-family-grouped (anonymous formula + layer group) cross-validation with gradient-boosted trees on 141 composition-and-geometry features. For the 2D Young's modulus (7,258 mechanically stable materials), the mean absolute error rises from 13.7 to 19.8 N/m (factor 1.44, 95% cluster-bootstrap CI 1.30–1.62) when families are held out, whereas holding out chemical systems changes it by 1% (1.01 [0.99, 1.04]). The same family penalty appears for all twelve C2DB properties we tested (MAE ratio 1.12–1.92, every interval above 1), and it is not removed by a graph neural network (family/random ratio 1.38 [1.28, 1.47]) or by nested group-aware hyper-parameter tuning (ratio 1.54 [1.37, 1.77]; tuning improves random-split MAE by 6% and family-split MAE by 0%). We also tested whether a label-only statistic computed under random CV (the skill of a family-mean predictor) could forecast how much skill a property loses on unseen families; it could not (Spearman ρ = 0.00, CI [−0.60, 0.81]; pre-registered falsification criterion met). On bilayer binding energies (BiDB) the same logic applies at the monolayer level: grouping by monolayer raises the error 3.9-fold. We recommend family-grouped evaluation as the default for 2D-materials benchmarks and report the failed diagnostic so others need not repeat it.

## 1 Introduction
- Known: random CV overestimates materials-ML performance; redundancy and cluster hold-out (Meredig 2018; MD-HIT; MatFold; kernelised LOCO-CV; Matbench). BiMat-ML already splits BiDB by monolayer.
- Open for 2D layered materials: *which* kind of overlap inflates scores — chemistry or structure prototype — and whether this is a property-specific or general effect; whether model class or tuning changes it; whether it can be forecast from labels alone.
- Contributions (in order of strength of evidence): (1) in C2DB the inflation comes from structure-family overlap, not chemistry; (2) it is present for all twelve tested properties with cluster-bootstrap intervals; (3) it survives a GNN and nested tuning; (4) a pre-registered label-only forecast of the inflation fails; (5) monolayer-level leakage dominates bilayer binding-energy benchmarks. We do **not** claim a new model or a new state of the art.

## 2 Data and methods
**Data.** C2DB (GPAW/PBE) monolayers; stiffness tensor stability checks (positive definite, asymmetry ≤ 10%), Y2D = (C11C22 − C12²)/C22 in N/m, Poisson ν = C12/C22 with |ν| ≤ 1; 3,806 chemical systems (2,572 singletons) and 676 families for Y2D. JARVIS-DFT 2D (186 usable tensors) as an external set. BiDB bilayers for binding energy and layer gap. [`data_audit.json`, `external_ci_report.md`, `taskb_results.md`]
**Features.** 141 composition (132 Magpie) and geometry features; no DFT outputs for Task A. **Models.** Ridge, random forest, LightGBM (default), CGCNN-type network, LightGBM with nested tuning.
**Splits.** Random, chemical-system-grouped, family-grouped (family = anonymous formula + layer group); 5-fold, StratifiedGroupKFold on target quantile bins; seeds 42–46 (Task A), 42–44 (panel). Overlap fractions in Table 1 of the Part 1 draft.
**Statistics.** Seed standard deviations understate uncertainty (they hold the dataset fixed). All headline intervals are cluster bootstraps over families (2,000 resamples; 1,000 in the panel); paired differences use the same resample. [`phaseC_cluster_bootstrap.json`]
**Pre-registration.** Designs for the panel, graph-network comparison and robustness upgrade were written to `results/panel_design.md`, `phaseA_design.md`, `phaseC_design.md` before the corresponding models were fitted; thresholds were not changed afterwards.

## 3 Results
### 3.1 Family overlap, not chemistry, inflates random-split accuracy (Y2D)
LightGBM MAE (N/m): random 13.74 [12.08, 15.37]; chemical-system 13.91; family 19.82 [16.33, 24.24]. Family/random 1.44 [1.30, 1.62]; chemsys/random 1.01 [0.99, 1.04]. R²: 0.874 → 0.740 [0.602, 0.828]. The group-mean baseline predicts MAE 27.0 N/m under random CV from the family label alone and falls to the global mean (42.4 N/m) on unseen families. Prototype descriptors (layer group, anonymous formula) and the space-group number do not close the gap. [`phaseC_cluster_bootstrap_metrics.csv`, Part 1 draft §3.3–3.4]

### 3.2 The effect is general across properties (panel)
Twelve C2DB properties met the pre-registered selection rules (≥ 3,000 labels; non-degenerate; not an identifier or extensive quantity). Skill S = 1 − MAE/MAE(global mean); retention R = S_family/S_random.

| Property | n | MAE ratio family/random [95% CI] | Skill retained R_family [95% CI] | chemsys/random |
|---|---|---|---|---|
| hform | 16,905 | 1.49 [1.43, 1.56] | 0.90 [0.89, 0.91] | 1.07 |
| ehull | 16,905 | 1.48 [1.41, 1.55] | 0.63 [0.58, 0.68] | 1.03 |
| gap (PBE) | 8,699 | 1.53 [1.41, 1.64] | 0.79 [0.75, 0.83] | 1.07 |
| gap (HSE) | 3,363 | 1.42 [1.34, 1.51] | 0.72 [0.67, 0.78] | 1.06 |
| evac | 8,699 | 1.92 [1.77, 2.09] | 0.78 [0.73, 0.82] | 1.03 |
| efermi | 8,699 | 1.46 [1.39, 1.54] | 0.76 [0.73, 0.80] | 1.08 |
| vbm | 4,567 | 1.53 [1.43, 1.67] | 0.75 [0.69, 0.79] | 1.10 |
| alphax_el | 5,352 | 1.12 [1.09, 1.16] | 0.83 [0.74, 0.90] | 0.99 |
| plasmafrequency_x | 3,931 | 1.55 [1.43, 1.69] | 0.61 [0.51, 0.70] | 0.99 |
| emass_cbm | 3,532 | 1.23 [1.16, 1.31] | 0.68 [0.57, 0.78] | 1.00 |
| Y2D | 7,258 | 1.43 [1.29, 1.62] | 0.81 [0.71, 0.88] | 1.00 |
| Poisson ν | 7,229 | 1.28 [1.24, 1.32] | 0.34 [0.21, 0.43] | 0.99 |

All twelve ratios exceed 1.05 and all R_family < 1 (H1). For every property the family axis removes more skill than the chemical-system axis by the pre-registered rule. Y2D is mid-panel; evac is the most affected (the largest ratio) and Poisson's ratio keeps the least skill. [`panel_summary.csv`, `panel_hypotheses.json`, Fig. 7]

### 3.3 Robust to model class and tuning
*Graph network (Phase A, one seed, untuned).* CGCNN MAE: random 15.31, chemsys 15.86, family 21.12 N/m versus LightGBM 13.74, 13.91, 19.82. Family/random ratio 1.38 [1.28, 1.47] vs 1.44 [1.29, 1.62]; difference −0.06 [−0.19, +0.08]. On families the paired MAE difference LightGBM − CGCNN is −1.30 [−3.43, +0.58]: no evidence that the graph model is better, and none that it narrows the gap. Early stopping on a group-held-out validation set stops after 46 epochs on average for family folds versus 122 for random folds. [`phaseA_cgcnn_vs_lgbm.json`]
*Nested tuning (Phase C2).* 16 random configurations plus the default, inner CV with the same grouping as the outer split. MAE random 13.74 → 12.86 (gain 0.87 [0.55, 1.26]); family 19.82 → 19.83 (gain −0.01 [−0.28, +0.25]); ratio 1.44 → 1.54 [1.37, 1.77]. Tuned configurations chosen by inner CV did not transfer to unseen families in four of five folds. [`phaseC_tuning.json`]

### 3.4 A label-only diagnostic does not forecast the loss (negative result)
Pre-registered H2: Spearman between D_family (skill of a family-mean predictor under random CV) and R_family ≤ −0.6 with CI below 0. Observed ρ = 0.00 [−0.60, +0.81]; leave-one-property-out error ratio of a linear fit vs a mean = 1.01 (needed < 0.8). The chemical-system analogue gave ρ = −0.53 [−0.92, +0.16], ratio 0.88: the direction is as hypothesised but it does not meet the criteria. ICC, sample size, family count and kurtosis also do not explain R_family (|ρ| ≤ 0.44, CIs include 0). With twelve properties the test has low power; the safe reading is "no evidence of forecasting", not "impossible".

### 3.5 Bilayer binding energy: leakage is at the monolayer level
BiDB binding energy MAE (meV/Å²): random 2.42; monolayer-grouped 9.07; family-grouped 13.59; mean predictor 18.5. Ratios monolayer/random 3.86 [3.03, 4.80], family/random 5.68 [4.02, 7.92]; family-grouped R² 0.15 [−0.09, 0.33]. Layer gap: monolayer/random 1.09 [1.04, 1.15], family/random 1.29 [1.16, 1.52]. Monolayer C2DB stiffness added nothing; stacking descriptors helped the gap. [`taskb_results.md`, `phaseC_cluster_bootstrap.json`]

### 3.6 External check (JARVIS)
Ranking transfers (Spearman 0.82 [0.72, 0.90], n = 186); errors roughly double for chemistries absent from C2DB (MAE 29.1 N/m [20.5, 39.6], n = 55). The two databases disagree on shared materials (MAE 8.6 N/m, n = 106), which bounds achievable agreement. [`external_ci_report.md`]

## 4 Discussion and limitations
- Family-grouped evaluation should be the default for 2D-materials benchmarks; chemical-system grouping alone would have missed the effect in C2DB (ratio ≈ 1.0).
- Mechanism is not established. We show that the penalty tracks prototype overlap, not why (symmetry-determined structure–property maps vs. database construction by element substitution). Singleton-heavy chemistries (2,572 single-member systems) reduce the effect of chemsys grouping by construction.
- Single database (C2DB, one code/functional); GNN is one seed, untuned, with hand-chosen node features; tuning searched 16 configurations on Y2D only; panel uses default LightGBM; 12 properties give limited power for H2–H5.
- Folds with groups are uneven; polymorphs retained; stable materials only (≈14% of tensor-bearing entries removed).
- MLIP stiffness-ratio results (median 2.008 [1.95, 2.03] bilayer/monolayer, MACE-MPA-0, small strain, 60 bilayers, not DFT-validated) and the failed BiDB gap gate (D1) are reported in the supplement; no claim about bilayer stiffness is made.
- Not yet done: prospective check on an independent dataset (JARVIS-2D/Matbench) with predictions written beforehand; mechanistic analysis; multi-seed GNN; Task B tuning.

## 5 Reproducibility
Code in `mat-ml/src`, pre-registration files and outputs in `mat-ml/results`; seeds, fold files (`cache/phaseA_folds.json`) and OOF predictions saved. Raw C2DB/JARVIS/BiDB data are not redistributed.

## Suggested title options
1. "Random splits overstate accuracy in 2D-materials property prediction through structure-family overlap, not chemistry"
2. "Hold out the prototype: family-grouped evaluation of ML property models for 2D materials"
3. Add ", and a label-only diagnostic does not predict it" to either if venue space allows.

## To do before submission
1. Verify every citation (see `positioning_and_related_work.md`). 2. Decide on a prospective replication (Phase B-2). 3. Optional: multi-seed CGCNN; Task B tuning. 4. Regenerate figures (panel, tuning) in publication style. 5. Check that README/docs numbers agree with this draft.
