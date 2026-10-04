# Stage 1 pre-registration: fixed-DFT-cell diagnostic (written BEFORE the full 150-structure run)

Date written: 2026-10-02. Written after a 4-structure smoke test (smoke results are part of the final 150 and are not used to set any rule below).

## Question
Is the remaining MACE-MPA-0 error on monolayer stiffness mainly
  A. an incorrect equilibrium lattice/geometry, or
  B. an incorrect energy/strain response (curvature of the potential-energy surface) of the potential itself?

## Design
Same 150 C2DB monolayers as the calibration (`source == c2db` in `benchmark_inputs.json`), MACE-MPA-0 only, same protocol
(+/-1% strains in xx, yy, xy, ions relaxed at each strained cell, fmax 0.005 eV/A, float64, 15 A minimum vacuum).
  - relaxed_cell : existing calibration results (ions + in-plane cell relaxed with MACE-MPA-0 first). Not recomputed.
  - dft_cell     : the C2DB cell is kept exactly (verified per structure); ions are relaxed; then the same strains.
Reference: C2DB c_11, c_12, c_22. Y2D = (C11*C22 - C12^2)/C22, poisson = C12/C22.
Known limitation of the fixed-cell variant: at the C2DB geometry MACE has a residual in-plane stress (recorded per structure as
`pre_stress_xx_yy_xy_N_m`). Central differences remove the constant offset, but the stress-strain coefficients of a pre-stressed
state differ from equilibrium elastic constants at second order in the pre-stress. Interpretation is cautioned where
|pre-stress| / C11 exceeds 2%.

## Fixed rules (not to be changed after seeing results)
1. Metrics: MAE, RMSE, R2, Spearman, slope through the origin (pred ~ k*ref), bias, median ratio, for C11, C12, C22, Y2D, poisson.
   Poisson is reported with robust metrics (MAE, median absolute error, Spearman, bias on |nu|<=1 subset); R2 is not used for it.
2. Basis: "all computed" (ML-unstable tensors kept, exactly as in the calibration gate); "ML-stable only" is reported as a sensitivity.
3. The previously defined gate is unchanged: Spearman >= 0.9 AND Y2D MAE <= 12.93 N/m (1.5 x 8.62, the C2DB-vs-JARVIS disagreement).
   It is evaluated for both variants on these 150 structures for information; the original gate verdicts (C2DB pass, JARVIS fail,
   combined fail for MACE-MPA-0) are NOT revised by Stage 1.
4. Attribution statistic (paired over structures, Y2D, "all computed"):
       f_geom = (MAE_relaxed - MAE_fixed) / MAE_relaxed,  95% bootstrap CI (2000 resamples over structures, seed 0).
   Verdict:
     A (geometry-dominated)  if f_geom >= 0.33 AND CI lower bound > 0
     B (response-dominated)  if f_geom <= 0.10
     Mixed                   otherwise
   A negative f_geom (fixed cell worse) is reported as such.
   The same statistic is also given for C11, C12, C22 as supporting evidence, and the slope shift (slope_fixed - slope_relaxed) is reported.
5. Supporting analysis (descriptive, no verdict): correlation of the per-structure change Y2D_fixed - Y2D_relaxed with the lattice error
   (a_MACE - a_C2DB)/a_C2DB, and with the pre-stress.
6. Nothing is removed. Failed, non-converged, unstable, and asymmetric (>10%) results are listed with reasons.

## Provenance
raw_mace_mpa0_medium_dft_cell.jsonl (append-only, resumable, one record per structure: model tag, library versions, settings,
reference relaxation info, strains, runtime, GPU memory, cell_fixed_ok flag, pre-stress).

## Addendum, 2026-10-04 (wording only; the rule, thresholds and numbers above are unchanged)
At the author's request the verdict is reported as "no detectable geometry contribution (95% CI includes zero)" instead of "response-dominated". The
pre-registered rule still outputs label B (f_geom <= 0.10) with f_geom = 0.050, 95% CI [-0.093, 0.176]. The causal reading "the remaining error comes from the
potential's strain response" is an inference by exclusion, not something this experiment measures directly.
