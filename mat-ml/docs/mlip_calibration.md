# Universal ML interatomic potentials for 2D stiffness and interlayer binding: calibration and validation

Scope: MACE-MP-0, MACE-MPA-0 and CHGNet for monolayer in-plane stiffness; MACE-MPA-0 (+D3) for interlayer binding energy, interlayer gap and bilayer in-plane stiffness.
Status: **closed.** MLIP work stopped on 2026-10-04 after the D3-off run. No bilayer stiffness pilot was run and none is planned. Nothing here is a DFT validation of bilayer stiffness.
Every threshold below was fixed before the corresponding results existed (pre-registrations in `results/mlip/`), and none was changed afterwards. Failed, unstable and excluded cases are kept and listed.

## 1. Final verdicts

| Question | Pre-registered rule | Verdict |
|---|---|---|
| Monolayer stiffness gate (Y2D: Spearman >= 0.9 and MAE <= 12.93 N/m, every computed structure) | `calibration_gate.csv` | MACE-MPA-0: **pass on C2DB (n = 150), fail on JARVIS (n = 184) and fail combined (n = 334)**. MACE-MP-0 and CHGNet: fail everywhere. |
| Geometry contribution to the remaining MACE-MPA-0 error (fixed-C2DB-cell diagnostic) | `stage1_preregistration.md` | **No detectable geometry contribution (95% CI includes zero).** f_geom = 0.050, CI [-0.093, 0.176]. |
| Interlayer binding, JARVIS-referenced (40 materials, D3 on) | `stage2_preregistration.md` | C2 pass; **C3 fail, C4 fail** (weaker, bulk-derived reference). |
| Interlayer, BiDB (307 bilayers, 30 monolayers, rigid z-scan, D3 on) | `bidb_preregistration.md` | **Gate failed on D1 (interlayer gap).** E1, E2, E3 and S1 passed. |
| In-plane stiffness of homobilayers vs monolayers | descriptive, no gate | **Consistent with additive in-plane stiffness (median R 2.008, CI [1.95, 2.03]) in this model, small strains, 60 bilayers, not DFT-validated.** |
| Bilayer stiffness pilot | - | **Not run (decision).** |

## 2. Environment and provenance
Separate virtual environment `.venv-mlip` (Python 3.11.9); the Task A environment was not touched. torch 2.14.1+cu130, ase 3.29.0, mace-torch 0.3.16, chgnet 0.4.2, e3nn 0.4.4, torch-dftd 0.5.3,
numpy 2.4.6, pandas 3.0.6. Hardware: i7-14700HX, RTX 4050 laptop GPU (6.4 GB). Models: MACE-MP-0 (`medium`), MACE-MPA-0 (`medium-mpa-0`), CHGNet 0.3.0; MACE in float64; dispersion = D3(BJ, PBE) via
`TorchDFTD3Calculator` (MACE's own `mace_mp(dispersion=True)` wiring). Models under the Academic Software License (OMAT, MatPES checkpoints) were not used.
Every result record (`results/mlip/raw_*.jsonl`) stores the model tag, library versions, settings, relaxation outcomes, runtime and GPU memory; runs are append-only and resumable.
Numerical self-check run at the start of every run (run aborts outside 0.97-1.03): analytic stress vs finite-difference energy derivative, including shear and, for the D3 runs, the dispersion calculator.

### Problems found and fixed (all of mine, none hidden)
1. With the vacuum axis non-periodic, MACE's ASE stress was wrong by a constant factor (0.185 in a 34.9 A cell; ratio 1.0000002 with a periodic cell). Cells are now fully periodic with >= 15 A vacuum (20 A for BiDB).
2. Tilted-cell JARVIS slabs were rejected by my orientation routine; 8 of 10 now run. Two (Sr(BiO2)2, ZnWO4) could straddle the boundary and stay excluded for every model.
3. CHGNet (float32) failed the self-check at the shear component with a 1e-4 finite-difference step (ratio 1.067). At steps of 1e-3 and 3e-3 the ratio is 1.000 +/- 0.004 on four structures, so the step of the check was changed (tolerance unchanged). The stiffness calculations use analytic stresses and are unaffected.

## 3. Methods and conventions
**Stiffness protocol** (mirrors the C2DB files: +/-1% strains in xx, yy, xy). Orient the cell (a || x, c || z); relax ions and the in-plane cell (c fixed; fmax 0.005 eV/A) except in the fixed-cell variant;
strain +/-1% (xy: engineering shear gamma_xy, simple shear x -> x + gamma y); relax ions only; central differences of the analytic stress; 2D stiffness C [N/m] = dsigma/deps [eV/A^3] x L_z [A] x 16.0218, L_z = cell volume / in-plane area.
Symmetrised 3x3 (xx, yy, xy) tensor; stability = positive definite; Y2D = (C11 C22 - C12^2)/C22; nu = C12/C22. The xy constant is Voigt C66; C2DB's `c_33` appears to be about twice that (slope 0.45 versus 0.89 on the diagonal for MACE-MPA-0),
an inference from the numbers and not confirmed from a source, so shear is not used for any conclusion.
**Total vs per-layer.** Monolayers: `n_layers = 1`, total = per-layer. Bilayer stiffness is stored as total (Y2D_total) and per layer (Y2D_total / 2) in separate columns and never mixed. Binding energy is per interface (a bilayer has one) and is not divided by layers.
**Binding energy:** E_b [meV/A^2] = -(E_bilayer - 2 E_monolayer)/A x 1000, positive = bound. Gap = min z(top layer) - max z(bottom layer).

## 4. Monolayer calibration (336 structures: 150 C2DB + 186 JARVIS; family-uniform random sample, seed 42)
Reference: C2DB c_11, c_12, c_22; JARVIS tensor xx-yy block. Yardstick for the gate: the C2DB-vs-JARVIS disagreement on 106 shared materials (Y2D MAE 8.62 N/m; C11 8.77, C12 6.52, C22 9.32), so the gate MAE is 1.5 x 8.62 = 12.93 N/m.
Candidate benchmark of the plan: non-magnetic, ehull <= 0.1, nat <= 12 (3,380 monolayers, 395 families); 150 drawn by choosing 150 families at random and one member each (this skews soft: median Y2D 30.5 vs 44.9 N/m in the pool).

**Y2D, all computed structures (unstable tensors kept), N/m.**

| Model | Subset | n | MAE | Spearman | Slope vs ref | Median ratio | Gate |
|---|---|---|---|---|---|---|---|
| MACE-MPA-0 | C2DB | 150 | 9.9 | 0.939 | 0.886 | 0.905 | pass |
| MACE-MPA-0 | JARVIS | 184 | 77.5 | 0.813 | 0.106 | 0.921 | fail |
| MACE-MPA-0 | combined | 334 | 47.1 | 0.874 | 0.375 | 0.916 | fail |
| MACE-MP-0 | C2DB / JARVIS / combined | 150 / 184 / 334 | 19.3 / 35.4 / 28.1 | 0.827 / 0.763 / 0.820 | 0.66 / 0.60 / 0.62 | 0.69 / 0.73 / 0.72 | fail |
| CHGNet | C2DB / JARVIS / combined | 150 / 184 / 334 | 23.4 / 42.4 / 33.8 | 0.830 / 0.710 / 0.787 | 0.54 / 0.46 / 0.48 | 0.54 / 0.51 / 0.53 | fail |

The MACE-MPA-0 JARVIS MAE of 77.5 is dominated by a few ML-unstable tensors whose Y2D blows up when C22 -> 0 (FeO2: -10,968 N/m). With those removed (sensitivity, not the gate) MACE-MPA-0 gives MAE 9.6 / 16.2 / 13.3 N/m and Spearman 0.942 / 0.856 / 0.897 (C2DB / JARVIS / combined), which narrowly misses both thresholds on the combined set.
Components, MACE-MPA-0, MAE on C2DB / JARVIS: C11 11.1 / 17.2, C12 5.7 / 6.8, C22 10.8 / 15.5 N/m (database disagreement 8.8 / 6.5 / 9.3). Poisson ratio is the weakest quantity (MAE 0.18 on C2DB, Spearman 0.72); R2 is meaningless for it because of near-zero C22 outliers.
Instabilities (of 336): MACE-MPA-0 14 unstable tensors, 11 with asymmetry > 10%; MACE-MP-0 19 and 16; CHGNet 28 and 44. Lattice constant of the potential's own equilibrium: median error 0.54% (MPA-0), 0.50% (MP-0), 0.25% (CHGNet); 90th percentile 3.7-5.3%.
Median runtime 5.8-8.3 s per monolayer (three jobs sharing the GPU); peak GPU memory 150-224 MB.
Reference caution: JARVIS lists HgI2 (JVASP-20004) at 386 N/m, which is implausible for that soft material; it is the largest stable-structure error (the model gives 6 N/m). Part of the JARVIS error is reference noise.

## 5. Fixed-DFT-cell diagnostic (MACE-MPA-0, the same 150 C2DB structures)
The C2DB cell was kept exactly (verified against the original `structure.json`: max relative difference 4.4e-16, 0 flag failures out of 150); ions relaxed; identical strain protocol. The relaxed-cell results are the calibration records.

| Y2D, N/m | MAE | RMSE | R2 | Spearman | Slope | Bias |
|---|---|---|---|---|---|---|
| MACE-relaxed cell | 9.91 | 17.7 | 0.915 | 0.939 | 0.886 | -5.6 |
| Fixed C2DB cell | 9.42 | 15.5 | 0.935 | 0.936 | 0.878 | -6.2 |

Components (MAE, relaxed / fixed): C11 11.1 / 10.1, C12 5.7 / 5.5, C22 10.8 / 10.3. Poisson ratio (robust, |nu| <= 1, n = 149): MAE 0.171 / 0.130, median absolute error 0.064 / 0.075, Spearman 0.753 / 0.804.
Unstable tensors 9 / 9; asymmetry > 10%: 8 / 9; reference relaxation not converged 1 / 1; no exceptions.
**Result: no detectable geometry contribution.** f_geom = (MAE_relaxed - MAE_fixed)/MAE_relaxed = 0.050, 95% CI [-0.093, 0.176] (paired bootstrap over structures), which includes zero; the upper bound is below the 0.33 level that would indicate a geometry-dominated error.
The ~11% softening is unchanged at the DFT geometry (slope 0.886 -> 0.878). The pre-registered rule outputs label B (f_geom <= 0.10); the statement that the remaining error reflects the potential's strain response is an inference by exclusion, not something this experiment measures directly.
Individual structures change a lot (NbI2O 136.7 -> 78.8 N/m against a reference of 65.6; IrS2 30.8 -> 73.6 against 109.5; BeP2(HS)4 becomes stable) but gains and losses cancel (fixed cell improved 82 structures, worsened 68; median change -0.1 N/m).
Most of the error is material-specific scatter: ratio MACE/reference has IQR 0.77-1.10 (relaxed); dividing by the global slope leaves the relaxed MAE unchanged (9.91 -> 9.91).
Caveat: at the C2DB geometry MACE has a residual stress (median 0.9% of C11, 90th percentile 2.9%, 20% of structures above 2%), so fixed-cell values are stress-strain coefficients of a pre-stressed state.

## 6. Interlayer validation
### 6.1 Reference data
BiDB was not in the project until 2026-10-04. Before that no BiDB comparison was attempted and none was fabricated; an alternative with JARVIS exfoliation energies was run (6.2). `youngs_modulus.csv` (bulk metals) is unrelated and unused.
The UID map (`bidb_to_c2db_uid_map.csv`) maps 982 of 992 BiDB monolayers one-to-one to C2DB, all formula-consistent; verification is at formula level only and its provenance is undocumented.

### 6.2 JARVIS-referenced check (40 random non-magnetic monolayers with nat <= 10; seed 42)
**Unit and conversion of the reference.** JARVIS `exfoliation_energy` is E_2D - E_bulk per atom, i.e. **meV/atom** (this follows search-result summaries of the JARVIS papers, which give a 200 meV/atom exfoliability cutoff; I could not open the primary text, so this is documented only at that level; the data agree: graphene 70.4 meV/atom). Conversion to a per-interface binding energy:
`E_b,ref [meV/A^2] = exfoliation_energy [meV/atom] x nat / A [A^2]` (A = in-plane cell area; removing one layer from the bulk breaks one interface's worth of interaction). 1 meV/A^2 = 0.001 eV/A^2 = 0.01602 J/m^2.
**The 40 meV/A^2 regime threshold** = 0.040 eV/A^2 = 0.641 J/m^2. In per-atom terms it equals 40 x A/nat = 71-246 meV/atom (median 134) over JARVIS 2D, because the area per atom ranges from 1.8 to 6.1 A^2. It was chosen by me from the reference distribution (pool median 23, IQR 18.9-30.3 meV/A^2) before any MACE number existed; it is a proposal, not a standard.
The reference is bulk-derived (OptB88vdW), the bulk stacking is not stored, and there is no reference distance, so this check cannot validate distance or stacking.
**Results, D3 on, van der Waals regime (reference <= 40 meV/A^2, n = 37):** median ratio MACE/ref 1.09 [0.88, 1.27] (C2 pass); within +/-30% 45.9% (needed 70%) and within a factor 2 86.5% (needed 90%): C3 fail; failures 6/37 = 16.2% (needed <= 10%; five collapsed gaps < 2 A and one tilted-cell exception): C4 fail.
Spearman 0.50 [0.17, 0.78] (informational). The three bonded slabs (reference 46-138 meV/A^2) are under-bound (ratios 0.34-0.71), as expected for a dispersion correction.
**Dispersion off (same set):** median ratio 0.15 [0.09, 0.22]; 8% within +/-30%; 46% failures (17 of 37): C2, C3, C4 all fail. The D3 term contributes a median of 19.4 meV/A^2 (D3-on 22.4 vs D3-off 3.2 meV/A^2, reference 20.6), so MACE-MPA-0 binding without dispersion is not usable.

### 6.3 BiDB (PBE-D3 DFT) comparison
**What was established from the files** (full text of the BiDB paper could not be retrieved; statements marked [summary] come from search summaries of Pakdel et al., Nat. Commun. 15, 932, 2024):
`distance` is the vertical gap between the layers, min z(top) - max z(bottom): reproduced exactly (MAE 0.0000 A, 100% within 0.01 A) for all 10,192 bilayers. The stored bilayers are rigid copies of the C2DB monolayer (top layer identical to the bottom layer to 1e-8 A, in-plane cell identical in 100%),
so `binding_energy_zscan` is an energy from a rigid vertical scan [summary: PBE-D3] and `distance` its minimum. Unit meV/A^2 and positive = bound [summary: "a few to ~100 meV/A^2"; graphene AB 19.0]. The stacking is encoded in the uid (rotation matrix, optional z-flip `Iz`, fractional shift).
`binding_energy_gs` has no available definition and was not used as a reference (its Spearman with the z-scan column is 0.904 here, informational).
**Procedure** (like-for-like): for each stored rigid geometry, scan the top layer's vertical gap with MACE-MPA-0 + D3 (coarse -1 to 6 A in 0.25 A steps, then 0.02 A steps around the minimum), E_mono on the unrelaxed C2DB monolayer, >= 20 A vacuum.
**Set:** 30 non-magnetic monolayers (nat <= 6, >= 5 valid stackings) drawn at random (seed 42) from 389 eligible; all 307 of their bilayers; all passed the reference-validity rule; no exceptions. Intervals: cluster bootstrap over monolayers.

| Criterion (fixed beforehand) | Requirement | Result | |
|---|---|---|---|
| D1 gap | median |dd| <= 0.10 A and p90 <= 0.30 A | median 0.175 A [0.126, 0.275]; p90 0.479 A [0.320, 0.565] | **FAIL** |
| E1 | Spearman >= 0.80 | 0.903 [0.788, 0.950] | pass |
| E2 | median ratio in [0.8, 1.25]; >= 70% within +/-30%; >= 90% within x2 | 1.097 [1.037, 1.208]; 72.6%; 98.4% | pass |
| E3 | failures <= 10% | 2 of 307 (0.7%) | pass |
| S1 | mean within-monolayer Spearman >= 0.60 and BiDB-best stacking in MACE's top 2 for >= 60% of monolayers | 0.749 [0.665, 0.847]; 70% | pass |

Pre-registered verdicts: "binding energy and interlayer distance reproduced" = **not satisfied** (D1); "stacking dependence reproduced" = **not satisfied** (it requires D1), although S1 itself passes.
Details: gap bias -0.0002 A, MAE 0.237 A, RMSE 0.353 A, 29% within 0.1 A, 72% within 0.3 A, Spearman of the gap 0.74. Binding energy: MAE 3.6, RMSE 4.8 meV/A^2, bias +1.6 (MACE over-binds by ~10%; median ratio 1.23 for E_b < 10, 1.05-1.15 for 10-30, 0.93 above 30 meV/A^2).
Worst gap cases: four C2H2 (C4H4) bilayers (BiDB gap ~2.0 A and E_b 17-19 meV/A^2; MACE gap ~4.1 A and E_b ~2 meV/A^2; two of them are the 2 failures). Weak stacking cases: PbSe2 (within-monolayer Spearman -0.70), C2H2 (0.00), Zr2S2Te2 (0.49), Pd4Te4 (0.46); in 9 of 30 monolayers BiDB's best stacking is not in MACE's top two.
Median stacking spread: BiDB 6.8, MACE 8.2 meV/A^2; median error of the stacking-resolved differences 1.3 meV/A^2.
Reading: binding energy (ranking and magnitude, with a small over-binding) and the ordering of stackings are reproduced to the stated tolerances; the interlayer gap is not (median error ~0.18 A). I did not test whether a shallow energy well explains the gap scatter; the by-range pattern does not show it cleanly.
Limits: one potential; 30 monolayers; BiDB's D3 variant may differ from D3(BJ); rigid C2DB geometries.

## 7. In-plane stiffness of homobilayers vs monolayers (MACE-MPA-0 + D3; descriptive)
60 bilayers = the best and worst reference stacking of each of the 30 BiDB monolayers; both layers relaxed (ions + in-plane cell) with the same calculator and strained +/-1%. R = Y2D_total(bilayer) / Y2D(monolayer); per-layer ratio = R/2.
**The result is consistent with additive in-plane stiffness (median R 2.008, CI [1.95, 2.03]) in this model, small strains, 60 bilayers, not DFT-validated.**

| | All computed (n = 60) | Strict: both tensors stable, relaxations converged, stacking preserved (n = 51) |
|---|---|---|
| Median R | 2.008 | 2.009 |
| 95% CI of the median (cluster bootstrap over monolayers) | [1.95, 2.03] | [1.95, 2.03] |
| IQR | 1.92-2.07 | 1.93-2.08 |
| 5-95% range | 1.68-2.28 | 1.79-2.29 |
| Mean +/- sd | 1.94 +/- 0.60 | 2.04 +/- 0.31 |
| Share within [1.8, 2.2] | 82% | 82% |
| Median |R - 2|/2 | 3.8% | 3.5% |

C11 and C22 ratios: medians 1.99 and 1.95. Spread across stackings of one monolayer: median |R_best - R_worst| = 0.085 (about 4 N/m of total Y2D; 24 monolayers). R is only weakly related to the reference binding energy (Spearman 0.14-0.18) and not to the relaxed gap (0.01-0.05).
Kept cases: SnSe (Se2Sn2) R = 3.3-3.4; BaBr2 1.42; BaI2 1.70; PbSe2 negative (unstable bilayer tensor). Flags: 6 unstable bilayer tensors, 4 unstable monolayer tensors, 3 stackings changed on relaxation, 0 unconverged.
Not established: whether errors cancel in the ratio (it is a ratio of two MACE calculations), any DFT confirmation of bilayer stiffness, behaviour at larger strains, magnetic or metallic layers.

## 8. The six questions
1. **Is MACE-MPA-0 reliable for monolayer mechanics?** Partly. On stable C2DB-like monolayers it ranks well (Spearman 0.94) and is within about 10 N/m MAE with a systematic ~10% softening that is not a lattice effect, and it passes the pre-registered gate there. It fails the gate on the full benchmark (JARVIS and combined), mainly through a few unstable-tensor outliers and weaker JARVIS agreement, and sits just outside the thresholds even when unstable tensors are removed. MACE-MP-0 and CHGNet are 30-45% too soft and are not reliable.
2. **Is its interlayer physics reliable?** Partly. Against BiDB, binding-energy ranking (Spearman 0.90), magnitude (median ratio 1.10) and stacking ordering pass the pre-registered criteria; the interlayer gap fails (median error 0.18 A against 0.10 A allowed). The weaker JARVIS-referenced check fails. Dispersion is essential (D3 supplies a median 19 meV/A^2).
3. **Main remaining uncertainties.** (a) Material-specific scatter in monolayer stiffness (ratio IQR 0.77-1.10) on top of the ~10% softening; (b) weaker agreement on JARVIS, partly reference noise; (c) interlayer gap error; (d) whether errors cancel in bilayer-minus-monolayer quantities; (e) no DFT validation of any bilayer stiffness; (f) small samples (150 + 186 monolayers, 30 monolayers for BiDB) and a single passing potential; (g) magnetic, metallic and strongly bonded layers excluded or poorly handled; (h) the `gs` definition and BiDB's exact D3 variant are unknown.
4. **Is a 100-200 homobilayer pilot defensible?** Not pursued, by decision. The evidence so far would not support claims of stacking-dependent stiffness: the stacking-dependent variation is small (about 4 N/m) and its significance against the model uncertainty could not be established without a second potential or DFT reference.
5. **Pilot protocol:** none defined, since no pilot will be run.
6. **What would be required before any such claim:** a DFT reference for in-plane stiffness on a few bilayers (including SnSe and BaBr2), an independent second potential to test cancellation in R, an explanation of the gap error, a larger and cleaner monolayer calibration set, and treatment of magnetic and metallic layers.

## 9. Files
`results/mlip/`: `raw_*.jsonl` (all per-structure records), `calibration_{metrics,gate,tables}`, `stage1_*`, `stage2_jarvis_disp{on,off}_*`, `bidb_{summary.json,per_bilayer.csv,per_monolayer.csv,stacking_per_monolayer.csv,stiffness_*}`, `consistency_*.json`, pre-registrations
(`stage1_preregistration.md`, `stage2_preregistration.md`, `bidb_preregistration.md`), selection files, and `plots/`. Code: `src/mlip/` (`stiffness.py`, `calculators.py`, `run_calibration.py`, `report_calibration.py`, `stage1_fixed_cell.py`, `interlayer*.py`, `bidb_*.py`).
Commands: `python -m src.mlip.run_calibration --model <name> --variant relaxed_cell|dft_cell [--source c2db]`; `python -m src.mlip.report_calibration`; `python -m src.mlip.stage1_fixed_cell`; `python -m src.mlip.interlayer_select`, `interlayer --dispersion on|off`, `interlayer_analysis`;
`python -m src.mlip.bidb_select`, `bidb_zscan`, `bidb_analysis`, `bidb_stiffness_ratio`, `bidb_stiffness_analysis` (all in the `.venv-mlip` environment).
