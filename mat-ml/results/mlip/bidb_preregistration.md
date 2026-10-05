# BiDB validation pre-registration (written BEFORE any MACE calculation on BiDB structures)

Date written: 2026-10-04. The earlier JARVIS-referenced criteria (`stage2_preregistration.md`) are unchanged and were evaluated separately.

## What BiDB is, and what was established from the data (no MACE result involved)
Source: Pakdel, Rasmussen, Taghizadeh, Kruse, Olsen, Thygesen, Nat. Commun. 15, 932 (2024) (full text could not be retrieved here; the
statements below marked [paper-summary] come from search-result summaries of it, the rest were verified on the files in `Dataset/`).
- 11,184 entries = 992 monolayers + 10,192 homobilayers (`number_of_layers`), 30 columns of properties; structures for all of them in `bidb_structures.json`.
- Stacking is encoded in the uid: `<monolayer_uid>-2-<R: a b c d>-[Iz-]<tx>_<ty>` = in-plane matrix (identity or 180-degree rotation), optional z-flip `Iz`, fractional
  in-plane translation. This is exactly the family of operations in `src/mlip/interlayer.py`.
- `distance` = VERTICAL GAP between the layers, min z(top layer) - max z(bottom layer): reproduced exactly (MAE 0.0000 A, 100% within 0.01 A) for all 10,192 bilayers
  (atoms 1..n = bottom layer, n+1..2n = top layer).
- The stored bilayers are RIGID: top layer = copy of the bottom layer (z-profile deviation 1e-8 A) and the in-plane cell equals the monolayer's in 100% of the cases.
  So `binding_energy_zscan` is an energy from a rigid vertical scan of C2DB monolayer geometries, and `distance` is the position of its minimum
  [paper-summary: z-scan with PBE-D3].
- Unit/sign: positive = bound; typical values 10-30 meV/A^2 (graphene AB 19.0) [paper-summary: "a few to ~100 meV/A^2"]. Unit meV/A^2.
- `binding_energy_gs`: definition NOT available from any source I can read; it is NOT used as a pass/fail reference (Spearman with zscan is only 0.56 on 9,860 rows).
- Reference quality: 116 negative gs values, 26 negative zscan values, 3 distances <= 0 A, 275 distances < 2 A, zscan up to 7970 meV/A^2.
  Paper-summary says distances 1.5-3.8 A; the CSV is evidently unfiltered.
- UID map (`bidb_to_c2db_uid_map.csv`): 982/992 monolayers mapped to C2DB one-to-one, formula-consistent; used only to attach C2DB labels, never for validation.

## Procedure (mirrors BiDB's own workflow so that the comparison is like-for-like)
For each selected bilayer: take the stored rigid geometry (C2DB monolayer geometry, NO relaxation), rebuild the cell with >= 20 A vacuum, scan the vertical gap g
of the top layer with MACE-MPA-0 + D3(BJ, PBE) (float64): coarse g in [-1.0, 6.0] A step 0.25, then fine step 0.02 A within +/-0.25 A of the coarse minimum.
    d_MACE = g at the minimum;   E_b = -(E_min - 2 E_mono)/A * 1000  [meV/A^2], E_mono = same calculator on the unrelaxed C2DB monolayer, same vacuum rule.
Per interface (a bilayer has one), not divided by layers, positive = bound.
A minimum at g >= 6.0 A (edge of the scan) counts as "unbound".
Comparison quantities: BiDB `distance` and `binding_energy_zscan` (reference-valid rows only; invalid rows are listed with counts and excluded from reference metrics).

## Set (fixed by seed; `bidb_validation_set.csv`, `bidb_selection.json`)
30 non-magnetic monolayers (natoms <= 6, >= 5 reference-valid stackings) drawn uniformly at random (seed 42) from 389 eligible; ALL their bilayers: 307 bilayers,
all reference-valid under the rule distance > 0, 0 < zscan <= 173.9 meV/A^2 (log-IQR, k = 3, computed on non-magnetic BiDB only). Nothing is removed after seeing results.

## Criteria (proposals fixed now; not scientific facts; not to be changed after seeing results)
  D1  interlayer gap: median |d_MACE - d_BiDB| <= 0.10 A  AND  90th percentile <= 0.30 A
  E1  Spearman(E_b MACE, E_b BiDB zscan) >= 0.80 over all reference-valid bilayers
  E2  median ratio E_b(MACE)/E_b(BiDB) in [0.80, 1.25];  >= 70% of bilayers with ratio in [0.70, 1.30];  >= 90% with ratio in [0.50, 2.00]
  E3  failures (exception, or unbound: MACE E_b < 2 meV/A^2 or minimum at the scan edge, when the reference is >= 5 meV/A^2) <= 10%
  S1  stacking dependence (monolayers with >= 4 stackings): mean within-monolayer Spearman >= 0.60  AND  the BiDB-best stacking is among MACE's 2 best in >= 60% of monolayers
Verdicts fixed in advance:
  "binding energy and interlayer distance reproduced"  iff  D1, E1, E2, E3 all hold.
  "stacking dependence reproduced"                      iff additionally S1 holds.
Reported without a verdict: MAE/RMSE/bias, Pearson, per-monolayer results, outliers and failure cases (all kept), the same statistics with the
reference-invalid rows included.
Limits stated now: one potential, 30 monolayers (a small, random sample); reference is DFT (PBE-D3) with its own errors, rigid layers at C2DB geometry.

## In-plane stiffness ratio (no pass/fail; descriptive, 60 bilayers = best and worst reference stacking of each of the 30 monolayers)
Both monolayer and bilayer are relaxed with MACE-MPA-0 + D3 (ions + in-plane cell, c fixed) and strained +/-1% with the validated protocol.
    R = Y2D_total(bilayer) / Y2D(monolayer)     (expected ~2 for weakly coupled layers; per-layer ratio R/2 ~ 1)
Reported: median, mean, standard deviation, IQR, 5-95% range, share within [1.8, 2.2], the deviation |R - 2|/2, the within-monolayer difference between
the best and worst stacking, whether the stacking was preserved during relaxation, and unstable/failed cases. Total and per-layer values are stored separately.
This is NOT the 100-200 bilayer pilot and gates nothing; it states the size of the interlayer effect relative to the monolayer calibration scatter
(Y2D MAE 9.9 N/m, ratio IQR 0.77-1.10).
