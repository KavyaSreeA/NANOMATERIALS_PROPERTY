# Stage 2 pre-registration: interlayer validation (written BEFORE any interlayer MACE calculation)

Date written: 2026-10-02.

## What is available (checked)
- BiDB properties/structures (`bidb_properties.csv`, `bidb_structures.json`, `bidb.db`): **not in the project** (searched `Dataset/` and the whole project tree).
- `Dataset/bidb_to_c2db_uid_map.csv` IS present: 992 BiDB monolayer UIDs, 982 mapped one-to-one to C2DB UIDs (10 unmapped), all 982 exist in the C2DB
  tree and agree on reduced formula. Verification is formula-level only; its provenance is undocumented and it has not been checked against BiDB
  lattice constants or symmetry (BiDB structures are not available). It cannot be used for validation until BiDB data are found.
- `Dataset/youngs_modulus.csv`: bulk metals, unrelated to 2D layers. Not used.
- No BiDB-based validation is attempted and none is fabricated.

## Smallest defensible alternative actually available in the project
JARVIS `exfoliation_energy` (748 monolayers; OptB88vdW DFT; energy to remove a layer from the bulk). Converted to a per-interface binding energy per area:
    Eb_ref [meV/A^2] = exfoliation_energy [meV/atom] * nat / A.
Limitations (all stay in the report): meV/atom is a plausibility reading, not a documented definition; reference is bulk-derived and therefore includes
beyond-nearest-layer interactions; the bulk stacking is not stored; there is no reference interlayer distance and no reference for stacking dependence;
the functional (OptB88vdW) is itself uncertain against experiment, so agreement beyond the spread between vdW functionals is not meaningful.

## Set (fixed by seed, no MACE result used)
Pool: numeric exfoliation energy, nat <= 10, non-magnetic (|magmom_oszicar| < 0.05): 485 materials. Random sample N = 40, seed 42
(`interlayer_reference_set.csv`). Nothing is removed afterwards. Structures that cannot be built into a slab are reported as failures, not dropped silently.

## Calculation
MACE-MPA-0, float64, D3(BJ, PBE) dispersion via torch-dftd (primary; "dispersion on"); MACE-MPA-0 alone ("dispersion off") is run on the same set to document what
dispersion contributes. Monolayer relaxed with ions + in-plane cell (fmax 0.005). Homobilayers: 4 orientations x 16 in-plane shifts x 5 gaps single points;
3 lowest relaxed (ions only, monolayer cell fixed, fmax 0.01, <= 500 steps); the most strongly bound relaxed candidate is kept.
E_b = -(E_bilayer - 2 E_monolayer)/A, per interface (one interface per bilayer; NOT divided by layers), positive = bound, meV/A^2.
Self-check (stress vs energy derivative, dispersion calculator included) must give 0.97-1.03 or the run aborts.

## Regimes (defined from the REFERENCE value only)
- vdW regime  : Eb_ref <= 40 meV/A^2. Pass/fail criteria below apply here, because the planned bilayer pilot concerns weakly bound layers.
- strongly bonded: Eb_ref > 40 meV/A^2. Reported for completeness (a binding-energy model for van der Waals gaps is not expected to reproduce bonded slabs).
- Full set also reported.

## Criteria (proposals fixed now; not scientific facts; not to be changed after seeing results)
On the vdW-regime subset, with D3 on:
  C2  median ratio E_b(MACE)/E_b(ref) in [0.80, 1.25]
  C3  >= 70% of materials with ratio in [0.70, 1.30]  AND  >= 90% with ratio in [0.50, 2.00]
  C4  failures (exception, collapsed gap < 2.0 A, or unbound E_b < 2 meV/A^2) <= 10% of the subset
All three must hold for "binding energy magnitude reproduced". Failed structures count as failures (and are kept in the ratio statistics as
unbound/zero where a number exists).
Informational, not gating: Spearman and Pearson correlation (the reference range inside the vdW regime is narrow, so ranking is dominated by
reference noise), MAE/RMSE/bias in meV/A^2, outliers, and the interlayer distance (gap/d_int) plausibility, since no independent distance reference exists.
Stacking dependence: only the MACE-internal energy spread across stackings is reported; it is NOT validated.

## Verdict wording fixed in advance
Even if C2-C4 hold, the most that can be claimed is: "binding-energy magnitude of homobilayers is consistent with a bulk-derived OptB88vdW reference within
the stated tolerance for the vdW-regime sample". Interlayer distance, stacking order and bilayer in-plane stiffness remain unvalidated.
