# Phase B-2: prospective replication of the property panel on JARVIS-DFT 2D (pre-registration)

Written 2026-10-05 BEFORE any model was fitted on JARVIS panel targets. Nothing below may be changed after the results are seen; deviations go in a dated addendum.

## Why
All earlier results come from one database (C2DB, GPAW/PBE). JARVIS-2D (VASP/OptB88vdW, 1,103 monolayers) differs in code, functional and construction. A replication with predictions written first tests whether the family penalty and the failed diagnostic are properties of C2DB or of the data-splitting problem.

## Data and groups
`Dataset/jarvis_dft_2d.json`, all 1,103 monolayers (no stiffness requirement). Features: the same 141 composition+geometry features as Task A (`build_features`). Groups: chemical system as before; **family = anonymous formula + space-group number** (JARVIS has no layer group; this is a documented deviation and is coarser/finer in different cases than C2DB's family).

## Target selection (rules fixed now)
Candidates: every numeric key with >= 500 non-null finite labels. Excluded by rule: identifiers and settings (kpoint_length_unit, encut, maxdiff_*), purely geometric or trivially derived (density, nat), extensive (optb88vdw_total_energy), constant (ehull), degenerate (> 50% at the modal value, e.g. likely magmom). One representative per property family; where a family has an n- and a p- version the n- version is used; epsx for the dielectric family; avg_elec_mass for effective mass. Log transform if all > 0 and max/min > 100. Expected panel (not guaranteed): formation_energy_peratom, optb88vdw_bandgap, exfoliation_energy, epsx, spillage, avg_elec_mass, n-Seebeck, n-powerfact, ncond, nkappa (up to ~10 targets).

## Protocol
Identical to the C2DB panel: LightGBM Task-A settings, random / chemsys / family 5-fold StratifiedGroupKFold on 10 quantile bins, seeds 42-44, global-mean / chemsys-mean / family-mean baselines, same skill, retention, D_family, ICC and cluster-bootstrap definitions. Out-of-fold predictions saved. Code: `src/panel_jarvis.py` reuses `panel.run_one` and `panel_analysis`.

## Predictions (written now)
- **P1.** MAE ratio family/random > 1.05 for at least 80% of the JARVIS targets, with the bootstrap CI above 1 for at least 70%. (Expected because the C2DB result was 12 of 12.)
- **P2.** chemsys/random < 1.15 for at least 70% of targets (chemistry grouping adds little). Weaker confidence than P1: JARVIS has more singleton chemistries, which may raise it.
- **P3.** H2 replicates as a failure: Spearman(D_family, R_family) on JARVIS targets will not be <= -0.6 with a CI wholly below 0. We expect |rho| < 0.5.
- **P4 (pooled, secondary).** Pooling the 12 C2DB and the JARVIS targets (about 22), the same criterion (rho <= -0.6, CI below 0) is also not met. If it IS met in the pooled data, this is reported as a possible signal needing a larger panel, not as a confirmed forecaster.
- **P5.** Where a property is in both databases (formation energy, PBE vs OptB88vdW band gap), the direction of the penalty agrees (both > 1.05).

## Decision rules
- If P1 fails (ratio <= 1.05 for more than 20% of targets), the abstract and title are softened from "general" to "in C2DB".
- If P3 and P4 hold, the diagnostic is reported as a replicated negative result. If P4 fails (pooled test passes), report it as an exploratory signal only, because the criterion was fixed for a single panel and the pooled test was not the primary one.
- Failed predictions are reported in the paper alongside the passed ones.
