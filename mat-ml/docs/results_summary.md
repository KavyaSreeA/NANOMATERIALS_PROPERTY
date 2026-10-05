# Results summary (Task A: stiffness of 2D monolayers)

All numbers come from runs of the code in this repository (Python 3.11, scikit-learn 1.9.1, LightGBM 4.7.0). MAE is in N/m
unless stated; `R2(ln)` is R2 of ln(Y2D). Raw tables are in `results/`. **Task B (BiDB) has not been run**: the BiDB files are not in
the project folder.

## 1. Data used

| Item | Count |
|---|---|
| C2DB materials | 16,905 |
| with a stiffness file | 8,462 (8,430 in the new layout + 32 older flat files that the Step 1 audit had skipped) |
| tensor positive definite | 7,474 |
| max tensor asymmetry <= 10% | 7,258 |
| final Y2D set (Y2D > 0) | **7,258** (3,806 chemical systems, 2,572 of them with a single member; 676 structure families, largest 432 rows) |
| Poisson set (\|nu\| <= 1) | 7,229 |
| JARVIS tensors | 229, of which 18 corrupt (NaN / denormals) -> 211; xx,yy block stable -> **186** |

Targets: `Y2D = (C11*C22 - C12^2)/C22` (N/m, ln-fitted) and `poisson = C12/C22`. Every row is a single layer (`n_layers = 1`), so total and
per-layer stiffness are identical. Features: Magpie composition (132) + geometry (a, b, gamma, area/atom, atom count, thickness,
`n_layers`, `spg_number`) + `n_elements` = 141; DFT outputs are excluded. Full detail is in the README.

## 2. Baselines (seed 42, 5-fold, full-size models)

Y2D MAE (N/m) / R2:

| Model | random | cluster | chemsys | family |
|---|---|---|---|---|
| Ridge | 22.5 / 0.69 | 22.6 / 0.69 | 22.8 / 0.67 | 31.1 / -11.0 (artefact, see below) |
| Random forest | 14.7 / 0.85 | 14.7 / 0.84 | 14.8 / 0.83 | 20.9 / 0.67 |
| **LightGBM** | **13.7 / 0.87** | 14.0 / 0.87 | 13.9 / 0.86 | **19.8 / 0.70** |

Poisson MAE / R2: LightGBM 0.111 / 0.39 (random), 0.109 / 0.42 (chemsys), 0.140 / 0.14 (family); random forest is within 0.004 MAE of it;
Ridge is 0.144 / 0.11 (random) and 0.152 / 0.03 (family).

The Ridge family-scheme R2 of -11 comes from one fold whose ln-space extrapolation blew up after exp() (R2 -57.5 in N/m; the same fold has
R2(ln) 0.36, five-fold mean 0.55). It is an artefact of the back-transform, not a data error.

Leakage (fraction of test rows whose key also occurs in training):

| Scheme | chemical system | formula | structure family |
|---|---|---|---|
| random | 0.61 | 0.36 | 0.96 |
| cluster | 0.61 | 0.36 | 0.96 |
| chemsys | 0.00 | 0.00 | 0.96 |
| family | 0.59 | 0.34 | 0.00 |

## 3. Multi-seed robustness and null baselines (seeds 42-46)

| Model / features | random MAE | chemsys MAE | family MAE | family/random | chemsys/random |
|---|---|---|---|---|---|
| global mean | 42.4 | 42.4 | 42.4 | 1.00 | 1.00 |
| family mean (no features) | 27.0 | 27.0 | 42.4 | 1.57 | 1.00 |
| chemsys mean (no features) | 34.5 | 42.4 | 34.3 | 0.99 | 1.23 |
| LightGBM, composition only | 19.2 | 18.6 | 25.5 | 1.33 | 0.97 |
| LightGBM, structure only | 19.8 | 20.0 | 26.6 | 1.35 | 1.01 |
| **LightGBM, all** | **13.8 +/- 0.1** | **13.9 +/- 0.1** | **19.9 +/- 0.2** | **1.44 +/- 0.02** | **1.00 +/- 0.01** |
| random forest, all | 14.6 +/- 0.1 | 14.7 +/- 0.1 | 20.8 +/- 0.1 | 1.43 +/- 0.01 | 1.01 +/- 0.00 |

Seed std is across the five seeds and reflects split and model randomness only; the data are fixed, so it understates the uncertainty from the
sample size.

Findings:
1. **The family gap is robust to seeds and to resampling families:** 1.42-1.44x MAE in every seed for both strong models; a cluster bootstrap over structure families (`phaseC_cluster_bootstrap.json`) gives 1.44 with 95% CI 1.30-1.62, and 1.01 [0.99, 1.04] for chemical-system grouping. The seed standard deviations alone understate the uncertainty.
2. **The chemical-system gap is zero:** removing the 61% of test rows that share a chemical system with training does not hurt (ratio 1.00-1.01; composition-only 0.97).
   Random-split inflation in this data comes from structure-family overlap, not from chemistry.
3. **Both feature groups matter:** composition-only and structure-only each reach about 19-20 N/m; together 13.8 (about 28% lower).
4. **Prototype is only part of the story.** A family-mean baseline explains R2(ln) 0.46 under random CV and 0 for unseen families, but composition-only
   LightGBM still reaches 0.50 on unseen families.

## 4. `spg_number` ablation (LightGBM, seeds 42-46)

| Features | random MAE | family MAE | family/random |
|---|---|---|---|
| `spg_number` only | 38.0 | 44.3 | 1.16 |
| geometry without `spg_number` | 20.9 | 26.3 | 1.26 |
| structure (geometry + `spg_number`) | 19.8 | 26.6 | 1.35 |
| composition + `spg_number` | 16.9 | 24.4 | 1.44 |
| composition only | 19.2 | 25.5 | 1.33 |
| all, without `spg_number` | 14.1 | 20.2 | 1.44 |
| all | 13.8 | 19.9 | 1.44 |

- `spg_number` alone is nearly uninformative (MAE 38.0 vs 42.4 for the global mean under random CV, and worse than the global mean under family CV).
- The structure-only score comes from geometry, not symmetry: removing `spg_number` costs 1.1 N/m under random CV and none under family CV.
- The full model barely depends on it (0.3 N/m). The family/random gap stays 1.44 either way, so the gap is not an artefact of this feature.
- Limit: this tests the coarse space-group number only. The finer prototype descriptors (layer group, anonymous formula) are tested in section 4b.

## 4b. Layer group and anonymous formula as features (LightGBM, seeds 42-46)

New columns (`prototype_features`): layer-group number, one-hot layer group (53 columns), one-hot anonymous formula (59 columns); categories with
fewer than 10 rows are collapsed to "other". Category lists use row counts only (no labels). Not available for JARVIS, so not part of the external check.
Note the `family` split holds out the (anonymous formula, layer group) **pair**; each marginal still appears elsewhere in training.

| Features | random MAE | family MAE | R2(ln) random | R2(ln) family | family/random |
|---|---|---|---|---|---|
| prototype columns only | 28.3 | 40.8 | 0.46 | 0.06 | 1.44 |
| composition only (reference) | 19.2 | 25.5 | 0.65 | 0.50 | 1.33 |
| composition + prototype | 16.3 | 23.9 | 0.73 | 0.57 | 1.47 |
| composition + `spg_number` (reference) | 16.9 | 24.4 | 0.72 | 0.55 | 1.44 |
| all (reference) | 13.80 | 19.90 | 0.80 | 0.68 | 1.44 |
| all + layer group | 13.63 | 19.75 | 0.81 | 0.69 | 1.45 |
| all + anonymous formula | 13.76 | 20.03 | 0.80 | 0.67 | 1.46 |
| all + both | 13.63 | 19.69 | 0.81 | 0.69 | 1.45 |

Seed std is 0.03-0.26 N/m.

- Under random CV the prototype columns alone reach R2(ln) 0.455, the same as the family-mean baseline (0.463): the model recovers the family identity from the two
  one-hots. Under family CV they reach 0.06, essentially nothing: layer group and anonymous formula individually carry almost no stiffness information, and the
  signal sits in the specific combination, which an unseen family does not provide.
- Adding them to the full feature set changes MAE by -0.2 to +0.1 N/m (at most 1.3%), which is within about 1-3 seed standard deviations. Composition, geometry and the other columns already encode most of this
  information (atom count, thickness, composition).
- They help composition-only features (-3.0 N/m random, -1.6 family), about as much as `spg_number` or the geometry columns do.
- The family/random gap stays at 1.44-1.47x. It is not closable by adding prototype descriptors, so it reflects genuine extrapolation to new prototypes.
- Limits: single model (LightGBM); one-hot trees can rebuild family identity, which is why random CV looks good; improvements of 0.2 N/m are not distinguishable from noise.

## 5. JARVIS cross-database check (train on all C2DB, predict JARVIS)

LightGBM, all features, 95% bootstrap intervals over JARVIS rows:

| JARVIS subset | n | MAE | R2(ln) | Spearman |
|---|---|---|---|---|
| all | 186 | 19.0 [14.4, 24.5] | 0.55 [0.34, 0.73] | 0.82 |
| matched to a C2DB material (leaky) | 106 | 10.2 [7.9, 12.5] | 0.81 [0.67, 0.93] | 0.93 |
| chemical system unseen in C2DB | 55 | 29.1 [20.5, 39.6] | 0.45 [-0.14, 0.75] | 0.73 |

- **Label agreement** between C2DB and JARVIS on the 106 matched materials: MAE 8.6 N/m, R2 0.90, R2(ln) 0.75, Spearman 0.92, median ratio 1.04.
  This is a noise floor for any cross-database evaluation (an upper bound on label disagreement, because matching used formula, atom count and lattice
  constants but not symmetry). The ratio near 1 supports reading JARVIS values as N/m, though that is not proven.
- On unseen chemistry the model error is about 3.4x the noise floor, so most of it is genuine model error. The noise floor was measured on matched
  (possibly easier) materials.
- Model seeds change the MAE by only 0.4-1.1 N/m; sampling uncertainty (the 186-row JARVIS set) dominates.
- On unseen chemistry composition-only collapses (R2(ln) 0.06) while structure-only holds (0.41); intervals overlap, so this is suggestive only.
- The "all" row mixes leaky and non-leaky materials and should not be quoted as a generalisation figure.

## 6. What the evidence supports

Supported by these runs:
- Random splits overestimate stiffness-prediction accuracy on C2DB when structure family is held out: MAE is 1.4x higher and R2 falls from 0.87 to 0.70.
- Holding out chemical systems does not produce a gap (1.00x), so family-level grouping is the meaningful stress test on this dataset.
- Cross-database performance on unseen chemistry is clearly worse than on matched materials, with wide uncertainty.
- Geometry features, not symmetry class, carry the structure signal; layer group and anonymous formula add at most about 1% to the full model.

Not supported / not tested:
- Anything about bilayers or multilayers: no bilayer stiffness exists in any available file, and BiDB is missing. Task B code is untested on real data.
- Multi-task prediction (stiffness + binding energy): not attempted.
- "Prototype alone drives performance": only partly supported (section 3, point 4).
- Individual feature importance: groups of features were ablated, not single columns.

## 7. Limits and known issues

- Single dataset for training (C2DB, GPAW/PBE) and a small external set (186 rows, VASP/OptB88vdW); JARVIS units are inferred, not documented.
- Polymorphs are kept as separate rows (no deduplication), and family sizes are very uneven (up to 432 rows), which makes family folds uneven.
- C2DB `spg_number` (`number`) and JARVIS `spg_number` may not mean the same thing; this affects only the JARVIS check.
- The JARVIS-C2DB matching is a heuristic (formula, atom count, a/b within 3%).
- Hyperparameters were not tuned; all models use fixed settings from `config.yaml` / `src/models.py`.

## 8. Files

| File | Content |
|---|---|
| `results/task_A_*` | seed-42 baselines: fold metrics, summary, comparison table, run info |
| `results/robustness_*` | five-seed runs, feature ablation, null baselines, paired ratios |
| `results/spg_ablation_*` | `spg_number` ablation |
| `results/proto_features_*` | layer-group / anonymous-formula features |
| `results/external_*` | JARVIS multi-seed check and label-agreement table |
| `results/data_audit.json` | cleaning counts |
| `results/_quick_task_A_comparison.md` | smoke test with reduced trees (not a result) |
| `docs/literature_notes.md` | extraction from the seven PDFs |

Commands: `python -m src.train --task A`, `python -m src.robustness --seeds 42 43 44 45 46`,
`python -m src.robustness ... --sets geometry_no_spg spg_only all_no_spg composition_plus_spg --prefix spg_ablation`,
`python -m src.robustness ... --prototype --sets all_plus_layergroup all_plus_anon all_plus_prototype composition_plus_prototype prototype_only --prefix proto_features`, `python -m src.external --seeds 42 43 44 45 46`.

## 9. Suggested next steps

1. Locate BiDB (and the uid map) to run Task B.
2. Add bilayer stiffness labels (e.g. from a universal ML potential validated against C2DB/JARVIS) before any bilayer or multi-task claim.
