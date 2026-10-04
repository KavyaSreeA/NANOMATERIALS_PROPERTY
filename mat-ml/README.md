# mat-ml: ML for 2D-material stiffness under honest splits

Primary data: **C2DB** (GPAW/PBE, 8,462 monolayers with a stiffness tensor). Cross-check: **JARVIS-DFT 2D**
(VASP/OptB88vdW, 229 tensors). Secondary task (Task B): **BiDB** bilayer binding energy / interlayer distance.

## Install and run

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; use bin/activate on Linux/macOS
pip install numpy pandas scikit-learn scipy pyyaml pymatgen matminer lightgbm
cd mat-ml
python -m src.data --audit            # data audit -> results/data_audit.json
python -m src.train --task A          # full Task A (C2DB CV + JARVIS external check)
python -m src.train --task A --quick  # smoke test with fewer trees
python -m src.train --task B          # BiDB Task B (~1 h)
python -m src.robustness --seeds 42 43 44 45 46   # multi-seed + feature ablation + null baselines (~55 min)
```

Tested with Python 3.11, numpy 2.4.6, pandas 2.3.3, scikit-learn 1.9.1, pymatgen 2026.9.24,
matminer 0.10.1, lightgbm 4.7.0. Raw data stay in `../Dataset/` (path in `config.yaml`); nothing is copied.
The first run reads ~17k C2DB folders and caches to `cache/` (minutes); later runs are fast. `--rebuild` re-reads.

## Layout

```
config.yaml           paths, seed (42), cleaning rules, CV settings, target transforms
src/data.py           loaders + cleaning (C2DB, JARVIS, BiDB) + audit CLI
src/features.py       Magpie composition features + structural features
src/splits.py         random / cluster / chemsys / family (/ monolayer for Task B) + leakage stats
src/models.py         Ridge, RandomForest, LightGBM (sklearn GradientBoosting fallback)
src/train.py          CV loop, external check, results tables
results/              fold metrics, summaries, task_*_comparison.md
docs/literature_notes.md
```

Results are summarised in `docs/results_summary.md` (Task A), `docs/taskb_results.md` (Task B) and `docs/mlip_calibration.md` (ML-potential validation). `docs/methods_and_models.md` is a one-page manifest of everything trained, the features, splits, seeds and metric formulas; `docs/literature_comparison.md` compares with the papers in `research papers/`; `results/headline_metrics.md` has MAE, RMSE, R2, ln R2 for every configuration; `results/diagnostics/` has out-of-fold parity and residual plots; `models/` has the saved final models with model cards (in-sample numbers there are optimistic; use the cross-validated ones); tests run with `python -m pytest tests -q`.

## Conventions

**Units.** Stiffness is in N/m (2D elastic constants), never converted. C2DB's unit is documented
(GPAW 2D stress); JARVIS' is inferred and supported by agreement with C2DB on shared formulas.
BiDB binding-energy units are unconfirmed (probably meV/A^2) and are not converted.

**Per-layer vs total.** This matters for anything involving more than one layer:

| Quantity | Definition used | Layers |
|---|---|---|
| `Y2D` (C2DB, JARVIS) | stiffness of the whole slab as computed ("total") | `n_layers = 1` for every row |
| `Y2D_per_layer` | `Y2D / n_layers` | identical to `Y2D` today |
| BiDB `binding_energy_*` | energy per unit area of the **single interface** of a bilayer; **not** divided by layers | `n_layers = 2` |
| BiDB `distance` | interlayer distance, Angstrom | `n_layers = 2` |

No bilayer *stiffness* exists in any available file, so nothing here compares monolayer and bilayer
stiffness. If bilayer N/m labels are added later, store total and per-layer separately; N/m scales with
layer count, GPa does not.

**Targets.** `Y2D = (C11*C22 - C12^2)/C22`, `poisson = C12/C22`. Y2D (and BiDB targets) are fitted on ln scale;
poisson is not (it can be <= 0). Metrics are reported in original units and in fitting space.

**Stability is tensor-derived** (never the file's own label):
- C2DB: symmetrised 3x3 (xx, yy, xy) tensor positive definite; plus max|C-C^T|/max|C| <= 0.10.
- JARVIS: only the xx,yy block is used (C11>0, C22>0, det>0). The shear entries are not trusted: for MoS2 neither `[3,3]` nor `[5,5]` reproduces (C11-C12)/2.
  Corrupt tensors (NaN / denormal values such as 5e-324) are dropped.
- Poisson task only: rows with |nu| > 1 are dropped.

**Splits** (5-fold, seed 42): `random` (comparison only), `cluster` (K-means on Magpie features, then stratified; in-domain, as in
Fronzi et al.), `chemsys` (grouped by chemical system), `family` (grouped by anonymous formula + layer group).
`results/task_A_comparison.md` includes a leakage table: the fraction of test rows whose chemical system / formula / family also
occurs in training.

**Features** use composition and geometry only (Magpie 132 + a, b, gamma, area/atom, nat, thickness, n_layers, spg number, n_elements).
DFT outputs (ehull, hform, stability labels, gap) are excluded unless `features.use_dft_descriptors: true`.

## Task B status

BiDB (`bidb.db`, `bidb_properties.csv`, `bidb_structures.json`) and the uid map are in `Dataset/` (since 2026-10-04). Task B is implemented in `src/taskb.py`
(`python -m src.taskb`, also `python -m src.train --task B`): BiDB DFT binding energy (z-scan) and interlayer gap predicted from monolayer composition, geometry and
C2DB stiffness (via the uid map), with random / grouped-by-monolayer / grouped-by-family CV, per-fold spread, five seeds and a mean-predictor baseline.
Design: `results/taskB_design.md`; results: `docs/taskb_results.md`, `results/taskB_*`. The older stub `load_bidb` / `clean_bidb` in `src/data.py` is superseded and unused.
ML-potential work (`src/mlip/`, separate `.venv-mlip` environment) is closed; see `docs/mlip_calibration.md`.

## Tested vs untested

- Run and verified here: data loading/cleaning for C2DB and JARVIS, Magpie features, all four Task A split schemes, all three models,
  JARVIS external check, results writing.
- Task B (BiDB) was run in full; see `docs/taskb_results.md`. Not run: `use_dft_descriptors: true`; leave-one-group-out variants; graph neural networks.
- Known limits: JARVIS has only ~186 usable tensors; the 18 corrupt ones are dropped, not repaired; polymorphs are kept as distinct rows
  (no deduplication) and are the reason `family` grouping matters.
