# Literature notes (7 PDFs in `research papers/`)

Extracted with `pdftotext` and keyword search. Tables or figures embedded as images were not read.
These seven papers are a reading list, not a literature review: they support statements about
themselves, not "nobody has done X".

## Per-paper summary

| # | Paper | Type | Data and models | Split | Layers? |
|---|---|---|---|---|---|
| 1 | Fronzi et al., arXiv 1911.11559, ML for 2D superlubricants | ML on bilayers | 282 interlayer-energy and 226 C33 DFT labels (VASP, PBE+vdW-TS) from 6,138 monolayers (2DMatPedia). Bayesian NN; 2,764 descriptors cut by GA+LASSO to 42 (energy) / 89 (C33). Extrapolates to 18.8M bilayers. | K-means cluster-stratified 75/25 | Yes: bilayers, heterobilayers |
| 2 | Venturi et al., arXiv 2003.13418, design principles for 2D materials | ML on monolayers | C2DB (~3,500 entries, Aug 2019; GPAW/PBE). 100-model CGCNN ensemble: Hform, band gap, log(c11), log(c22), c12. Screens ~20k perovskites, ~24k MXenes. | Random 70% per model; parity plots over all of C2DB (training points included) | Monolayers only |
| 3 | Malakar et al., ML for monolayer TMDs | ML on MD data | 1,440 LAMMPS tensile MD runs (Stillinger-Weber): MoS2, MoSe2, WS2, WSe2. FFNN for fracture stress/strain/E; LSTM for stress-strain curves. | train/val/test; ratio not stated in extracted text; LSTM had 14 test samples from the same four materials | Monolayers only |
| 4 | Elder et al., SISPAD, graphene/MoS2 layered MD | MD, no ML | Nanoindentation MD; mono/bi/trilayer and heterostructures | n/a | Yes |
| 5 | Davidovikj et al., arXiv 1704.05433 | Experiment, no ML | Duffing fit of nonlinear drum resonance | n/a | Few-layer graphene (5 nm), MoS2 drum |
| 6 | Maurizi et al., Sci Rep 2022 | ML on continuum FE | GNN for stress/strain fields in composites, lattices (~500 microstructures) | Random 90/10 | Layered composites, not atomistic 2D |
| 7 | Zhang & Cheung, book chapter | Review, no ML | Experimental summary tables | n/a | Yes: layer dependence discussed |

## Units

- **N/m (2D stiffness):** Venturi (C2DB c11, c22; MXene screening threshold 175 N/m) and Malakar (MD Young's modulus).
- **GPa / TPa (3D-equivalent):** Zhang's review table, Davidovikj (E), Elder (TPa = E2D / thickness h).
- **Fronzi:** interlayer energy in eV/A^2; C33 (out-of-plane) in GPa.
- Venturi uses C2DB c11 in N/m, which fixes the C2DB unit. The JARVIS unit remains an inference (it matches C2DB closely on shared formulas).
- **Layer-count trap:** N/m scales with the number of layers, GPa does not. Malakar quotes (citing others) h-MoS2 at 120-240 N/m (monolayer) and 190-330 N/m (bilayer); Elder gets 0.27 TPa (bilayer MoS2) vs 0.16 TPa (monolayer). Any monolayer-to-bilayer comparison must state per-layer vs total.

## Benchmark Young's modulus values

| Material | Value | Source | Method |
|---|---|---|---|
| Graphene | 1000 +/- 100 GPa (monolayer) | Zhang table | Indentation |
| Graphene | 594 +/- 45 GPa (5 nm few-layer) | Davidovikj | Nonlinear resonance |
| Graphene | 1.05 TPa mono, 1.06 TPa bilayer | Elder | MD |
| MoS2 | 270 +/- 100 GPa (1L), 200 +/- 60 GPa (2L); 330 +/- 70 GPa (5-25 L) | Zhang table | Indentation |
| MoS2 (CVD) | 260 +/- 18 (1L), 231 +/- 10 (2L) GPa | Zhang table | Indentation |
| MoS2 | 315 +/- 23 GPa (thick drum) | Davidovikj | Nonlinear resonance |
| MoS2 | 0.16 TPa mono, 0.27 TPa bilayer | Elder | MD |
| MoSe2 | 107.4 (armchair), 105.6 (zigzag) N/m; lit. 95-110 | Malakar | MD |
| WS2 | 126.6 / 123.8 N/m; lit. 115-125 | Malakar | MD |
| WSe2 | 130.0 / 126.1 N/m; lit. ~118-120 | Malakar | MD |
| WSe2 (experiment) | ~165-170 GPa, flat from 5 to 14 layers | Zhang table | Indentation |
| MXene threshold | c11, c22 >= 175 N/m | Venturi | CGCNN prediction |

Flags:
- Elder claims "excellent agreement" with ~0.25 TPa for MoS2, but its own monolayer value is 0.16 TPa (literature 0.27 +/- 0.1). Do not use that table entry as a benchmark.
- Fronzi's Table II lists C33 test MAE (16.04 GPa) larger than RMSE (9.98 GPa), which is mathematically impossible; treat that row as unreliable.

## Fronzi accuracy (closest prior work to Task B)

- Interlayer energy, test: R2 0.80, RMSE 0.055, MAE 0.035 eV/A^2.
- C33, test: R2 0.80 (RMSE/MAE inconsistent, above); validation R2 0.73.
- C33 comes from the curvature of the interlayer energy vs distance along z, so it is an out-of-plane constant, not in-plane stiffness.

## Multilayer coverage

- With ML: only Fronzi (out-of-plane C33 and interlayer energy).
- Without ML: Elder (MD), Zhang's review (E drops with layer number for MoS2, BP, h-BN, attributed to stacking faults and sliding; WSe2 stays flat).
- No: Venturi, Malakar, Davidovikj (few layers but not studied as a variable), Maurizi.

## What this project can legitimately claim

Supportable relative to these papers:
1. **Grouped (leave-family-out) evaluation.** Every ML paper here uses random splits (Venturi, Maurizi) or a split built to keep the test set in-domain (Fronzi's K-means; Malakar's test samples come from the same four materials). None measure generalisation to unseen families.
2. **A quantified random-vs-grouped gap on mechanical targets.**
3. **Stiffness learning with thousands of DFT labels (C2DB, ~7,300 clean) plus an independent cross-database check (JARVIS).**

Not supportable, or must be narrowed:
- "Bilayer mechanics is under-explored": Fronzi already did ML for bilayer interlayer energy and C33 at scale. The narrower claim (ML of in-plane bilayer stiffness / monolayer-to-bilayer change by stacking) is plausible but needs data we do not have (BiDB has no stiffness) and a wider literature search.
- "Multi-task is novel": Malakar predicts three outputs jointly (MD setting). A joint stiffness + binding-energy model needs BiDB or ML-potential labels.
- "First to use C2DB stiffness": Venturi did this in 2020.

Methodological carry-overs: report N/m with an explicit per-layer/total definition; log-transform stiffness targets (as Venturi did); include Fronzi's cluster-stratified split next to random and grouped; keep MD-derived values out of the benchmark set (classical-potential uncertainty is of the order of the quoted ranges).
