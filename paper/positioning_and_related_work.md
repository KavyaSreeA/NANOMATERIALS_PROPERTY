# Positioning and related work (working notes for the manuscript)

Status: written 2026-10-04 before the Phase B and Phase A results are known. Bibliographic details come from search results and the PDFs in `research papers/`; **verify authors, volumes and DOIs before submission**.

## 1. What is already known (do not claim)
| Claim | Where it is established |
|---|---|
| Random cross-validation overestimates materials-ML performance; extrapolation to new clusters is harder | Meredig et al., "Can machine learning identify the next high-temperature superconductor? Examining extrapolation performance for materials discovery", Mol. Syst. Des. Eng. 2018 (leave-one-cluster-out CV) |
| Materials datasets are highly redundant; redundancy inflates random-split scores | "Exploiting redundancy in large materials datasets for efficient machine learning with less data", Nat. Commun. 2023; "MD-HIT: Machine learning for material property prediction with dataset redundancy control", npj Comput. Mater. 2024 |
| Standardised chemistry- and structure-based hold-out protocols; properties differ in sensitivity to the split criterion | MatFold ("systematic insights into materials discovery models' performance through standardized cross-validation protocols"; github.com/d2r2group/MatFold) |
| Kernelised leave-one-cluster-out as an evaluation tool | "Random projections and kernelised leave one cluster out cross-validation", arXiv 2206.08841 |
| Benchmarks for materials property prediction | Matbench (arXiv 2005.00707); Matbench Discovery (arXiv 2308.14920) |
| For BiDB, monolayer-level splitting avoids leakage between stackings of one monolayer | BiMat-ML, arXiv 2606.01012 (band-gap prediction; BiDB cross-validation "at the monolayer level") |
| ML for 2D mechanics uses random or in-domain test sets | Venturi et al., arXiv 2003.13418 (CGCNN on C2DB; metrics on the full set including training points); Fronzi et al., arXiv 1911.11559 (bilayer interlayer energy and C33; K-means-stratified 75/25) |

## 2. What this work can add (each is conditional on the results)
1. **A forecast, not another demonstration.** Phase B tests whether the loss of skill under family- or chemistry-grouped splits can be predicted before training from labels alone (skill of a pure group-mean predictor). Pre-registered (`mat-ml/results/panel_design.md`), with explicit falsification criteria. If it holds across the panel and later datasets it is a cheap diagnostic; if not, it is a documented negative result.
2. **2D-specific evidence with honest uncertainty.** First leakage-aware evaluation of 2D stiffness (C2DB) and of interlayer labels from monolayer information (BiDB) with cluster-bootstrap intervals: e.g. Y2D MAE ratio family/random 1.44 [1.30, 1.62], chemical-system/random 1.01 [0.99, 1.04]; BiDB binding energy monolayer/random 3.9 [3.0, 4.8].
3. **Does a graph network help? (Phase A)** A paired comparison on identical folds; either outcome is informative.
4. **A validation record for universal ML potentials on 2D elasticity and interlayer binding** (separate short paper or supplement): pre-registered gates, failures kept, D3 essential, interlayer gap failing, bilayer stiffness additive in-model but not DFT-validated.

## 3. What this work cannot claim
- That random-split inflation exists (known). That chemical-system splits matter (known; MatFold).
- Anything about DFT-validated bilayer stiffness.
- Generality beyond C2DB/BiDB until Phase B-2 (prospective datasets) is run.

## 4. Candidate titles (to be chosen after the results)
- "Can the damage done by random splits be predicted? A label-based diagnostic across 2D-material properties" (if H2/H3 hold)
- "Leakage-aware evaluation of machine-learning models for 2D mechanical and interlayer properties" (if the diagnostic fails; benchmark-and-negative-results paper)
