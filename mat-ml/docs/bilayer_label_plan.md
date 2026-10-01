# Plan: bilayer stiffness labels from a universal ML interatomic potential (UMLIP)

Status: **plan only; nothing in this document has been run.** Facts about this machine and the data come from checks made in this repository.
Statements marked *(from memory, verify)* are my background knowledge and have not been checked against a source in this project.
Time and compute figures are unmeasured guesses with wide error bars; Phase 0 replaces them with measurements.

## 1. Why this is needed

- No file we have contains bilayer stiffness. C2DB and JARVIS are monolayers; BiDB (binding energy, distance, no stiffness) is not in the project folder.
- Without bilayer labels the project cannot support its bilayer or multi-task claims (`docs/literature_notes.md`, `docs/results_summary.md` section 6).
- Fronzi et al. already covered out-of-plane C33 and interlayer energy for bilayers. The open space is **in-plane** stiffness of bilayers and how it depends on stacking.

## 2. What we would compute

For each bilayer (a monolayer taken twice, or two different monolayers) and stacking:

| Label | Definition | Unit |
|---|---|---|
| `C11, C12, C22, C66_xy` (Voigt xx, yy, xy block) | finite-strain stress response, 2D stress = 3D stress x cell height | N/m, **total for the bilayer** |
| `Y2D_total`, `Y2D_per_layer` | `Y2D_total = (C11*C22 - C12^2)/C22`; `per_layer = total / 2` | N/m |
| `poisson` | `C12/C22` | - |
| `dY_vs_mono` | `Y2D_total(bilayer) - 2 * Y2D(monolayer, same method)` | N/m |
| `E_int` | `(E_bilayer - 2 E_mono) / A`, signed (negative = bound), per interface, not divided by layers | meV/A^2 (1 eV/A^2 = 1000 meV/A^2) |
| `d_int` | equilibrium interlayer distance | Angstrom |

**Per-layer vs total, stated once and used everywhere:** stiffness in N/m is an extensive-in-layers quantity, so bilayer values are stored as **total** and
**per layer** (`/2`) side by side, never mixed. Monolayer reference rows have `n_layers = 1`, where the two coincide. `E_int` is per interface (a bilayer has
one) and is not divided by layers. The sign and unit conventions of the missing BiDB binding energy are unconfirmed, so no automatic comparison with BiDB is built in.

Unit conversion for stiffness: `1 eV/A^2 = 16.0218 N/m`; ASE stress is in `eV/A^3`, so `C [N/m] = (d sigma / d eps) [eV/A^3] * L_z [A] * 16.0218`,
with `L_z` the full cell height (vacuum included), the same convention that makes C2DB's N/m values come out right.

## 3. Method (matches the C2DB protocol so the validation is apples to apples)

C2DB's own stiffness files show the protocol: six strained structures at +/-1% in xx, yy and xy (strain_percent = 1.0).

1. **Relax** the structure with the UMLIP: ions plus in-plane cell, with the out-of-plane cell length fixed (vacuum constant), force threshold 0.005 eV/A
   (tight, because UMLIP noise would otherwise swamp 1% strains). ASE `FrechetCellFilter` with an in-plane mask is the intended tool.
2. **Strain** +/-1% in xx, yy and xy; relax **ions only** at fixed cell; read the stress.
3. **Central differences** give the 3x3 tensor. Apply the same rules as the existing pipeline: symmetrise; drop if not positive definite or if asymmetry exceeds 10%.
4. 7 relaxations per structure (1 reference + 6 strained). For bilayers the interlayer distance relaxes freely with the ions.
5. Record convergence (steps, final fmax) and drop non-converged structures.

## 4. Building the bilayers

- **Start with homobilayers** (the same C2DB monolayer twice). They avoid lattice mismatch and strain, so any stiffness change comes from stacking and interlayer coupling only.
- **Stackings:** enumerate a grid of in-plane shifts (for example 3x3 fractional offsets of layer 2 relative to layer 1, plus a 180-degree-rotated variant for non-centrosymmetric
  layers), set the initial interlayer distance from the vdW-radius sum or a short distance scan, relax, then **deduplicate** stackings that relax to the same structure
  (structure-matcher plus an energy tolerance). Record the stacking label and the energy ranking.
- **Heterobilayers** (two different monolayers with lattice mismatch up to about 2%, as in Fronzi) are a later phase. They introduce strain, so the strain contribution must be separated from the stacking contribution.
- **Twisted bilayers** are out of scope (large cells).
- **Selection for the pilot:** nonmagnetic, `ehull <= 0.1 eV/atom`, `nat <= 12`. From the clean C2DB set this leaves **3,380 monolayers in 395 structure families**
  (of 7,258 clean; 5,810 nonmagnetic; 6,276 with nat <= 12). Magnetic materials (1,448) are excluded at first because UMLIPs handle magnetic ordering poorly *(from memory, verify)*.
  Thickness median 4.2 A, 99th percentile 13.3 A; use vacuum of at least 15 A in the bilayer cell.

## 5. Models

Candidates *(from memory, verify availability and versions)*: the MACE-MP family, CHGNet, and another recent universal model such as SevenNet, ORB or MatterSim.
`mace-torch` 0.3.16, `chgnet` 0.4.2 and `ase` 3.29.0 are on PyPI for this Python. None of `torch`, `ase`, `mace`, `chgnet` is installed in the project venv yet.

Plan: run **two models** so their disagreement gives a per-structure uncertainty estimate that needs no DFT. Choose the final pair in Phase 0 after a speed and accuracy pilot.

Known problems to test for, not assume:
- **Systematic softening:** universal potentials tend to underestimate stiffness and phonon frequencies *(from memory, verify)*. Phase 1 measures the slope against C2DB directly.
- **Missing dispersion:** models trained on PBE data lack van der Waals interactions *(from memory, verify)*, which are what hold the bilayer together. A D3-type correction
  is needed for interlayer distance and `E_int`; whether it changes in-plane stiffness is a question for Phase 2.
- **Magnetism and metals** are less reliable, hence the nonmagnetic start.
- **Training-set overlap:** some of our monolayers derive from bulk layered compounds that may be in the model's training data; the effect on validation is unknown.

## 6. Phases and decision gates

| Phase | Work | Output | Gate |
|---|---|---|---|
| **0. Setup** (about 1-2 days) | install torch (CUDA), ase, mace-torch, chgnet in the venv; timing test on 20 structures; fix model pair and settings | `docs/` note with measured seconds per relaxation | Windows/CUDA install works (if not, use WSL) |
| **1. Monolayer calibration** | run the protocol on about 300 C2DB monolayers (stratified over families, nonmagnetic) and all 186 JARVIS; compare with DFT | table: MAE, R2(ln), Spearman, slope, bias per model and per family | **G1:** Spearman >= 0.9 and MAE no worse than about 1.5x the 8.6 N/m C2DB-vs-JARVIS disagreement; otherwise stop or calibrate |
| **2. Interlayer physics** | binding energy and distance for well-studied bilayers (graphite-like, MoS2, h-BN, WSe2) against literature values; test with and without D3 | `E_int`, `d_int` sanity table | **G2:** reasonable `E_int` and `d_int` for known cases (values from literature, to be confirmed by you) |
| **3. Pilot labels** | about 100-200 homobilayers, all unique stackings | `data/bilayer_labels.csv` | **G3:** the spread of `Y2D` across stackings of one monolayer must exceed the model-pair disagreement; otherwise stacking is not learnable from these labels |
| **4. Uncertainty** | two-model disagreement; optional small DFT spot check if DFT access exists | per-label uncertainty column | decide what the paper can claim |
| **5. Scale-up** | extend to the selected 3,380 monolayers; then consider heterobilayers | full label table | compute budget |
| **6. Learning** | same pipeline as Task A, with grouped splits by monolayer and by chemical system (the `monolayer` scheme already exists); optional multi-task with `E_int` | Task B on our own labels | - |

**Risk that matters most (G3):** the in-plane stiffness change from stacking is probably small, because layers are bound by weak vdW forces.
If `dY_vs_mono` and the spread across stackings are of the order of the UMLIP error (C2DB and JARVIS differ by about 8.6 N/m MAE on the same materials),
the labels cannot support a stacking-dependence claim. Errors may be correlated between the monolayer and bilayer calculations and so cancel in `dY_vs_mono`; that is a hypothesis to test in Phase 3 with the two-model comparison, not an assumption.

## 7. Compute estimate (unmeasured)

- Hardware here: 20-core i7-14700HX, RTX 4050 laptop GPU (6 GB), 15.7 GB RAM, 619 GB free disk. Small cells (bilayers of up to 24 atoms) should fit in GPU memory.
- Rough guess: 1-4 s per relaxation for a 14-atom bilayer on the GPU, so about 10-30 s per structure (7 relaxations) and about 1-3 min per monolayer with 6 stackings.
  The 3,380-monolayer scale-up would then take roughly 55-170 hours, and the Phase 1 + 3 pilots (about 500 structures) roughly one day.
  Treat this as +/- 3x until Phase 0 measures it.

## 8. Code layout (proposed; not created)

```
src/bilayer/build.py       stacking enumeration, vacuum, deduplication
src/bilayer/calc.py        UMLIP calculator wrapper (model name, device, dispersion on/off)
src/bilayer/stiffness.py   relax + 6 strains + central differences -> 3x3 tensor, convergence flags
src/bilayer/validate.py    monolayer comparison with C2DB / JARVIS, interlayer sanity table
src/bilayer/run.py         CLI with resume support, one row appended per structure
data/bilayer_labels.csv    columns: uid, monolayer_uid, partner_uid, stacking, n_layers, C11, C12, C22, C66_xy,
                           Y2D_total, Y2D_per_layer, poisson, dY_vs_mono, E_int_meV_A2, d_int_A, model, dispersion,
                           converged, fmax, steps
```

## 9. What this can and cannot claim

- Labels would be **surrogate** labels from a potential, validated against DFT only for monolayers (and for a few bilayers only if a DFT spot check is possible).
  Any paper claim must say "trained on UMLIP-derived labels" and report the validation.
- A model trained on these labels cannot be more accurate than the labels. The Task A grouped-split result (family gap of about 1.4x) tells us how it will behave on new prototypes, not how accurate the labels are.
- Bilayer **in-plane stiffness** and **multi-task** claims become testable only after G1-G3 pass.

## 10. Decisions needed from you

1. **Scope:** homobilayers first (recommended), or include heterobilayers from the start?
2. **Compute and install:** may I install torch (CUDA), ase, mace-torch and chgnet into the project venv (several GB)? Windows CUDA support is untested; WSL is the fallback.
3. **Models:** any preference, or should Phase 0 pick the pair on a speed and accuracy pilot?
4. **DFT access:** is there VASP, GPAW or Quantum ESPRESSO available for a small bilayer spot check? This decides whether labels get any bilayer DFT validation.
5. **Magnetic materials:** exclude at first (recommended)?
6. **BiDB:** if the files turn up, they give independent binding-energy and distance values for Phase 2; their units and sign convention need confirming first.
7. **Gate thresholds:** the G1-G3 numbers above are my proposals; adjust if you have different standards.
