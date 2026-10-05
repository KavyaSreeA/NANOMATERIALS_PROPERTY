# Phase A design: does a structure-aware graph network narrow the family gap?  (written BEFORE any graph model was trained)

Date written: 2026-10-04. Task A results are not modified; outputs are prefixed `phaseA_`.

## Question
LightGBM on composition + simple geometry loses skill when whole structure families are held out (MAE ratio family/random 1.44, 95% CI [1.30, 1.62], cluster bootstrap over families).
Does a crystal-graph network (CGCNN-type), which sees the actual atomic neighbourhood, lose less?

## Design
- Target: ln(Y2D) of the 7,258 clean C2DB monolayers (same rows and units as Task A). Metrics in N/m.
- Folds: IDENTICAL to the LightGBM out-of-fold run (seed 42; schemes random, chemical-system, family), exported by `src/gnn/make_folds.py`, so differences are paired.
- Model: CGCNN-type network written in PyTorch (3 gated graph-convolution layers, 64 hidden units, mean pooling, 2-layer read-out); node input = 10 standardised elemental properties
  (atomic number, group, row, electronegativity, atomic radius, atomic mass, first ionisation energy, electron affinity, Mendeleev number, molar volume; missing -> column median);
  edges = all neighbours within 6 A (periodic in the plane only, at least one neighbour per atom), 40 Gaussian distance functions (0-6 A, width 0.15 A).
  Training: L1 loss on standardised ln Y2D, Adam (lr 2e-3, cosine decay), batch 128, up to 200 epochs, early stopping (patience 30) on a validation subset taken from the training fold with the SAME grouping rule as the outer split
  (so early stopping never sees test-family information). No hyper-parameter search. One seed in the main run (seed 42).
- Comparison: CGCNN vs the stored LightGBM out-of-fold predictions on the same rows.

## Hypotheses (fixed now)
- **A1.** Under the family-grouped split CGCNN does not beat LightGBM by a margin that excludes zero: the 95% interval (bootstrap over families, paired) of MAE_LightGBM - MAE_CGCNN includes 0 or is below 2 N/m.
- **A2.** CGCNN's family/random MAE ratio is within +/-0.15 of LightGBM's (1.44), i.e. a structure-aware model does not remove the gap.
- **A3 (alternative that would be a finding).** If CGCNN's family/random ratio is below 1.25 with a 95% interval that excludes 1.44, graph models narrow the gap and the paper's framing must change.
Reported regardless of outcome: absolute MAE and R2 for all three schemes, with family-cluster bootstrap CIs, and the number of training epochs used per fold.

## Limits stated in advance
One seed; no tuning; 7k training graphs is small for a GNN, so a weak CGCNN may reflect data size and not representation; node features are hand-chosen elemental properties; the family definition is by anonymous formula and layer group.
