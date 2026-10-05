# Phase C design: honest uncertainty and fair baselines (written BEFORE any Phase C number was computed)

Date written: 2026-10-04. Earlier results are not modified; outputs are prefixed `phaseC_`.

## C1. Cluster-bootstrap confidence intervals (cheap; uses stored out-of-fold predictions, seed 42)
Why: the seed standard deviations reported so far re-seed splits and models on a FIXED dataset, so they do not include the uncertainty from having a finite number of structure families / monolayers.
Method: resample the clustering unit with replacement (2,000 resamples, seed 0), recompute pooled out-of-fold metrics on the resampled rows, report 2.5th and 97.5th percentiles.
- Task A (Y2D, LightGBM, 141 features; `results/diagnostics/taskA_oof_predictions.csv`): unit = structure family. Quantities: MAE and R2 for random / chemical-system / family splits; MAE ratios chemsys/random and family/random; skill retention.
- Task B (binding energy, gap; LightGBM `mono_stiffness`; `taskB_oof_predictions.csv`): unit = monolayer for the random and monolayer-grouped schemes and for their ratio; unit = family for the family-grouped scheme and its ratio.
No hypothesis threshold is attached: the point is to attach intervals to every headline number and to state whether the key contrasts (family/random > 1; monolayer/random >> 1) survive.

## C2. Nested, group-aware hyper-parameter tuning (CPU-heavy; run after the panel)
Why: all models so far used fixed settings; a reviewer will ask whether tuning closes the gaps or changes the conclusions.
Method: LightGBM random search (24 configurations over num_leaves, learning rate, n_estimators, min_child_samples, subsample, colsample_bytree, reg_lambda), 3-fold INNER cross-validation that matches the outer scheme
(random inner folds for the random outer split; GroupKFold by family for the family outer split; by monolayer for Task B monolayer split), 5 OUTER folds, seed 42. Targets: Task A Y2D; Task B binding energy.
Reported: tuned vs default MAE under each scheme and the change in the grouped/random ratio.
Expectation recorded in advance (not a pass/fail gate): tuning lowers absolute MAE by a few percent but leaves the grouped/random ratios essentially unchanged (|change| < 0.1). If tuning shrinks the family gap by more than 0.1 in ratio, the paper's framing of the gap as structural must be softened.
