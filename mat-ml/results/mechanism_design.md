# Phase D1 design: where does the family penalty come from?  (written BEFORE any D1 number was computed)

Date written: 2026-10-06. Earlier results are not modified; outputs of this phase are prefixed `mechanism_`.

## Question
For the 2D Young's modulus (LightGBM, 141 features) the MAE rises by 1.44x when whole structure families are held out, but by 1.01x when whole chemical systems are held out.
The working explanation is **lost twins**: under random splitting a test material usually has relatives from its own structure family in the training rows, under family-grouped splitting it has none.
This phase tests that explanation with a dose-response analysis and with two controls, using only saved out-of-fold predictions. It does not establish a physical mechanism.

## Data
`results/diagnostics/taskA_oof_predictions.csv` (seed 42, 5 folds, LightGBM, all 141 features, Y2D in N/m, schemes random / chemsys / family; 7,258 rows).
Per row: absolute error e_s = |pred - y| under scheme s. Family size n_f = number of rows (of the 7,258) with the same family key; chemical-system size n_c likewise.
Expected twins in the training set: under a random split a row has about 0.8 (n_f - 1) family relatives in training; under the family split it has none, for every n_f.

## Quantity
Penalty ratio for a set of rows B: Q_fam(B) = sum_{i in B} e_family(i) / sum_{i in B} e_random(i); Q_chem(B) analogous with e_chemsys.
Family-size bins: 1 (singleton), 2-3, 4-9, 10-49, >= 50. Chemical-system bins: 1, 2-3, 4-9, >= 10.
Uncertainty: cluster bootstrap over families (2,000 resamples, seed 0; chemical-system analyses resample chemical systems). Differences use the same resample.

## Hypotheses (fixed now)
- **M1 (control: singletons).** A singleton family has no twin under either scheme, so the two splits differ only by which other materials are in training. Prediction: Q_fam(singletons) has a 95% interval containing 1 and a point estimate in [0.90, 1.10].
- **M2 (dose-response).** The more twins a family has, the more the family split removes. Prediction: Q_fam(n_f >= 10) - Q_fam(n_f <= 3) >= 0.15, with the 95% interval of the difference above 0.
- **M3 (per-family trend, exploratory).** Among families with at least 3 members, Spearman(log n_f, per-family Q_fam) > 0.2 with the 95% interval above 0.
- **M4 (control: chemistry).** There is no dose-response for chemistry: Q_chem lies in [0.90, 1.15] in every chemical-system bin that contains at least 100 rows.
- **M5 (descriptive).** Report the share of the total penalty (sum e_family - sum e_random) carried by families with n_f >= 10, next to their share of rows.

## What each outcome would mean
- M1 and M2 hold: consistent with lost twins as the main cause; the penalty is largest where random splits benefited most.
- M1 fails (singletons also show Q_fam > 1.10): the penalty cannot be explained by lost twins alone; part of it reflects that family-grouped folds put different (possibly harder) materials in the test set, i.e. selection. State this and soften the wording of the paper accordingly.
- M2 fails (no dose-response): "overlap" is not supported as the mechanism; report as a negative result.
- M4 fails: chemistry also shows twin-like leakage in some bins; revise the statement that chemistry does not matter.

## Part B (needs the features, run locally; optional)
M6. Feature-space shift: for each test row, the distance to its nearest training row in the standardised 141-feature space under random and family splits (seed 42, same folds).
Prediction: the median distance under the family split exceeds that under the random split by at least 30%, and the per-family penalty correlates with the per-family distance increase (Spearman > 0.3, 95% interval above 0).
If the distance does not increase, the penalty is not an extrapolation-in-feature-space effect and the "new prototype" reading needs a different explanation.

## Limits stated in advance
One seed and one model; family sizes are very uneven (largest 432), so large-family bins are driven by a few families and folds; the size bins are chosen by me (rule above, not tuned); Q is a ratio of sums, so it is dominated by stiff materials with large absolute errors;
a dose-response is evidence for the explanation, not proof of a mechanism (large families may also be intrinsically different).

## Reproducibility
`python -m src.mechanism` (Part A, needs only pandas and numpy; reads the saved predictions) and `python -m src.mechanism --distance` (Part B, needs the project data and cache).
