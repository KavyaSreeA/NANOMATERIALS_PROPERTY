# Robustness: Y2D, seeds [42, 43, 44, 45, 46], 5-fold

MAE in N/m; r2_log is R2 of ln(Y2D) (stable under extrapolation). sd = std across seeds of the fold-mean.

## Summary by model / features / scheme

| model | features | scheme | mae_mean | mae_sd | r2_mean | r2_log_mean | r2_log_sd |
|---|---|---|---|---|---|---|---|
| gradient_boosting | all_no_spg | random | 14.085 | 0.127 | 0.867 | 0.794 | 0.002 |
| gradient_boosting | all_no_spg | family | 20.214 | 0.155 | 0.697 | 0.667 | 0.005 |
| gradient_boosting | composition_plus_spg | random | 16.931 | 0.064 | 0.823 | 0.719 | 0.003 |
| gradient_boosting | composition_plus_spg | family | 24.351 | 0.125 | 0.625 | 0.551 | 0.004 |
| gradient_boosting | geometry_no_spg | random | 20.895 | 0.046 | 0.726 | 0.667 | 0.001 |
| gradient_boosting | geometry_no_spg | family | 26.280 | 0.286 | 0.543 | 0.511 | 0.008 |
| gradient_boosting | spg_only | random | 38.011 | 0.013 | 0.073 | 0.205 | 0.001 |
| gradient_boosting | spg_only | family | 44.257 | 0.338 | -0.168 | -0.016 | 0.020 |

## Paired MAE ratios across seeds (grouped / random)

| model | features | family_over_random_mean | family_over_random_std | family_over_random_min |
|---|---|---|---|---|
| gradient_boosting | all_no_spg | 1.435 | 0.019 | 1.407 |
| gradient_boosting | composition_plus_spg | 1.438 | 0.011 | 1.421 |
| gradient_boosting | geometry_no_spg | 1.258 | 0.012 | 1.247 |
| gradient_boosting | spg_only | 1.164 | 0.009 | 1.153 |

runtime 4631s