# Robustness: Y2D, seeds [42, 43, 44, 45, 46], 5-fold

MAE in N/m; r2_log is R2 of ln(Y2D) (stable under extrapolation). sd = std across seeds of the fold-mean.

## Summary by model / features / scheme

| model | features | scheme | mae_mean | mae_sd | r2_mean | r2_log_mean | r2_log_sd |
|---|---|---|---|---|---|---|---|
| gradient_boosting | all_plus_anon | random | 13.760 | 0.079 | 0.873 | 0.802 | 0.001 |
| gradient_boosting | all_plus_anon | family | 20.028 | 0.256 | 0.707 | 0.674 | 0.007 |
| gradient_boosting | all_plus_layergroup | random | 13.634 | 0.061 | 0.877 | 0.806 | 0.001 |
| gradient_boosting | all_plus_layergroup | family | 19.750 | 0.170 | 0.718 | 0.685 | 0.005 |
| gradient_boosting | all_plus_prototype | random | 13.629 | 0.094 | 0.876 | 0.807 | 0.002 |
| gradient_boosting | all_plus_prototype | family | 19.692 | 0.141 | 0.716 | 0.686 | 0.006 |
| gradient_boosting | composition_plus_prototype | random | 16.264 | 0.087 | 0.834 | 0.732 | 0.003 |
| gradient_boosting | composition_plus_prototype | family | 23.939 | 0.185 | 0.641 | 0.566 | 0.004 |
| gradient_boosting | prototype_only | random | 28.318 | 0.031 | 0.525 | 0.455 | 0.001 |
| gradient_boosting | prototype_only | family | 40.825 | 0.420 | -0.044 | 0.058 | 0.028 |

## Paired MAE ratios across seeds (grouped / random)

| model | features | family_over_random_mean | family_over_random_std | family_over_random_min |
|---|---|---|---|---|
| gradient_boosting | all_plus_anon | 1.456 | 0.021 | 1.436 |
| gradient_boosting | all_plus_layergroup | 1.449 | 0.012 | 1.431 |
| gradient_boosting | all_plus_prototype | 1.445 | 0.015 | 1.427 |
| gradient_boosting | composition_plus_prototype | 1.472 | 0.019 | 1.453 |
| gradient_boosting | prototype_only | 1.442 | 0.014 | 1.424 |

runtime 398s