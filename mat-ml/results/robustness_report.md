# Robustness: Y2D, seeds [42, 43, 44, 45, 46], 5-fold

MAE in N/m; r2_log is R2 of ln(Y2D) (stable under extrapolation). sd = std across seeds of the fold-mean.

## Summary by model / features / scheme

| model | features | scheme | mae_mean | mae_sd | r2_mean | r2_log_mean | r2_log_sd |
|---|---|---|---|---|---|---|---|
| chemsys_mean | none | random | 34.477 | 0.077 | 0.310 | 0.278 | 0.005 |
| chemsys_mean | none | chemsys | 42.394 | 0.000 | -0.117 | -0.000 | 0.000 |
| chemsys_mean | none | family | 34.281 | 0.205 | 0.282 | 0.281 | 0.004 |
| family_mean | none | random | 27.027 | 0.095 | 0.557 | 0.463 | 0.005 |
| family_mean | none | chemsys | 27.007 | 0.077 | 0.557 | 0.463 | 0.003 |
| family_mean | none | family | 42.394 | 0.001 | -0.120 | -0.000 | 0.000 |
| global_mean | none | random | 42.397 | 0.002 | -0.118 | -0.001 | 0.000 |
| global_mean | none | chemsys | 42.394 | 0.000 | -0.117 | -0.000 | 0.000 |
| global_mean | none | family | 42.394 | 0.001 | -0.120 | -0.000 | 0.000 |
| gradient_boosting | all | random | 13.798 | 0.108 | 0.872 | 0.802 | 0.001 |
| gradient_boosting | all | chemsys | 13.859 | 0.065 | 0.863 | 0.806 | 0.001 |
| gradient_boosting | all | family | 19.901 | 0.215 | 0.712 | 0.679 | 0.004 |
| gradient_boosting | composition | random | 19.237 | 0.088 | 0.786 | 0.653 | 0.005 |
| gradient_boosting | composition | chemsys | 18.593 | 0.096 | 0.789 | 0.697 | 0.002 |
| gradient_boosting | composition | family | 25.514 | 0.058 | 0.602 | 0.502 | 0.007 |
| gradient_boosting | structure | random | 19.817 | 0.056 | 0.753 | 0.698 | 0.001 |
| gradient_boosting | structure | chemsys | 19.982 | 0.074 | 0.746 | 0.696 | 0.003 |
| gradient_boosting | structure | family | 26.647 | 0.246 | 0.541 | 0.525 | 0.007 |
| random_forest | all | random | 14.606 | 0.080 | 0.847 | 0.785 | 0.002 |
| random_forest | all | chemsys | 14.738 | 0.090 | 0.838 | 0.788 | 0.000 |
| random_forest | all | family | 20.821 | 0.080 | 0.678 | 0.665 | 0.003 |

## Paired MAE ratios across seeds (grouped / random)

| model | features | family_over_random_mean | family_over_random_std | family_over_random_min | chemsys_over_random_mean | chemsys_over_random_std | chemsys_over_random_min |
|---|---|---|---|---|---|---|---|
| chemsys_mean | none | 0.994 | 0.006 | 0.985 | 1.230 | 0.003 | 1.227 |
| family_mean | none | 1.569 | 0.006 | 1.559 | 0.999 | 0.003 | 0.995 |
| global_mean | none | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| gradient_boosting | all | 1.442 | 0.018 | 1.419 | 1.004 | 0.007 | 0.994 |
| gradient_boosting | composition | 1.326 | 0.009 | 1.318 | 0.967 | 0.004 | 0.962 |
| gradient_boosting | structure | 1.345 | 0.011 | 1.330 | 1.008 | 0.004 | 1.003 |
| random_forest | all | 1.426 | 0.006 | 1.420 | 1.009 | 0.002 | 1.007 |

runtime 3173s