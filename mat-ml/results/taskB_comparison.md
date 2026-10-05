# Task B results (auto-generated)

Design: results/taskB_design.md. Seed 42 full grid, 5-fold; MAE in original units (meV/A^2 for binding energy, A for distance).
Per-fold spread = mean +/- std (min-max) over the folds.

## binding_energy_zscan

valid 9993 (removed 199), monolayers 961, families 108, ceiling (between-monolayer share of ln-variance) 0.955

| scheme | features | model | MAE | R2 | r2_log_mean | spearman_mean | folds_better |
|---|---|---|---|---|---|---|---|
| random | none | mean_arith | 18.51 ± 0.41 (17.82-18.92) | -0.000 ± 0.000 | -0.206 | nan | - |
| random | none | mean_geom | 15.75 ± 0.38 (15.12-16.09) | -0.053 ± 0.004 | -0.000 | nan | 5/5 |
| random | mono_basic | ridge | 12.03 ± 0.45 (11.38-12.47) | 0.278 ± 0.039 | 0.515 | 0.744 | 5/5 |
| random | mono_basic | random_forest | 2.41 ± 0.05 (2.35-2.46) | 0.986 ± 0.003 | 0.939 | 0.943 | 5/5 |
| random | mono_basic | gradient_boosting | 2.46 ± 0.05 (2.42-2.53) | 0.984 ± 0.003 | 0.940 | 0.944 | 5/5 |
| random | mono_stiffness | ridge | 11.98 ± 0.46 (11.31-12.44) | 0.283 ± 0.039 | 0.522 | 0.748 | 5/5 |
| random | mono_stiffness | random_forest | 2.41 ± 0.05 (2.35-2.46) | 0.986 ± 0.003 | 0.939 | 0.943 | 5/5 |
| random | mono_stiffness | gradient_boosting | 2.43 ± 0.06 (2.36-2.52) | 0.986 ± 0.003 | 0.940 | 0.944 | 5/5 |
| random | mono_stiffness_stacking | ridge | 11.96 ± 0.46 (11.31-12.42) | 0.285 ± 0.039 | 0.523 | 0.749 | 5/5 |
| random | mono_stiffness_stacking | random_forest | 1.92 ± 0.06 (1.84-2.01) | 0.988 ± 0.003 | 0.962 | 0.965 | 5/5 |
| random | mono_stiffness_stacking | gradient_boosting | 2.02 ± 0.08 (1.90-2.10) | 0.974 ± 0.009 | 0.968 | 0.972 | 5/5 |
| monolayer | none | mean_arith | 18.50 ± 1.83 (16.62-21.17) | -0.005 ± 0.004 | -0.215 | nan | - |
| monolayer | none | mean_geom | 15.75 ± 2.07 (13.65-18.79) | -0.054 ± 0.010 | -0.001 | nan | 5/5 |
| monolayer | mono_basic | ridge | 14.71 ± 2.52 (11.85-18.67) | -0.051 ± 0.382 | 0.265 | 0.628 | 5/5 |
| monolayer | mono_basic | random_forest | 8.98 ± 1.52 (6.53-10.39) | 0.490 ± 0.149 | 0.649 | 0.842 | 5/5 |
| monolayer | mono_basic | gradient_boosting | 9.05 ± 1.51 (6.79-10.77) | 0.486 ± 0.127 | 0.642 | 0.832 | 5/5 |
| monolayer | mono_stiffness | ridge | 14.69 ± 2.32 (12.11-18.41) | 0.024 ± 0.228 | 0.262 | 0.630 | 5/5 |
| monolayer | mono_stiffness | random_forest | 8.92 ± 1.42 (6.59-10.31) | 0.502 ± 0.135 | 0.655 | 0.846 | 5/5 |
| monolayer | mono_stiffness | gradient_boosting | 9.37 ± 1.71 (6.69-11.10) | 0.429 ± 0.163 | 0.628 | 0.837 | 5/5 |
| monolayer | mono_stiffness_stacking | ridge | 14.65 ± 2.31 (12.06-18.32) | 0.032 ± 0.223 | 0.266 | 0.630 | 5/5 |
| monolayer | mono_stiffness_stacking | random_forest | 8.39 ± 1.45 (6.00-9.88) | 0.506 ± 0.128 | 0.675 | 0.865 | 5/5 |
| monolayer | mono_stiffness_stacking | gradient_boosting | 8.80 ± 1.42 (6.44-10.12) | 0.448 ± 0.151 | 0.659 | 0.866 | 5/5 |
| family | none | mean_arith | 18.54 ± 1.94 (17.08-21.69) | -0.005 ± 0.005 | -0.217 | nan | - |
| family | none | mean_geom | 15.77 ± 2.26 (13.93-19.51) | -0.054 ± 0.015 | -0.003 | nan | 5/5 |
| family | mono_basic | ridge | 20.69 ± 5.82 (15.24-30.21) | -0.800 ± 0.834 | -0.151 | 0.528 | 2/5 |
| family | mono_basic | random_forest | 13.52 ± 3.91 (10.54-20.12) | 0.173 ± 0.157 | 0.337 | 0.720 | 5/5 |
| family | mono_basic | gradient_boosting | 13.39 ± 3.99 (10.27-20.28) | 0.185 ± 0.169 | 0.339 | 0.711 | 5/5 |
| family | mono_stiffness | ridge | 20.69 ± 4.81 (15.51-28.33) | -1.096 ± 1.315 | -0.162 | 0.513 | 2/5 |
| family | mono_stiffness | random_forest | 13.39 ± 3.89 (10.44-19.99) | 0.188 ± 0.156 | 0.364 | 0.725 | 5/5 |
| family | mono_stiffness | gradient_boosting | 13.85 ± 4.67 (10.64-22.06) | 0.188 ± 0.200 | 0.325 | 0.685 | 4/5 |
| family | mono_stiffness_stacking | ridge | 20.74 ± 4.91 (15.36-28.46) | -1.184 ± 1.462 | -0.166 | 0.514 | 2/5 |
| family | mono_stiffness_stacking | random_forest | 13.23 ± 4.01 (10.16-20.06) | 0.201 ± 0.155 | 0.382 | 0.740 | 5/5 |
| family | mono_stiffness_stacking | gradient_boosting | 13.92 ± 5.03 (10.34-22.75) | 0.188 ± 0.224 | 0.346 | 0.706 | 4/5 |

## distance

valid 10189 (removed 3), monolayers 992, families 109, ceiling (between-monolayer share of ln-variance) 0.771

| scheme | features | model | MAE | R2 | r2_log_mean | spearman_mean | folds_better |
|---|---|---|---|---|---|---|---|
| random | none | mean_arith | 0.42 ± 0.01 (0.41-0.43) | -0.001 ± 0.002 | -0.009 | nan | - |
| random | none | mean_geom | 0.43 ± 0.01 (0.41-0.44) | -0.016 ± 0.009 | -0.001 | nan | 0/5 |
| random | mono_basic | ridge | 0.28 ± 0.01 (0.27-0.29) | 0.546 ± 0.019 | 0.531 | 0.715 | 5/5 |
| random | mono_basic | random_forest | 0.25 ± 0.00 (0.25-0.26) | 0.651 ± 0.012 | 0.678 | 0.757 | 5/5 |
| random | mono_basic | gradient_boosting | 0.25 ± 0.00 (0.24-0.25) | 0.661 ± 0.010 | 0.685 | 0.766 | 5/5 |
| random | mono_stiffness | ridge | 0.28 ± 0.01 (0.27-0.29) | 0.548 ± 0.020 | 0.536 | 0.714 | 5/5 |
| random | mono_stiffness | random_forest | 0.25 ± 0.00 (0.25-0.26) | 0.651 ± 0.012 | 0.678 | 0.757 | 5/5 |
| random | mono_stiffness | gradient_boosting | 0.25 ± 0.00 (0.24-0.25) | 0.659 ± 0.011 | 0.684 | 0.764 | 5/5 |
| random | mono_stiffness_stacking | ridge | 0.28 ± 0.01 (0.27-0.29) | 0.554 ± 0.020 | 0.540 | 0.720 | 5/5 |
| random | mono_stiffness_stacking | random_forest | 0.17 ± 0.00 (0.16-0.17) | 0.824 ± 0.007 | 0.799 | 0.885 | 5/5 |
| random | mono_stiffness_stacking | gradient_boosting | 0.15 ± 0.00 (0.15-0.15) | 0.852 ± 0.009 | 0.811 | 0.909 | 5/5 |
| monolayer | none | mean_arith | 0.42 ± 0.01 (0.41-0.42) | -0.000 ± 0.000 | -0.008 | nan | - |
| monolayer | none | mean_geom | 0.43 ± 0.01 (0.42-0.43) | -0.015 ± 0.004 | -0.000 | nan | 0/5 |
| monolayer | mono_basic | ridge | 0.31 ± 0.03 (0.28-0.35) | 0.407 ± 0.137 | 0.448 | 0.678 | 5/5 |
| monolayer | mono_basic | random_forest | 0.29 ± 0.02 (0.26-0.32) | 0.509 ± 0.086 | 0.512 | 0.728 | 5/5 |
| monolayer | mono_basic | gradient_boosting | 0.28 ± 0.01 (0.26-0.28) | 0.554 ± 0.044 | 0.543 | 0.747 | 5/5 |
| monolayer | mono_stiffness | ridge | 0.31 ± 0.03 (0.28-0.35) | 0.410 ± 0.125 | 0.446 | 0.676 | 5/5 |
| monolayer | mono_stiffness | random_forest | 0.29 ± 0.02 (0.26-0.32) | 0.508 ± 0.088 | 0.512 | 0.727 | 5/5 |
| monolayer | mono_stiffness | gradient_boosting | 0.27 ± 0.01 (0.25-0.28) | 0.572 ± 0.048 | 0.564 | 0.754 | 5/5 |
| monolayer | mono_stiffness_stacking | ridge | 0.31 ± 0.03 (0.28-0.35) | 0.412 ± 0.129 | 0.449 | 0.683 | 5/5 |
| monolayer | mono_stiffness_stacking | random_forest | 0.21 ± 0.03 (0.18-0.26) | 0.638 ± 0.123 | 0.588 | 0.828 | 5/5 |
| monolayer | mono_stiffness_stacking | gradient_boosting | 0.20 ± 0.02 (0.17-0.21) | 0.696 ± 0.045 | 0.630 | 0.852 | 5/5 |
| family | none | mean_arith | 0.42 ± 0.01 (0.41-0.43) | -0.000 ± 0.000 | -0.008 | nan | - |
| family | none | mean_geom | 0.43 ± 0.01 (0.42-0.44) | -0.015 ± 0.005 | -0.001 | nan | 0/5 |
| family | mono_basic | ridge | 0.36 ± 0.05 (0.29-0.41) | 0.168 ± 0.108 | 0.272 | 0.606 | 5/5 |
| family | mono_basic | random_forest | 0.33 ± 0.05 (0.25-0.38) | 0.333 ± 0.235 | 0.272 | 0.676 | 5/5 |
| family | mono_basic | gradient_boosting | 0.32 ± 0.02 (0.28-0.34) | 0.385 ± 0.056 | 0.323 | 0.690 | 5/5 |
| family | mono_stiffness | ridge | 0.36 ± 0.05 (0.29-0.43) | 0.180 ± 0.159 | 0.284 | 0.605 | 4/5 |
| family | mono_stiffness | random_forest | 0.33 ± 0.05 (0.25-0.38) | 0.321 ± 0.220 | 0.220 | 0.670 | 5/5 |
| family | mono_stiffness | gradient_boosting | 0.32 ± 0.02 (0.29-0.34) | 0.375 ± 0.080 | 0.291 | 0.673 | 5/5 |
| family | mono_stiffness_stacking | ridge | 0.36 ± 0.05 (0.29-0.43) | 0.174 ± 0.159 | 0.281 | 0.607 | 4/5 |
| family | mono_stiffness_stacking | random_forest | 0.30 ± 0.05 (0.23-0.36) | 0.377 ± 0.240 | 0.284 | 0.711 | 5/5 |
| family | mono_stiffness_stacking | gradient_boosting | 0.27 ± 0.02 (0.24-0.30) | 0.489 ± 0.052 | 0.391 | 0.744 | 5/5 |

## binding_energy_gs

valid 9740 (removed 452), monolayers 971, families 106, ceiling (between-monolayer share of ln-variance) 0.856

| scheme | features | model | MAE | R2 | r2_log_mean | spearman_mean | folds_better |
|---|---|---|---|---|---|---|---|
| random | none | mean_arith | 6.23 ± 0.12 (6.05-6.34) | -0.001 ± 0.001 | -0.056 | nan | - |
| random | none | mean_geom | 5.98 ± 0.16 (5.73-6.15) | -0.036 ± 0.007 | -0.000 | nan | 5/5 |
| random | mono_stiffness | ridge | 3.07 ± 0.14 (2.88-3.21) | 0.598 ± 0.137 | 0.693 | 0.839 | 5/5 |
| random | mono_stiffness | random_forest | 2.37 ± 0.07 (2.28-2.44) | 0.875 ± 0.019 | 0.804 | 0.882 | 5/5 |
| random | mono_stiffness | gradient_boosting | 2.34 ± 0.07 (2.26-2.42) | 0.881 ± 0.016 | 0.807 | 0.884 | 5/5 |
| monolayer | none | mean_arith | 6.23 ± 0.12 (6.10-6.35) | -0.000 ± 0.000 | -0.055 | nan | - |
| monolayer | none | mean_geom | 5.98 ± 0.12 (5.85-6.10) | -0.036 ± 0.003 | -0.000 | nan | 5/5 |
| monolayer | mono_stiffness | ridge | 3.41 ± 0.30 (3.02-3.72) | 0.529 ± 0.142 | 0.627 | 0.803 | 5/5 |
| monolayer | mono_stiffness | random_forest | 2.95 ± 0.24 (2.64-3.28) | 0.688 ± 0.140 | 0.683 | 0.833 | 5/5 |
| monolayer | mono_stiffness | gradient_boosting | 2.97 ± 0.23 (2.74-3.32) | 0.694 ± 0.143 | 0.681 | 0.832 | 5/5 |
| family | none | mean_arith | 6.22 ± 0.32 (5.89-6.74) | -0.003 ± 0.003 | -0.059 | nan | - |
| family | none | mean_geom | 5.98 ± 0.38 (5.65-6.63) | -0.037 ± 0.013 | -0.001 | nan | 5/5 |
| family | mono_stiffness | ridge | 4.31 ± 0.71 (3.80-5.48) | 0.164 ± 0.380 | 0.480 | 0.749 | 5/5 |
| family | mono_stiffness | random_forest | 3.58 ± 0.44 (3.29-4.30) | 0.581 ± 0.134 | 0.542 | 0.740 | 5/5 |
| family | mono_stiffness | gradient_boosting | 3.59 ± 0.33 (3.28-4.09) | 0.575 ± 0.116 | 0.532 | 0.741 | 5/5 |

## Leakage (fraction of test bilayers whose key also occurs in training; seed 42)

| target|scheme | monolayer_in_train | family_in_train | formula_in_train | chemsys_in_train |
|---|---|---|---|---|
| binding_energy_zscan|random | 0.993 | 1.000 | 0.995 | 0.997 |
| binding_energy_zscan|monolayer | 0.000 | 0.888 | 0.209 | 0.371 |
| binding_energy_zscan|family | 0.000 | 0.000 | 0.107 | 0.296 |
| distance|random | 0.992 | 1.000 | 0.994 | 0.995 |
| distance|monolayer | 0.000 | 0.883 | 0.192 | 0.350 |
| distance|family | 0.000 | 0.000 | 0.173 | 0.332 |
| binding_energy_gs|random | 0.990 | 1.000 | 0.993 | 0.995 |
| binding_energy_gs|monolayer | 0.000 | 0.888 | 0.217 | 0.357 |
| binding_energy_gs|family | 0.000 | 0.000 | 0.117 | 0.298 |

## Five-seed robustness (LightGBM and baselines; mean and std over seeds of the fold means)

| target | rule | scheme | features | model | n_seeds | mae_mean | mae_sd | r2_mean | r2_sd | r2_log_mean | spearman_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| binding_energy_zscan | upper | family | mono_basic | gradient_boosting | 5 | 13.539 | 0.092 | 0.188 | 0.009 | 0.344 | 0.713 |
| binding_energy_zscan | upper | family | mono_stiffness | gradient_boosting | 5 | 13.589 | 0.301 | 0.200 | 0.019 | 0.351 | 0.707 |
| binding_energy_zscan | upper | family | none | mean_arith | 5 | 18.535 | 0.000 | -0.005 | 0.000 | -0.217 | nan |
| binding_energy_zscan | upper | family | none | mean_geom | 5 | 15.768 | 0.000 | -0.054 | 0.000 | -0.003 | nan |
| binding_energy_zscan | upper | monolayer | mono_basic | gradient_boosting | 5 | 9.064 | 0.237 | 0.490 | 0.024 | 0.639 | 0.829 |
| binding_energy_zscan | upper | monolayer | mono_stiffness | gradient_boosting | 5 | 9.069 | 0.424 | 0.483 | 0.052 | 0.643 | 0.833 |
| binding_energy_zscan | upper | monolayer | none | mean_arith | 5 | 18.507 | 0.002 | -0.003 | 0.001 | -0.212 | nan |
| binding_energy_zscan | upper | monolayer | none | mean_geom | 5 | 15.751 | 0.000 | -0.054 | 0.000 | -0.001 | nan |
| binding_energy_zscan | upper | random | mono_basic | gradient_boosting | 5 | 2.443 | 0.015 | 0.984 | 0.002 | 0.940 | 0.944 |
| binding_energy_zscan | upper | random | mono_stiffness | gradient_boosting | 5 | 2.422 | 0.010 | 0.985 | 0.003 | 0.940 | 0.944 |
| binding_energy_zscan | upper | random | none | mean_arith | 5 | 18.508 | 0.001 | -0.001 | 0.000 | -0.207 | nan |
| binding_energy_zscan | upper | random | none | mean_geom | 5 | 15.752 | 0.001 | -0.053 | 0.000 | -0.000 | nan |
| distance | upper | family | mono_basic | gradient_boosting | 5 | 0.317 | 0.005 | 0.386 | 0.024 | 0.308 | 0.682 |
| distance | upper | family | mono_stiffness | gradient_boosting | 5 | 0.316 | 0.002 | 0.391 | 0.009 | 0.310 | 0.684 |
| distance | upper | family | none | mean_arith | 5 | 0.415 | 0.000 | -0.000 | 0.000 | -0.008 | nan |
| distance | upper | family | none | mean_geom | 5 | 0.425 | 0.000 | -0.015 | 0.000 | -0.001 | nan |
| distance | upper | monolayer | mono_basic | gradient_boosting | 5 | 0.279 | 0.006 | 0.513 | 0.033 | 0.389 | 0.751 |
| distance | upper | monolayer | mono_stiffness | gradient_boosting | 5 | 0.278 | 0.005 | 0.513 | 0.036 | 0.398 | 0.751 |
| distance | upper | monolayer | none | mean_arith | 5 | 0.415 | 0.000 | -0.000 | 0.000 | -0.008 | nan |
| distance | upper | monolayer | none | mean_geom | 5 | 0.425 | 0.000 | -0.015 | 0.001 | -0.001 | nan |
| distance | upper | random | mono_basic | gradient_boosting | 5 | 0.247 | 0.000 | 0.663 | 0.003 | 0.689 | 0.767 |
| distance | upper | random | mono_stiffness | gradient_boosting | 5 | 0.248 | 0.001 | 0.660 | 0.003 | 0.688 | 0.764 |
| distance | upper | random | none | mean_arith | 5 | 0.415 | 0.000 | -0.001 | 0.000 | -0.008 | nan |
| distance | upper | random | none | mean_geom | 5 | 0.425 | 0.000 | -0.015 | 0.000 | -0.001 | nan |

## Sensitivity to the outlier rule (LightGBM on mono_stiffness and the arithmetic mean predictor; seed 42)

| target | rule | scheme | features | model | mae_mean | mae_std | r2_mean | r2_std |
|---|---|---|---|---|---|---|---|---|
| binding_energy_zscan | none | random | none | mean_arith | 34.448 | 3.448 | -0.002 | 0.004 |
| binding_energy_zscan | none | random | mono_stiffness | gradient_boosting | 4.237 | 1.820 | 0.846 | 0.260 |
| binding_energy_zscan | none | monolayer | none | mean_arith | 34.453 | 5.709 | -0.008 | 0.010 |
| binding_energy_zscan | none | monolayer | mono_stiffness | gradient_boosting | 19.032 | 6.131 | 0.203 | 0.281 |
| binding_energy_zscan | none | family | none | mean_arith | 34.528 | 4.159 | -0.006 | 0.013 |
| binding_energy_zscan | none | family | mono_stiffness | gradient_boosting | 21.929 | 4.875 | 0.062 | 0.091 |
| distance | none | random | none | mean_arith | 0.416 | 0.009 | -0.001 | 0.002 |
| distance | none | random | mono_stiffness | gradient_boosting | 0.248 | 0.004 | 0.659 | 0.011 |
| distance | none | monolayer | none | mean_arith | 0.415 | 0.007 | -0.000 | 0.000 |
| distance | none | monolayer | mono_stiffness | gradient_boosting | 0.271 | 0.013 | 0.572 | 0.048 |
| distance | none | family | none | mean_arith | 0.415 | 0.010 | -0.000 | 0.000 |
| distance | none | family | mono_stiffness | gradient_boosting | 0.320 | 0.023 | 0.375 | 0.080 |
| binding_energy_zscan | two_sided | random | none | mean_arith | 18.506 | 0.409 | -0.000 | 0.000 |
| binding_energy_zscan | two_sided | random | mono_stiffness | gradient_boosting | 2.428 | 0.058 | 0.986 | 0.003 |
| binding_energy_zscan | two_sided | monolayer | none | mean_arith | 18.505 | 1.833 | -0.005 | 0.004 |
| binding_energy_zscan | two_sided | monolayer | mono_stiffness | gradient_boosting | 9.366 | 1.715 | 0.429 | 0.163 |
| binding_energy_zscan | two_sided | family | none | mean_arith | 18.535 | 1.935 | -0.005 | 0.005 |
| binding_energy_zscan | two_sided | family | mono_stiffness | gradient_boosting | 13.850 | 4.672 | 0.188 | 0.200 |
| distance | two_sided | random | none | mean_arith | 0.393 | 0.004 | -0.001 | 0.001 |
| distance | two_sided | random | mono_stiffness | gradient_boosting | 0.245 | 0.002 | 0.610 | 0.006 |
| distance | two_sided | monolayer | none | mean_arith | 0.393 | 0.004 | -0.000 | 0.000 |
| distance | two_sided | monolayer | mono_stiffness | gradient_boosting | 0.251 | 0.010 | 0.582 | 0.044 |
| distance | two_sided | family | none | mean_arith | 0.393 | 0.011 | -0.002 | 0.002 |
| distance | two_sided | family | mono_stiffness | gradient_boosting | 0.278 | 0.014 | 0.489 | 0.069 |
