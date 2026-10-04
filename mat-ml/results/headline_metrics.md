# Headline metrics (auto-generated from the saved result CSVs; `python -m src.make_tables`)

MAE and RMSE are in original units (Task A: Y2D in N/m, Poisson ratio dimensionless; Task B: binding energy in meV/A^2, gap in A). `+/-` is the standard deviation over the five folds. ln R2 = R2 of the log-transformed target. Seed 42, 5-fold.

## Task A: Y2D (141 features)

| scheme | model | MAE | RMSE | R2 | ln R2 |
|---|---|---|---|---|---|
| random | ridge | 22.51 +/- 0.75 | 37.986 | 0.686 +/- 0.014 | 0.635 |
| random | random_forest | 14.67 +/- 0.38 | 26.263 | 0.849 +/- 0.018 | 0.784 |
| random | gradient_boosting | 13.74 +/- 0.25 | 24.059 | 0.873 +/- 0.013 | 0.802 |
| cluster | ridge | 22.57 +/- 0.60 | 38.010 | 0.685 +/- 0.038 | 0.634 |
| cluster | random_forest | 14.72 +/- 0.44 | 26.899 | 0.842 +/- 0.029 | 0.782 |
| cluster | gradient_boosting | 13.98 +/- 0.54 | 24.857 | 0.865 +/- 0.017 | 0.797 |
| chemsys | ridge | 22.76 +/- 0.80 | 38.697 | 0.672 +/- 0.054 | 0.632 |
| chemsys | random_forest | 14.82 +/- 0.59 | 27.551 | 0.834 +/- 0.032 | 0.787 |
| chemsys | gradient_boosting | 13.91 +/- 0.57 | 24.910 | 0.864 +/- 0.022 | 0.807 |
| family | ridge | 31.12 +/- 14.19 | 124.770 | -11.023 +/- 26.005 | 0.554 |
| family | random_forest | 20.88 +/- 2.52 | 36.419 | 0.670 +/- 0.186 | 0.669 |
| family | gradient_boosting | 19.82 +/- 2.78 | 34.066 | 0.703 +/- 0.190 | 0.681 |

## Task A: poisson (141 features)

| scheme | model | MAE | RMSE | R2 | ln R2 |
|---|---|---|---|---|---|
| random | ridge | 0.14 +/- 0.00 | 0.201 | 0.109 +/- 0.017 | 0.109 |
| random | random_forest | 0.11 +/- 0.00 | 0.163 | 0.412 +/- 0.023 | 0.412 |
| random | gradient_boosting | 0.11 +/- 0.00 | 0.166 | 0.394 +/- 0.024 | 0.394 |
| cluster | ridge | 0.14 +/- 0.00 | 0.200 | 0.114 +/- 0.014 | 0.114 |
| cluster | random_forest | 0.11 +/- 0.01 | 0.162 | 0.422 +/- 0.023 | 0.422 |
| cluster | gradient_boosting | 0.11 +/- 0.01 | 0.165 | 0.395 +/- 0.024 | 0.395 |
| chemsys | ridge | 0.14 +/- 0.00 | 0.201 | 0.110 +/- 0.019 | 0.110 |
| chemsys | random_forest | 0.10 +/- 0.00 | 0.159 | 0.440 +/- 0.009 | 0.440 |
| chemsys | gradient_boosting | 0.11 +/- 0.00 | 0.163 | 0.417 +/- 0.007 | 0.417 |
| family | ridge | 0.15 +/- 0.01 | 0.209 | 0.034 +/- 0.095 | 0.034 |
| family | random_forest | 0.14 +/- 0.01 | 0.197 | 0.142 +/- 0.062 | 0.142 |
| family | gradient_boosting | 0.14 +/- 0.01 | 0.197 | 0.139 +/- 0.055 | 0.139 |

## Task B: binding_energy_zscan

| scheme | features | model | MAE | RMSE | R2 | ln R2 | Spearman |
|---|---|---|---|---|---|---|---|
| random | none | mean_arith | 18.51 +/- 0.41 | 33.569 | -0.000 +/- 0.000 | -0.206 | nan |
| random | none | mean_geom | 15.75 +/- 0.38 | 34.449 | -0.053 +/- 0.004 | -0.000 | nan |
| random | mono_basic | ridge | 12.03 +/- 0.45 | 28.548 | 0.278 +/- 0.039 | 0.515 | 0.744 |
| random | mono_basic | random_forest | 2.41 +/- 0.05 | 3.914 | 0.986 +/- 0.003 | 0.939 | 0.943 |
| random | mono_basic | gradient_boosting | 2.46 +/- 0.05 | 4.196 | 0.984 +/- 0.003 | 0.940 | 0.944 |
| random | mono_stiffness | ridge | 11.98 +/- 0.46 | 28.438 | 0.283 +/- 0.039 | 0.522 | 0.748 |
| random | mono_stiffness | random_forest | 2.41 +/- 0.05 | 3.898 | 0.986 +/- 0.003 | 0.939 | 0.943 |
| random | mono_stiffness | gradient_boosting | 2.43 +/- 0.06 | 4.019 | 0.986 +/- 0.003 | 0.940 | 0.944 |
| random | mono_stiffness_stacking | ridge | 11.96 +/- 0.46 | 28.396 | 0.285 +/- 0.039 | 0.523 | 0.749 |
| random | mono_stiffness_stacking | random_forest | 1.92 +/- 0.06 | 3.657 | 0.988 +/- 0.003 | 0.962 | 0.965 |
| random | mono_stiffness_stacking | gradient_boosting | 2.02 +/- 0.08 | 5.355 | 0.974 +/- 0.009 | 0.968 | 0.972 |
| monolayer | none | mean_arith | 18.50 +/- 1.83 | 33.154 | -0.005 +/- 0.004 | -0.215 | nan |
| monolayer | none | mean_geom | 15.75 +/- 2.07 | 33.975 | -0.054 +/- 0.010 | -0.001 | nan |
| monolayer | mono_basic | ridge | 14.71 +/- 2.52 | 34.301 | -0.051 +/- 0.382 | 0.265 | 0.628 |
| monolayer | mono_basic | random_forest | 8.98 +/- 1.52 | 23.281 | 0.490 +/- 0.149 | 0.649 | 0.842 |
| monolayer | mono_basic | gradient_boosting | 9.05 +/- 1.51 | 23.392 | 0.486 +/- 0.127 | 0.642 | 0.832 |
| monolayer | mono_stiffness | ridge | 14.69 +/- 2.32 | 33.026 | 0.024 +/- 0.228 | 0.262 | 0.630 |
| monolayer | mono_stiffness | random_forest | 8.92 +/- 1.42 | 23.058 | 0.502 +/- 0.135 | 0.655 | 0.846 |
| monolayer | mono_stiffness | gradient_boosting | 9.37 +/- 1.71 | 24.640 | 0.429 +/- 0.163 | 0.628 | 0.837 |
| monolayer | mono_stiffness_stacking | ridge | 14.65 +/- 2.31 | 32.886 | 0.032 +/- 0.223 | 0.266 | 0.630 |
| monolayer | mono_stiffness_stacking | random_forest | 8.39 +/- 1.45 | 22.965 | 0.506 +/- 0.128 | 0.675 | 0.865 |
| monolayer | mono_stiffness_stacking | gradient_boosting | 8.80 +/- 1.42 | 24.258 | 0.448 +/- 0.151 | 0.659 | 0.866 |
| family | none | mean_arith | 18.54 +/- 1.94 | 33.171 | -0.005 +/- 0.005 | -0.217 | nan |
| family | none | mean_geom | 15.77 +/- 2.26 | 33.992 | -0.054 +/- 0.015 | -0.003 | nan |
| family | mono_basic | ridge | 20.69 +/- 5.82 | 43.132 | -0.800 +/- 0.834 | -0.151 | 0.528 |
| family | mono_basic | random_forest | 13.52 +/- 3.91 | 30.165 | 0.173 +/- 0.157 | 0.337 | 0.720 |
| family | mono_basic | gradient_boosting | 13.39 +/- 3.99 | 29.884 | 0.185 +/- 0.169 | 0.339 | 0.711 |
| family | mono_stiffness | ridge | 20.69 +/- 4.81 | 45.417 | -1.096 +/- 1.315 | -0.162 | 0.513 |
| family | mono_stiffness | random_forest | 13.39 +/- 3.89 | 29.929 | 0.188 +/- 0.156 | 0.364 | 0.725 |
| family | mono_stiffness | gradient_boosting | 13.85 +/- 4.67 | 29.974 | 0.188 +/- 0.200 | 0.325 | 0.685 |
| family | mono_stiffness_stacking | ridge | 20.74 +/- 4.91 | 46.095 | -1.184 +/- 1.462 | -0.166 | 0.514 |
| family | mono_stiffness_stacking | random_forest | 13.23 +/- 4.01 | 29.753 | 0.201 +/- 0.155 | 0.382 | 0.740 |
| family | mono_stiffness_stacking | gradient_boosting | 13.92 +/- 5.03 | 30.078 | 0.188 +/- 0.224 | 0.346 | 0.706 |

## Task B: distance

| scheme | features | model | MAE | RMSE | R2 | ln R2 | Spearman |
|---|---|---|---|---|---|---|---|
| random | none | mean_arith | 0.42 +/- 0.01 | 0.551 | -0.001 +/- 0.002 | -0.009 | nan |
| random | none | mean_geom | 0.43 +/- 0.01 | 0.555 | -0.016 +/- 0.009 | -0.001 | nan |
| random | mono_basic | ridge | 0.28 +/- 0.01 | 0.371 | 0.546 +/- 0.019 | 0.531 | 0.715 |
| random | mono_basic | random_forest | 0.25 +/- 0.00 | 0.325 | 0.651 +/- 0.012 | 0.678 | 0.757 |
| random | mono_basic | gradient_boosting | 0.25 +/- 0.00 | 0.320 | 0.661 +/- 0.010 | 0.685 | 0.766 |
| random | mono_stiffness | ridge | 0.28 +/- 0.01 | 0.370 | 0.548 +/- 0.020 | 0.536 | 0.714 |
| random | mono_stiffness | random_forest | 0.25 +/- 0.00 | 0.325 | 0.651 +/- 0.012 | 0.678 | 0.757 |
| random | mono_stiffness | gradient_boosting | 0.25 +/- 0.00 | 0.322 | 0.659 +/- 0.011 | 0.684 | 0.764 |
| random | mono_stiffness_stacking | ridge | 0.28 +/- 0.01 | 0.368 | 0.554 +/- 0.020 | 0.540 | 0.720 |
| random | mono_stiffness_stacking | random_forest | 0.17 +/- 0.00 | 0.231 | 0.824 +/- 0.007 | 0.799 | 0.885 |
| random | mono_stiffness_stacking | gradient_boosting | 0.15 +/- 0.00 | 0.212 | 0.852 +/- 0.009 | 0.811 | 0.909 |
| monolayer | none | mean_arith | 0.42 +/- 0.01 | 0.551 | -0.000 +/- 0.000 | -0.008 | nan |
| monolayer | none | mean_geom | 0.43 +/- 0.01 | 0.555 | -0.015 +/- 0.004 | -0.000 | nan |
| monolayer | mono_basic | ridge | 0.31 +/- 0.03 | 0.422 | 0.407 +/- 0.137 | 0.448 | 0.678 |
| monolayer | mono_basic | random_forest | 0.29 +/- 0.02 | 0.385 | 0.509 +/- 0.086 | 0.512 | 0.728 |
| monolayer | mono_basic | gradient_boosting | 0.28 +/- 0.01 | 0.367 | 0.554 +/- 0.044 | 0.543 | 0.747 |
| monolayer | mono_stiffness | ridge | 0.31 +/- 0.03 | 0.421 | 0.410 +/- 0.125 | 0.446 | 0.676 |
| monolayer | mono_stiffness | random_forest | 0.29 +/- 0.02 | 0.385 | 0.508 +/- 0.088 | 0.512 | 0.727 |
| monolayer | mono_stiffness | gradient_boosting | 0.27 +/- 0.01 | 0.360 | 0.572 +/- 0.048 | 0.564 | 0.754 |
| monolayer | mono_stiffness_stacking | ridge | 0.31 +/- 0.03 | 0.420 | 0.412 +/- 0.129 | 0.449 | 0.683 |
| monolayer | mono_stiffness_stacking | random_forest | 0.21 +/- 0.03 | 0.329 | 0.638 +/- 0.123 | 0.588 | 0.828 |
| monolayer | mono_stiffness_stacking | gradient_boosting | 0.20 +/- 0.02 | 0.303 | 0.696 +/- 0.045 | 0.630 | 0.852 |
| family | none | mean_arith | 0.42 +/- 0.01 | 0.550 | -0.000 +/- 0.000 | -0.008 | nan |
| family | none | mean_geom | 0.43 +/- 0.01 | 0.554 | -0.015 +/- 0.005 | -0.001 | nan |
| family | mono_basic | ridge | 0.36 +/- 0.05 | 0.500 | 0.168 +/- 0.108 | 0.272 | 0.606 |
| family | mono_basic | random_forest | 0.33 +/- 0.05 | 0.442 | 0.333 +/- 0.235 | 0.272 | 0.676 |
| family | mono_basic | gradient_boosting | 0.32 +/- 0.02 | 0.431 | 0.385 +/- 0.056 | 0.323 | 0.690 |
| family | mono_stiffness | ridge | 0.36 +/- 0.05 | 0.496 | 0.180 +/- 0.159 | 0.284 | 0.605 |
| family | mono_stiffness | random_forest | 0.33 +/- 0.05 | 0.447 | 0.321 +/- 0.220 | 0.220 | 0.670 |
| family | mono_stiffness | gradient_boosting | 0.32 +/- 0.02 | 0.434 | 0.375 +/- 0.080 | 0.291 | 0.673 |
| family | mono_stiffness_stacking | ridge | 0.36 +/- 0.05 | 0.497 | 0.174 +/- 0.159 | 0.281 | 0.607 |
| family | mono_stiffness_stacking | random_forest | 0.30 +/- 0.05 | 0.426 | 0.377 +/- 0.240 | 0.284 | 0.711 |
| family | mono_stiffness_stacking | gradient_boosting | 0.27 +/- 0.02 | 0.393 | 0.489 +/- 0.052 | 0.391 | 0.744 |

## Task B: binding_energy_gs

| scheme | features | model | MAE | RMSE | R2 | ln R2 | Spearman |
|---|---|---|---|---|---|---|---|
| random | none | mean_arith | 6.23 +/- 0.12 | 8.925 | -0.001 +/- 0.001 | -0.056 | nan |
| random | none | mean_geom | 5.98 +/- 0.16 | 9.084 | -0.036 +/- 0.007 | -0.000 | nan |
| random | mono_stiffness | ridge | 3.07 +/- 0.14 | 5.595 | 0.598 +/- 0.137 | 0.693 | 0.839 |
| random | mono_stiffness | random_forest | 2.37 +/- 0.07 | 3.142 | 0.875 +/- 0.019 | 0.804 | 0.882 |
| random | mono_stiffness | gradient_boosting | 2.34 +/- 0.07 | 3.070 | 0.881 +/- 0.016 | 0.807 | 0.884 |
| monolayer | none | mean_arith | 6.23 +/- 0.12 | 8.913 | -0.000 +/- 0.000 | -0.055 | nan |
| monolayer | none | mean_geom | 5.98 +/- 0.12 | 9.072 | -0.036 +/- 0.003 | -0.000 | nan |
| monolayer | mono_stiffness | ridge | 3.41 +/- 0.30 | 6.073 | 0.529 +/- 0.142 | 0.627 | 0.803 |
| monolayer | mono_stiffness | random_forest | 2.95 +/- 0.24 | 4.943 | 0.688 +/- 0.140 | 0.683 | 0.833 |
| monolayer | mono_stiffness | gradient_boosting | 2.97 +/- 0.23 | 4.887 | 0.694 +/- 0.143 | 0.681 | 0.832 |
| family | none | mean_arith | 6.22 +/- 0.32 | 8.845 | -0.003 +/- 0.003 | -0.059 | nan |
| family | none | mean_geom | 5.98 +/- 0.38 | 8.996 | -0.037 +/- 0.013 | -0.001 | nan |
| family | mono_stiffness | ridge | 4.31 +/- 0.71 | 7.818 | 0.164 +/- 0.380 | 0.480 | 0.749 |
| family | mono_stiffness | random_forest | 3.58 +/- 0.44 | 5.705 | 0.581 +/- 0.134 | 0.542 | 0.740 |
| family | mono_stiffness | gradient_boosting | 3.59 +/- 0.33 | 5.734 | 0.575 +/- 0.116 | 0.532 | 0.741 |

## Task A, Y2D: LightGBM, five seeds (mean and std over seeds of the fold means)

| scheme | mae_mean | mae_sd | r2_mean | r2_log_mean |
|---|---|---|---|---|
| chemsys | 13.859 | 0.065 | 0.863 | 0.806 |
| family | 19.901 | 0.215 | 0.712 | 0.679 |
| random | 13.798 | 0.108 | 0.872 | 0.802 |

## Task B: LightGBM on mono_stiffness, five seeds

| target | scheme | mae_mean | mae_sd | r2_mean | r2_log_mean | spearman_mean |
|---|---|---|---|---|---|---|
| binding_energy_zscan | family | 13.589 | 0.301 | 0.200 | 0.351 | 0.707 |
| binding_energy_zscan | monolayer | 9.069 | 0.424 | 0.483 | 0.643 | 0.833 |
| binding_energy_zscan | random | 2.422 | 0.010 | 0.985 | 0.940 | 0.944 |
| distance | family | 0.316 | 0.002 | 0.391 | 0.310 | 0.684 |
| distance | monolayer | 0.278 | 0.005 | 0.513 | 0.398 | 0.751 |
| distance | random | 0.248 | 0.001 | 0.660 | 0.688 | 0.764 |
