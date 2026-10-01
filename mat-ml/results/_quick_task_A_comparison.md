# Task A results (seed 42)

Metrics are means over folds, in original units (N/m for Y2D). `x(s/random)` = MAE under scheme s divided by MAE under random CV.

## Random vs cluster-stratified vs grouped

| target | model | MAE random | MAE cluster | MAE chemsys | MAE family | R2 random | R2 cluster | R2 chemsys | R2 family | MAE x(cluster/random) | MAE x(chemsys/random) | MAE x(family/random) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Y2D | gradient_boosting | 16.806 | 16.868 | 16.737 | 20.602 | 0.804 | 0.800 | 0.802 | 0.676 | 1.004 | 0.996 | 1.226 |
| Y2D | random_forest | 14.811 | 14.885 | 15.081 | 21.184 | 0.848 | 0.837 | 0.827 | 0.662 | 1.005 | 1.018 | 1.430 |
| Y2D | ridge | 22.505 | 22.573 | 22.758 | 31.118 | 0.686 | 0.685 | 0.672 | -11.023 | 1.003 | 1.011 | 1.383 |
| poisson | gradient_boosting | 0.119 | 0.119 | 0.119 | 0.140 | 0.341 | 0.343 | 0.347 | 0.153 | 0.997 | 0.998 | 1.174 |
| poisson | random_forest | 0.109 | 0.108 | 0.106 | 0.141 | 0.402 | 0.412 | 0.431 | 0.127 | 0.993 | 0.977 | 1.301 |
| poisson | ridge | 0.144 | 0.144 | 0.144 | 0.152 | 0.109 | 0.114 | 0.110 | 0.034 | 0.997 | 1.001 | 1.055 |

## Leakage: fraction of test rows whose key also occurs in training (mean over folds)

| scheme | chemsys_in_train | formula_in_train | family_in_train |
|---|---|---|---|
| random | 0.608 | 0.357 | 0.958 |
| cluster | 0.609 | 0.355 | 0.959 |
| chemsys | 0.000 | 0.000 | 0.958 |
| family | 0.590 | 0.336 | 0.000 |

## External check (train on all labelled C2DB, test on JARVIS)

| target | scheme | model | mae_mean | rmse_mean | r2_mean | r2_log_mean |
|---|---|---|---|---|---|---|
| Y2D | external_jarvis_all | ridge | 27.764 | 49.539 | 0.245 | 0.241 |
| Y2D | external_jarvis_unseen_chemsys | ridge | 41.294 | 61.517 | -0.121 | -0.316 |
| Y2D | external_jarvis_all | random_forest | 20.693 | 43.483 | 0.419 | 0.457 |
| Y2D | external_jarvis_unseen_chemsys | random_forest | 34.909 | 54.650 | 0.115 | 0.244 |
| Y2D | external_jarvis_all | gradient_boosting | 22.788 | 44.387 | 0.394 | 0.497 |
| Y2D | external_jarvis_unseen_chemsys | gradient_boosting | 29.740 | 52.037 | 0.198 | 0.420 |
| poisson | external_jarvis_all | ridge | 0.126 | 0.187 | 0.027 | 0.027 |
| poisson | external_jarvis_unseen_chemsys | ridge | 0.124 | 0.155 | 0.152 | 0.152 |
| poisson | external_jarvis_all | random_forest | 0.100 | 0.175 | 0.149 | 0.149 |
| poisson | external_jarvis_unseen_chemsys | random_forest | 0.126 | 0.162 | 0.079 | 0.079 |
| poisson | external_jarvis_all | gradient_boosting | 0.109 | 0.174 | 0.159 | 0.159 |
| poisson | external_jarvis_unseen_chemsys | gradient_boosting | 0.113 | 0.143 | 0.277 | 0.277 |

## Run info

```json
{
  "task": "A",
  "seed": 42,
  "c2db_cleaning": {
    "with_stiffness": 8462,
    "finite_tensor": 8462,
    "positive_definite": 7474,
    "asymmetry_ok": 7258,
    "Y2D_positive": 7258
  },
  "jarvis_cleaning": {
    "with_tensor_string": 229,
    "corrupt_tensor": 18,
    "stable_xx_yy_block": 186
  },
  "rows_per_target": {
    "Y2D": {
      "c2db_rows": 7258,
      "jarvis_rows": 186
    },
    "poisson": {
      "c2db_rows": 7229,
      "jarvis_rows": 185
    }
  },
  "leakage_first_target": {
    "random": {
      "chemsys_in_train": 0.608431726575953,
      "formula_in_train": 0.35726173456892085,
      "family_in_train": 0.9575655053131402
    },
    "cluster": {
      "chemsys_in_train": 0.6089873422528018,
      "formula_in_train": 0.3551945746545082,
      "family_in_train": 0.9589426309963869
    },
    "chemsys": {
      "chemsys_in_train": 0.0,
      "formula_in_train": 0.0,
      "family_in_train": 0.9578392786963679
    },
    "family": {
      "chemsys_in_train": 0.5901140659144544,
      "formula_in_train": 0.3359173781547066,
      "family_in_train": 0.0
    }
  },
  "n_features": 141,
  "targets": {
    "Y2D": "log",
    "poisson": "none"
  },
  "convention": "n_layers=1 for all rows; Y2D (total) == Y2D_per_layer",
  "quick": true,
  "versions": {
    "python": "3.11.9",
    "sklearn": "1.9.1",
    "numpy": "2.4.6",
    "pandas": "2.3.3"
  },
  "runtime_s": 344.3
}
```