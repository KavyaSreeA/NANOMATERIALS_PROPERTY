# Task A results (seed 42)

Metrics are means over folds, in original units (N/m for Y2D). `x(s/random)` = MAE under scheme s divided by MAE under random CV.

## Random vs cluster-stratified vs grouped

| target | model | MAE random | MAE cluster | MAE chemsys | MAE family | R2 random | R2 cluster | R2 chemsys | R2 family | MAE x(cluster/random) | MAE x(chemsys/random) | MAE x(family/random) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Y2D | gradient_boosting | 13.736 | 13.981 | 13.914 | 19.817 | 0.873 | 0.865 | 0.864 | 0.703 | 1.018 | 1.013 | 1.443 |
| Y2D | random_forest | 14.671 | 14.718 | 14.822 | 20.876 | 0.849 | 0.842 | 0.834 | 0.670 | 1.003 | 1.010 | 1.423 |
| Y2D | ridge | 22.505 | 22.573 | 22.758 | 31.118 | 0.686 | 0.685 | 0.672 | -11.023 | 1.003 | 1.011 | 1.383 |
| poisson | gradient_boosting | 0.111 | 0.111 | 0.109 | 0.140 | 0.394 | 0.395 | 0.417 | 0.139 | 0.998 | 0.985 | 1.267 |
| poisson | random_forest | 0.107 | 0.107 | 0.105 | 0.140 | 0.412 | 0.422 | 0.440 | 0.142 | 0.992 | 0.976 | 1.305 |
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
| Y2D | external_jarvis_all | random_forest | 20.665 | 43.021 | 0.431 | 0.482 |
| Y2D | external_jarvis_unseen_chemsys | random_forest | 34.952 | 53.619 | 0.148 | 0.284 |
| Y2D | external_jarvis_all | gradient_boosting | 18.974 | 41.138 | 0.480 | 0.548 |
| Y2D | external_jarvis_unseen_chemsys | gradient_boosting | 28.837 | 48.766 | 0.295 | 0.459 |
| poisson | external_jarvis_all | ridge | 0.126 | 0.187 | 0.027 | 0.027 |
| poisson | external_jarvis_unseen_chemsys | ridge | 0.124 | 0.155 | 0.152 | 0.152 |
| poisson | external_jarvis_all | random_forest | 0.096 | 0.171 | 0.187 | 0.187 |
| poisson | external_jarvis_unseen_chemsys | random_forest | 0.119 | 0.152 | 0.187 | 0.187 |
| poisson | external_jarvis_all | gradient_boosting | 0.103 | 0.172 | 0.179 | 0.179 |
| poisson | external_jarvis_unseen_chemsys | gradient_boosting | 0.119 | 0.154 | 0.163 | 0.163 |

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
  "quick": false,
  "versions": {
    "python": "3.11.9",
    "sklearn": "1.9.1",
    "numpy": "2.4.6",
    "pandas": "2.3.3"
  },
  "runtime_s": 1219.3
}
```