# JARVIS external check, seeds [42, 43, 44, 45, 46]

Fit on all clean C2DB rows (ln Y2D), predict JARVIS Y2D (N/m). mae_sd = std across model seeds; [lo, hi] = 95% bootstrap interval over JARVIS rows on the seed-averaged prediction.

## Models vs JARVIS

| model | features | subset | n | mae | mae_sd | mae_lo | mae_hi | r2_log | r2_log_lo | r2_log_hi | spearman | spearman_lo | spearman_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gradient_boosting | all | all | 186 | 18.984 | 0.399 | 14.435 | 24.493 | 0.546 | 0.335 | 0.728 | 0.819 | 0.722 | 0.901 |
| gradient_boosting | all | matched | 106 | 10.200 | 0.387 | 7.890 | 12.487 | 0.809 | 0.668 | 0.926 | 0.931 | 0.852 | 0.978 |
| gradient_boosting | all | unseen_chemsys | 55 | 29.126 | 1.084 | 20.458 | 39.628 | 0.450 | -0.135 | 0.747 | 0.730 | 0.512 | 0.888 |
| gradient_boosting | composition | all | 186 | 25.561 | 0.455 | 20.620 | 31.423 | 0.344 | 0.112 | 0.532 | 0.709 | 0.592 | 0.805 |
| gradient_boosting | composition | matched | 106 | 14.239 | 0.408 | 11.673 | 17.087 | 0.752 | 0.612 | 0.863 | 0.906 | 0.825 | 0.954 |
| gradient_boosting | composition | unseen_chemsys | 55 | 37.473 | 1.313 | 27.505 | 48.958 | 0.059 | -0.651 | 0.427 | 0.555 | 0.294 | 0.750 |
| gradient_boosting | structure | all | 186 | 27.162 | 0.318 | 21.628 | 33.282 | 0.377 | 0.126 | 0.576 | 0.720 | 0.607 | 0.817 |
| gradient_boosting | structure | matched | 106 | 22.373 | 0.574 | 17.941 | 26.928 | 0.593 | 0.365 | 0.774 | 0.780 | 0.644 | 0.891 |
| gradient_boosting | structure | unseen_chemsys | 55 | 32.554 | 0.261 | 21.781 | 45.363 | 0.412 | -0.061 | 0.674 | 0.644 | 0.416 | 0.826 |
| random_forest | all | all | 186 | 20.680 | 0.092 | 15.948 | 26.602 | 0.481 | 0.225 | 0.670 | 0.785 | 0.673 | 0.872 |
| random_forest | all | matched | 106 | 9.761 | 0.116 | 7.581 | 12.201 | 0.808 | 0.668 | 0.923 | 0.927 | 0.841 | 0.977 |
| random_forest | all | unseen_chemsys | 55 | 35.183 | 0.347 | 25.953 | 47.133 | 0.274 | -0.478 | 0.619 | 0.692 | 0.452 | 0.859 |

## Label agreement: C2DB label vs JARVIS label on matched materials (not model skill)

| mae | r2 | r2_log | mae_log | spearman | n | median_ratio_jarvis_over_c2db |
|---|---|---|---|---|---|---|
| 8.621 | 0.899 | 0.745 | 0.192 | 0.919 | 106.000 | 1.039 |

runtime 335s