# Calibration tables (auto-generated)

Database disagreement yardstick (C2DB vs JARVIS, 106 materials): c11 MAE 8.77, c12 MAE 6.52, c22 MAE 9.32, Y2D MAE 8.62, poisson MAE 0.107
Gate MAE threshold for Y2D: 12.93 N/m

## Gate (Y2D, all computed structures)
| model            | variant      | subset   |   n |   spearman |   mae |   mae_threshold | pass_spearman   | pass_mae   | GATE_PASS   |
|:-----------------|:-------------|:---------|----:|-----------:|------:|----------------:|:----------------|:-----------|:------------|
| chgnet_0.3.0     | relaxed_cell | all      | 334 |      0.787 | 33.8  |            12.9 | False           | False      | False       |
| chgnet_0.3.0     | relaxed_cell | c2db     | 150 |      0.83  | 23.4  |            12.9 | False           | False      | False       |
| chgnet_0.3.0     | relaxed_cell | jarvis   | 184 |      0.71  | 42.4  |            12.9 | False           | False      | False       |
| mace_mp0_medium  | relaxed_cell | all      | 334 |      0.82  | 28.1  |            12.9 | False           | False      | False       |
| mace_mp0_medium  | relaxed_cell | c2db     | 150 |      0.827 | 19.3  |            12.9 | False           | False      | False       |
| mace_mp0_medium  | relaxed_cell | jarvis   | 184 |      0.763 | 35.4  |            12.9 | False           | False      | False       |
| mace_mpa0_medium | relaxed_cell | all      | 334 |      0.874 | 47.1  |            12.9 | False           | False      | False       |
| mace_mpa0_medium | relaxed_cell | c2db     | 150 |      0.939 |  9.91 |            12.9 | True            | True       | True        |
| mace_mpa0_medium | relaxed_cell | jarvis   | 184 |      0.813 | 77.5  |            12.9 | False           | False      | False       |

## Metrics, all computed structures
| model            | variant      | subset   | filter       | quantity   |   n |    mae |    rmse |          r2 |   spearman |   slope_through_origin |    bias |   median_ratio |
|:-----------------|:-------------|:---------|:-------------|:-----------|----:|-------:|--------:|------------:|-----------:|-----------------------:|--------:|---------------:|
| chgnet_0.3.0     | relaxed_cell | all      | all_computed | c11        | 334 | 32.2   |  49.9   |    0.403    |      0.827 |                  0.54  | -30.3   |          0.572 |
| chgnet_0.3.0     | relaxed_cell | all      | all_computed | c12        | 334 |  8.61  |  12.8   |    0.598    |      0.79  |                  0.782 |  -4.34  |        nan     |
| chgnet_0.3.0     | relaxed_cell | all      | all_computed | c22        | 334 | 31.3   |  44.4   |    0.54     |      0.83  |                  0.591 | -29.4   |          0.573 |
| chgnet_0.3.0     | relaxed_cell | all      | all_computed | Y2D        | 334 | 33.8   |  55.4   |    0.159    |      0.787 |                  0.484 | -30.1   |          0.533 |
| chgnet_0.3.0     | relaxed_cell | all      | all_computed | poisson    | 334 |  0.268 |   0.951 |  -19.8      |      0.518 |                  1.19  |   0.157 |        nan     |
| chgnet_0.3.0     | relaxed_cell | c2db     | all_computed | c11        | 150 | 23.2   |  38.1   |    0.655    |      0.854 |                  0.589 | -20.9   |          0.586 |
| chgnet_0.3.0     | relaxed_cell | c2db     | all_computed | c12        | 150 |  6.89  |   9.57  |    0.76     |      0.73  |                  0.779 |  -3.75  |          0.676 |
| chgnet_0.3.0     | relaxed_cell | c2db     | all_computed | c22        | 150 | 24.3   |  39     |    0.702    |      0.826 |                  0.621 | -22.5   |          0.586 |
| chgnet_0.3.0     | relaxed_cell | c2db     | all_computed | Y2D        | 150 | 23.4   |  38.8   |    0.589    |      0.83  |                  0.539 | -21.2   |          0.543 |
| chgnet_0.3.0     | relaxed_cell | c2db     | all_computed | poisson    | 150 |  0.219 |   0.536 |   -4.99     |      0.616 |                  1.23  |   0.108 |          1.16  |
| chgnet_0.3.0     | relaxed_cell | jarvis   | all_computed | c11        | 184 | 39.5   |  57.7   |    0.091    |      0.771 |                  0.514 | -37.9   |          0.567 |
| chgnet_0.3.0     | relaxed_cell | jarvis   | all_computed | c12        | 184 | 10     |  15     |    0.437    |      0.796 |                  0.784 |  -4.82  |        nan     |
| chgnet_0.3.0     | relaxed_cell | jarvis   | all_computed | c22        | 184 | 37.1   |  48.3   |    0.3      |      0.798 |                  0.57  | -35.1   |          0.573 |
| chgnet_0.3.0     | relaxed_cell | jarvis   | all_computed | Y2D        | 184 | 42.4   |  66     |   -0.334    |      0.71  |                  0.455 | -37.4   |          0.512 |
| chgnet_0.3.0     | relaxed_cell | jarvis   | all_computed | poisson    | 184 |  0.308 |   1.19  |  -34.2      |      0.432 |                  1.16  |   0.197 |        nan     |
| mace_mp0_medium  | relaxed_cell | all      | all_computed | c11        | 334 | 24.6   |  42.3   |    0.569    |      0.865 |                  0.65  | -21.5   |          0.738 |
| mace_mp0_medium  | relaxed_cell | all      | all_computed | c12        | 334 |  8.1   |  16.4   |    0.348    |      0.829 |                  0.863 |  -3.08  |        nan     |
| mace_mp0_medium  | relaxed_cell | all      | all_computed | c22        | 334 | 23.9   |  35.9   |    0.7      |      0.871 |                  0.693 | -21.2   |          0.735 |
| mace_mp0_medium  | relaxed_cell | all      | all_computed | Y2D        | 334 | 28.1   |  61.5   |   -0.036    |      0.82  |                  0.62  | -17     |          0.715 |
| mace_mp0_medium  | relaxed_cell | all      | all_computed | poisson    | 334 |  0.216 |   0.632 |   -8.17     |      0.536 |                  0.8   |   0.002 |        nan     |
| mace_mp0_medium  | relaxed_cell | c2db     | all_computed | c11        | 150 | 19.3   |  32.1   |    0.755    |      0.877 |                  0.702 | -14.8   |          0.725 |
| mace_mp0_medium  | relaxed_cell | c2db     | all_computed | c12        | 150 |  7.22  |  11.3   |    0.667    |      0.696 |                  0.886 |  -2.23  |          0.849 |
| mace_mp0_medium  | relaxed_cell | c2db     | all_computed | c22        | 150 | 19.2   |  31.7   |    0.803    |      0.867 |                  0.708 | -17     |          0.732 |
| mace_mp0_medium  | relaxed_cell | c2db     | all_computed | Y2D        | 150 | 19.3   |  33.3   |    0.696    |      0.827 |                  0.661 | -13.7   |          0.693 |
| mace_mp0_medium  | relaxed_cell | c2db     | all_computed | poisson    | 150 |  0.235 |   0.594 |   -6.37     |      0.538 |                  0.822 |   0.016 |          1.18  |
| mace_mp0_medium  | relaxed_cell | jarvis   | all_computed | c11        | 184 | 29     |  49.1   |    0.341    |      0.833 |                  0.623 | -26.9   |          0.742 |
| mace_mp0_medium  | relaxed_cell | jarvis   | all_computed | c12        | 184 |  8.81  |  19.5   |    0.043    |      0.858 |                  0.851 |  -3.78  |        nan     |
| mace_mp0_medium  | relaxed_cell | jarvis   | all_computed | c22        | 184 | 27.6   |  39     |    0.545    |      0.848 |                  0.683 | -24.6   |          0.743 |
| mace_mp0_medium  | relaxed_cell | jarvis   | all_computed | Y2D        | 184 | 35.4   |  77.3   |   -0.829    |      0.763 |                  0.599 | -19.6   |          0.733 |
| mace_mp0_medium  | relaxed_cell | jarvis   | all_computed | poisson    | 184 |  0.199 |   0.662 |   -9.94     |      0.515 |                  0.78  |  -0.01  |        nan     |
| mace_mpa0_medium | relaxed_cell | all      | all_computed | c11        | 334 | 14.5   |  32     |    0.754    |      0.9   |                  0.835 |  -8.48  |          0.911 |
| mace_mpa0_medium | relaxed_cell | all      | all_computed | c12        | 334 |  6.28  |  10.4   |    0.738    |      0.851 |                  0.917 |  -1.06  |        nan     |
| mace_mpa0_medium | relaxed_cell | all      | all_computed | c22        | 334 | 13.4   |  21.6   |    0.891    |      0.906 |                  0.884 |  -7.69  |          0.909 |
| mace_mpa0_medium | relaxed_cell | all      | all_computed | Y2D        | 334 | 47.1   | 607     |  -99.7      |      0.874 |                  0.375 | -40.5   |          0.916 |
| mace_mpa0_medium | relaxed_cell | all      | all_computed | poisson    | 334 |  0.453 |   5.5   | -693        |      0.665 |                  1.79  |   0.327 |        nan     |
| mace_mpa0_medium | relaxed_cell | c2db     | all_computed | c11        | 150 | 11.1   |  19.4   |    0.911    |      0.942 |                  0.885 |  -6.46  |          0.889 |
| mace_mpa0_medium | relaxed_cell | c2db     | all_computed | c12        | 150 |  5.67  |   9.74  |    0.751    |      0.811 |                  0.861 |  -0.999 |          0.852 |
| mace_mpa0_medium | relaxed_cell | c2db     | all_computed | c22        | 150 | 10.8   |  17.9   |    0.937    |      0.942 |                  0.892 |  -6.9   |          0.904 |
| mace_mpa0_medium | relaxed_cell | c2db     | all_computed | Y2D        | 150 |  9.91  |  17.7   |    0.915    |      0.939 |                  0.886 |  -5.62  |          0.905 |
| mace_mpa0_medium | relaxed_cell | c2db     | all_computed | poisson    | 150 |  0.178 |   0.637 |   -7.46     |      0.721 |                  1.04  |   0.064 |          0.969 |
| mace_mpa0_medium | relaxed_cell | jarvis   | all_computed | c11        | 184 | 17.2   |  39.4   |    0.576    |      0.845 |                  0.809 | -10.1   |          0.928 |
| mace_mpa0_medium | relaxed_cell | jarvis   | all_computed | c12        | 184 |  6.78  |  10.9   |    0.705    |      0.852 |                  0.947 |  -1.1   |        nan     |
| mace_mpa0_medium | relaxed_cell | jarvis   | all_computed | c22        | 184 | 15.5   |  24.2   |    0.824    |      0.875 |                  0.879 |  -8.34  |          0.918 |
| mace_mpa0_medium | relaxed_cell | jarvis   | all_computed | Y2D        | 184 | 77.5   | 817     | -204        |      0.813 |                  0.106 | -68.9   |          0.921 |
| mace_mpa0_medium | relaxed_cell | jarvis   | all_computed | poisson    | 184 |  0.678 |   7.39  |   -1.36e+03 |      0.602 |                  2.47  |   0.542 |        nan     |

## Metrics, ML-stable structures only (Y2D)
| model            | variant      | subset   | filter         | quantity   |   n |   mae |   rmse |     r2 |   spearman |   slope_through_origin |   bias |   median_ratio |
|:-----------------|:-------------|:---------|:---------------|:-----------|----:|------:|-------:|-------:|-----------:|-----------------------:|-------:|---------------:|
| chgnet_0.3.0     | relaxed_cell | all      | ml_stable_only | Y2D        | 306 |  32.8 |   50.8 |  0.323 |      0.801 |                  0.49  | -30.5  |          0.543 |
| chgnet_0.3.0     | relaxed_cell | c2db     | ml_stable_only | Y2D        | 137 |  23.9 |   39.6 |  0.599 |      0.835 |                  0.542 | -21.6  |          0.543 |
| chgnet_0.3.0     | relaxed_cell | jarvis   | ml_stable_only | Y2D        | 169 |  40.1 |   58.3 | -0.022 |      0.715 |                  0.463 | -37.8  |          0.544 |
| mace_mp0_medium  | relaxed_cell | all      | ml_stable_only | Y2D        | 315 |  24.7 |   43.5 |  0.498 |      0.862 |                  0.6   | -21.7  |          0.712 |
| mace_mp0_medium  | relaxed_cell | c2db     | ml_stable_only | Y2D        | 138 |  18.9 |   32.9 |  0.722 |      0.889 |                  0.659 | -15.6  |          0.679 |
| mace_mp0_medium  | relaxed_cell | jarvis   | ml_stable_only | Y2D        | 177 |  29.2 |   50.2 |  0.235 |      0.799 |                  0.57  | -26.5  |          0.731 |
| mace_mpa0_medium | relaxed_cell | all      | ml_stable_only | Y2D        | 320 |  13.3 |   28.9 |  0.771 |      0.897 |                  0.836 |  -6.69 |          0.916 |
| mace_mpa0_medium | relaxed_cell | c2db     | ml_stable_only | Y2D        | 141 |   9.6 |   16.9 |  0.926 |      0.942 |                  0.888 |  -5.19 |          0.904 |
| mace_mpa0_medium | relaxed_cell | jarvis   | ml_stable_only | Y2D        | 179 |  16.2 |   35.6 |  0.596 |      0.856 |                  0.808 |  -7.86 |          0.921 |

## Failures / instabilities
| model            | variant      |   n |   exceptions |   ref_relax_not_converged |   strain_not_converged |   ml_unstable |   asym_gt_10pct |
|:-----------------|:-------------|----:|-------------:|--------------------------:|-----------------------:|--------------:|----------------:|
| chgnet_0.3.0     | relaxed_cell | 336 |            2 |                         1 |                      1 |            28 |              44 |
| mace_mp0_medium  | relaxed_cell | 336 |            2 |                         4 |                      4 |            19 |              16 |
| mace_mpa0_medium | relaxed_cell | 336 |            2 |                         2 |                      0 |            14 |              11 |

## Runtime and memory
| model            | variant      |   median_s |   p90_s |   total_h |   peak_gpu_mb |
|:-----------------|:-------------|-----------:|--------:|----------:|--------------:|
| chgnet_0.3.0     | relaxed_cell |       8.26 |    30.8 |      1.31 |           150 |
| mace_mp0_medium  | relaxed_cell |       5.82 |    25.1 |      1.18 |           188 |
| mace_mpa0_medium | relaxed_cell |       6.85 |    24   |      1.04 |           224 |

## Lattice constant error of the potential's own equilibrium (a, %)
| model            |   median |   mean |   p90_abs |
|:-----------------|---------:|-------:|----------:|
| chgnet_0.3.0     |    0.252 |  0.257 |      4.06 |
| mace_mp0_medium  |    0.503 |  0.749 |      5.33 |
| mace_mpa0_medium |    0.535 |  0.78  |      3.7  |

## Worst Y2D errors (relaxed_cell)
| model            | id             | formula      | family       |   Y2D_ref |     Y2D_ml |   rel_err_Y2D | ml_stable   |
|:-----------------|:---------------|:-------------|:-------------|----------:|-----------:|--------------:|:------------|
| mace_mpa0_medium | JVASP-27912    | FeO2         | nan          |    103    |   -1.1e+04 |       -108    | False       |
| mace_mp0_medium  | JVASP-14456    | CoO2         | nan          |     19.8  |  751       |         36.9  | False       |
| mace_mp0_medium  | 3CuI2-1        | CuI2         | AB2|lg70     |      9.71 |  103       |          9.65 | False       |
| mace_mp0_medium  | 4FTl-4         | TlF          | AB|lg28      |      9.97 |   83.2     |          7.34 | False       |
| mace_mp0_medium  | JVASP-6097     | VCl3         | nan          |      2.64 |   22       |          7.34 | True        |
| mace_mp0_medium  | JVASP-75120    | CI2          | nan          |     30.9  |  249       |          7.05 | False       |
| mace_mpa0_medium | JVASP-6097     | VCl3         | nan          |      2.64 |   21.2     |          7.05 | True        |
| chgnet_0.3.0     | 6AgF-1         | AgF          | AB|lg31      |      2.41 |   17.7     |          6.33 | True        |
| chgnet_0.3.0     | 2MoI2O2-1      | Mo(IO)2      | AB2C2|lg40   |      8.77 |   56.6     |          5.45 | True        |
| mace_mpa0_medium | 1BeP2H4S4-1    | BeP2(HS)4    | AB2C4D4|lg10 |     14.1  |  -61.2     |         -5.35 | False       |
| mace_mpa0_medium | JVASP-27754    | Ni(HO)2      | nan          |     12.3  |   65.3     |          4.3  | True        |
| chgnet_0.3.0     | 1HfZrPd2I4S4-1 | HfZrPd2(SI)4 | ABC2D4E4|lg3 |     20.4  |  -64.9     |         -4.18 | False       |
| chgnet_0.3.0     | JVASP-14457    | VO2          | nan          |    110    | -280       |         -3.53 | False       |
| chgnet_0.3.0     | JVASP-6097     | VCl3         | nan          |      2.64 |   11.7     |          3.42 | True        |
| mace_mpa0_medium | JVASP-14459    | BiO2         | nan          |     13.4  |   57.5     |          3.29 | True        |
| mace_mp0_medium  | JVASP-27754    | Ni(HO)2      | nan          |     12.3  |   49.7     |          3.04 | True        |
| chgnet_0.3.0     | JVASP-27754    | Ni(HO)2      | nan          |     12.3  |   46.3     |          2.76 | True        |
| mace_mpa0_medium | JVASP-14456    | CoO2         | nan          |     19.8  |   74.1     |          2.74 | True        |
| mace_mp0_medium  | JVASP-9020     | Rb(NdSe)2    | nan          |     13.9  |   50.4     |          2.63 | True        |
| chgnet_0.3.0     | JVASP-7057     | Ba(LiSi)2    | nan          |      6.89 |  -10.8     |         -2.56 | False       |
| mace_mp0_medium  | 6AgF-1         | AgF          | AB|lg31      |      2.41 |    8.29    |          2.44 | True        |
| chgnet_0.3.0     | JVASP-28153    | HgF          | nan          |      2.46 |    8.2     |          2.33 | False       |
| mace_mpa0_medium | JVASP-765      | TiO2         | nan          |      6.88 |   -5.69    |         -1.83 | False       |
| mace_mpa0_medium | JVASP-9020     | Rb(NdSe)2    | nan          |     13.9  |   39.2     |          1.82 | True        |