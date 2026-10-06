# D1 Part A: dose-response of the family penalty (auto-generated; design in `mechanism_design.md`)

Rows 7258; families 676; chemical systems 3806. Seed-42 out-of-fold MAE (N/m): random 13.74, chemsys 13.91, family 19.82.

## Penalty ratio Q = sum(grouped error) / sum(random error), by size of the material's group

| axis | group size | rows | groups | Q [95% CI] |
|---|---|---|---|---|
| family | 1 | 264 | 264 | 0.99 [0.93, 1.06] |
| family | 2-3 | 306 | 127 | 1.22 [1.05, 1.38] |
| family | 4-9 | 741 | 131 | 1.24 [1.14, 1.37] |
| family | 10-49 | 2769 | 133 | 1.47 [1.36, 1.60] |
| family | >=50 | 3178 | 21 | 1.53 [1.25, 1.83] |
| chemical system | 1 | 2572 | 2572 | 0.98 [0.96, 1.01] |
| chemical system | 2-3 | 1998 | 857 | 1.04 [1.00, 1.07] |
| chemical system | 4-9 | 1693 | 299 | 1.03 [0.98, 1.09] |
| chemical system | >=10 | 995 | 78 | 0.99 [0.95, 1.04] |

## Pre-specified tests

| test | outcome | passed |
|---|---|---|
| M1 singletons Q in [0.90, 1.10], CI contains 1 | Q = 0.99 [0.93, 1.06] | True |
| M2 Q(n>=10) - Q(n<=3) >= 0.15, CI above 0 | 1.50 - 1.10 = 0.40 [0.20, 0.64] | True |
| M3 Spearman(log n, per-family Q) > 0.2, CI above 0 | rho = 0.26 [0.17, 0.35] (337 families) | True |
| M4 chemistry Q in [0.90, 1.15] in every bin with >= 100 rows | see table | True |
| M5 (descriptive) | families with n >= 10 hold 82% of rows and 93% of the total penalty | - |
