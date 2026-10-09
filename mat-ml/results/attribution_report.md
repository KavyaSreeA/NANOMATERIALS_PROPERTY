# D2: feature-group attribution of the Task A LightGBM (auto-generated; design in `attribution_design.md`)

Permutation importance = increase in held-out MAE (N/m) when a group is permuted jointly; share = importance / sum of positive importances. TreeSHAP share = mean |contribution| (ln space). Mean over 15 fold-fits per scheme.

## random split, top 12 groups

| group | columns | permutation dMAE (N/m) | sd | perm. share | SHAP share |
|---|---|---|---|---|---|
| area_per_atom | 1 | 17.54 | 1.03 | 32.1% | 19.6% |
| magpie:NdUnfilled | 6 | 4.55 | 0.72 | 8.3% | 8.4% |
| magpie:MeltingT | 6 | 4.09 | 0.36 | 7.5% | 7.2% |
| a | 1 | 3.87 | 0.44 | 7.1% | 7.2% |
| magpie:NUnfilled | 6 | 3.79 | 0.34 | 6.9% | 6.4% |
| b | 1 | 1.97 | 0.36 | 3.6% | 2.9% |
| spg_number | 1 | 1.82 | 0.27 | 3.3% | 3.2% |
| z_extent | 1 | 1.65 | 0.20 | 3.0% | 2.5% |
| magpie:CovalentRadius | 6 | 1.58 | 0.45 | 2.9% | 3.2% |
| nat | 1 | 1.53 | 0.39 | 2.8% | 3.0% |
| magpie:NpUnfilled | 6 | 1.39 | 0.30 | 2.6% | 2.9% |
| magpie:Column | 6 | 1.35 | 0.33 | 2.5% | 3.0% |

## chemsys split, top 12 groups

| group | columns | permutation dMAE (N/m) | sd | perm. share | SHAP share |
|---|---|---|---|---|---|
| area_per_atom | 1 | 17.65 | 1.02 | 32.2% | 19.7% |
| magpie:NdUnfilled | 6 | 4.33 | 0.52 | 7.9% | 8.1% |
| magpie:MeltingT | 6 | 4.12 | 0.42 | 7.5% | 7.2% |
| a | 1 | 4.05 | 0.59 | 7.4% | 7.3% |
| magpie:NUnfilled | 6 | 3.93 | 0.80 | 7.2% | 6.6% |
| b | 1 | 1.93 | 0.32 | 3.5% | 2.9% |
| spg_number | 1 | 1.83 | 0.18 | 3.3% | 3.2% |
| z_extent | 1 | 1.69 | 0.17 | 3.1% | 2.5% |
| magpie:CovalentRadius | 6 | 1.55 | 0.38 | 2.8% | 3.3% |
| nat | 1 | 1.53 | 0.32 | 2.8% | 3.0% |
| magpie:Column | 6 | 1.37 | 0.25 | 2.5% | 3.1% |
| magpie:NpUnfilled | 6 | 1.30 | 0.20 | 2.4% | 2.8% |

## family split, top 12 groups

| group | columns | permutation dMAE (N/m) | sd | perm. share | SHAP share |
|---|---|---|---|---|---|
| area_per_atom | 1 | 14.72 | 4.08 | 38.8% | 19.5% |
| magpie:NdUnfilled | 6 | 3.29 | 0.98 | 8.7% | 8.1% |
| magpie:MeltingT | 6 | 2.84 | 0.68 | 7.5% | 7.3% |
| a | 1 | 2.77 | 0.70 | 7.3% | 7.7% |
| magpie:NUnfilled | 6 | 2.76 | 1.54 | 7.3% | 6.8% |
| magpie:CovalentRadius | 6 | 1.27 | 0.64 | 3.3% | 3.4% |
| b | 1 | 1.25 | 0.84 | 3.3% | 2.9% |
| z_extent | 1 | 0.88 | 0.44 | 2.3% | 2.6% |
| magpie:Column | 6 | 0.78 | 0.44 | 2.0% | 3.1% |
| magpie:NpUnfilled | 6 | 0.77 | 0.25 | 2.0% | 2.9% |
| nat | 1 | 0.74 | 0.43 | 2.0% | 2.8% |
| spg_number | 1 | 0.74 | 0.30 | 2.0% | 2.9% |

## Pre-specified tests

| test | outcome | passed |
|---|---|---|
| A1 Spearman(random, family importance) >= 0.70 | 0.97 | True |
| A2 top family-scheme group share <= 40% | area_per_atom: 38.8% | True |
| A3 spg_number share < 5% in every scheme | random 3.3%, chemsys 3.3%, family 2.0% | True |
| A4 permutation vs SHAP rank >= 0.70 | random 0.91, chemsys 0.91, family 0.91 | True |
| A5 (exploratory) geometry share | random 50.0%, chemsys 50.2%, family 54.1%; family > random: True | - |
