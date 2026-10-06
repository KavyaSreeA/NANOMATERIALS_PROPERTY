# Phase D2 design: which inputs does the model use?  (written BEFORE any D2 number was computed)

Date written: 2026-10-06. Earlier results are not modified; outputs of this phase are prefixed `attribution_`.

## Question
The ablation (feature groups removed, retrained) showed that composition and geometry carry complementary signal and that the space-group number is nearly irrelevant. It did not say how the *final* model uses individual inputs, and no per-feature attribution was run.
This phase measures how the trained LightGBM uses feature groups, under random and under family-grouped evaluation. It describes the model, not the physics.

## Design
Model and data exactly as Task A: LightGBM (500 trees, lr 0.05, 31 leaves, 80% rows, 50% columns), 141 features, ln Y2D, 7,258 clean C2DB monolayers; 5 folds under random, chemical-system and family schemes; seeds 42, 43, 44.
**Groups (fixed now).** The 22 Magpie element properties (each group = all six statistics of that property) and the individual structural inputs a, b, gamma, area_per_atom, nat, z_extent, spg_number, n_elements (n_layers is constant and is excluded). That is 30 groups.
**Method 1, permutation importance:** for each outer fold, fit on the training rows; on the held-out rows permute all columns of a group *jointly* (one row permutation shared by the group's columns), five repeats;
importance = MAE(permuted) - MAE(unpermuted) in N/m, back-transformed to N/m. Share = importance / sum of positive importances.
**Method 2, TreeSHAP:** LightGBM `pred_contrib` on the held-out rows (ln space); importance = mean |contribution| summed over a group's columns; share = importance / sum.
Summary over the 15 fold-fits per scheme: mean and standard deviation; no resampling of materials is done (stated limit).

## Hypotheses (fixed now)
- **A1.** The model uses the same signals under random and family evaluation: Spearman correlation of group permutation importance between the random and family schemes >= 0.70.
- **A2.** No single group carries more than 40% of the total permutation importance under the family scheme (the model does not rely on one shortcut).
- **A3.** The space-group number carries less than 5% of the total permutation importance under both schemes (consistent with the ablation: removing it cost 0.3 N/m).
- **A4 (method check).** The permutation and TreeSHAP group rankings agree: Spearman >= 0.70 under each scheme. If not, both are reported and the ranking is called method-dependent.
- **A5 (exploratory, no gate).** The combined share of the six geometry inputs (a, b, gamma, area_per_atom, nat, z_extent) is larger under the family scheme than under the random scheme, because composition statistics can act as a family fingerprint under random splits.

## What each outcome would mean
A1 fails: the family-held-out model leans on different inputs, which would be a further sign that random-split scores reward family recognition. A2 fails: a shortcut exists and should be named. A3 fails: the ablation understated the role of symmetry in the final model.

## Limits stated in advance
Correlated inputs share credit (permutation importance of one member of a correlated group is understated, SHAP spreads credit), permuting breaks correlations between a group and the rest of the inputs and produces unrealistic rows, group boundaries are my choice,
one model, one target, three seeds. Importance shows how the model predicts, not what physically determines stiffness; nothing here supports a causal statement.

## Reproducibility
`python -m src.attribution` (needs the project data and cache; about minutes on CPU) writes `results/attribution_fold.csv`, `attribution_summary.csv`, `attribution_hypotheses.json`, `attribution_report.md` and `results/figures/fig_attribution.png`.
