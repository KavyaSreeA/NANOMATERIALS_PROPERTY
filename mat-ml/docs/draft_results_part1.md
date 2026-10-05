# DRAFT: Results, Part 1 — Machine-learning prediction of monolayer stiffness under family-aware evaluation

Status: working draft written from the saved outputs in `results/` (file names given in brackets). Figure and table numbers are placeholders.
Every number below is a mean over the stated folds/seeds unless an interval is given. "Y2D" is the in-plane 2D Young's modulus in N/m,
Y2D = (C11 C22 - C12^2)/C22, from the C2DB elastic tensor; "ln R2" is the R2 of ln(Y2D).

## 3.1 Dataset and cleaning
We used the Computational 2D Materials Database (C2DB; GPAW/PBE) as the training set and JARVIS-DFT 2D (VASP/OptB88vdW) as an independent
external set. Of 16,905 C2DB materials, 8,462 have a stiffness tensor (8,430 in the current file layout and 32 in an older flat layout that a first pass missed).
Mechanical stability was determined from the tensor itself, not from database labels: the symmetrised (xx, yy, xy) tensor had to be positive definite
(7,474 materials) and its asymmetry max|C - C^T|/max|C| had to be at most 10% (7,258 materials); all retained materials have Y2D > 0.
The final set contains 3,806 chemical systems (2,572 of them with a single member) and 676 structure families (anonymous formula plus layer group; the largest has 432 members).
Polymorphs were kept as separate entries. For JARVIS, 229 materials have an elastic tensor; 18 are corrupt (NaN or denormal entries) and, using the xx-yy block only
(C11, C22 > 0 and C11 C22 - C12^2 > 0), 186 are usable. The JARVIS shear entries were not used because they do not reproduce (C11 - C12)/2 for MoS2.
Stiffness values are in N/m throughout; C2DB and JARVIS agree on shared materials (106 matched; median ratio 1.04, MAE 8.6 N/m on Y2D), which supports reading the
JARVIS values in the same unit. [`data_audit.json`, `external_ci_report.md`]

## 3.2 Features, models and evaluation protocol
Inputs were composition and geometry only: 132 Magpie composition statistics (matminer), lattice lengths a and b, the in-plane angle, area per atom,
number of atoms, slab thickness, a constant layer count, the space-group number, and the number of elements (141 features). DFT outputs (energy above hull,
formation energy, stability labels, band gap) were excluded because they are not available for a new material. Targets were fitted on the natural-log scale for Y2D.
Models were Ridge regression, a random forest (300 trees) and LightGBM (500 trees); no hyper-parameter search was performed. Five-fold cross-validation was run under four splitting schemes:
random; "cluster" (K-means on the composition features, then stratified folds, so that every cluster appears in training and test); grouped by chemical system; and grouped by structure family.
Grouped folds were balanced on ten quantile bins of the target but kept groups disjoint. We report the fraction of test materials whose chemical system, formula or family also occurs
in the training fold (Table 1), because this is what distinguishes the schemes. [`task_A_comparison.md`]

**Table 1. Overlap between test and training folds (fraction of test materials; seed 42).**

| Scheme | chemical system in train | formula in train | family in train |
|---|---|---|---|
| random | 0.61 | 0.36 | 0.96 |
| cluster | 0.61 | 0.36 | 0.96 |
| chemical-system grouped | 0.00 | 0.00 | 0.96 |
| family grouped | 0.59 | 0.34 | 0.00 |

## 3.3 Random versus grouped evaluation
Under random cross-validation LightGBM predicted Y2D with MAE 13.8 N/m (ln R2 0.80; R2 0.87). When whole structure families were held out, the MAE rose to 19.9 N/m
(ln R2 0.68; R2 0.70), a factor of 1.44 (95% cluster-bootstrap interval 1.30-1.62, resampling whole structure families; the standard deviation over five random seeds, 0.02, understates the uncertainty because it holds the dataset fixed). The random forest behaved the same way
(14.6 to 20.8 N/m; ratio 1.43 +/- 0.01). Holding out chemical systems, by contrast, made no measurable difference: MAE 13.9 N/m, a ratio of 1.00 +/- 0.01, even though 61% of
the random-split test materials share their chemical system with the training data. The cluster-stratified split was likewise indistinguishable from random (14.0 N/m).
In this dataset, therefore, random splits overestimate accuracy through overlap of structure families, not of chemistry (Table 2). Ridge regression was clearly worse
(MAE 22.5 N/m under random CV) and unstable under extrapolation (one family-grouped fold gave R2 = -57.5 in N/m after back-transformation, while its ln R2 was 0.36).
The reported standard deviations come from re-seeding the splits and the models on a fixed dataset; they do not include the uncertainty from the finite number of materials.

**Table 2. Y2D (N/m), LightGBM, all 141 features; five seeds.**

| Scheme | MAE | ln R2 |
|---|---|---|
| random | 13.8 +/- 0.1 | 0.802 |
| chemical-system grouped | 13.9 +/- 0.1 | 0.806 |
| family grouped | 19.9 +/- 0.2 | 0.679 |

Two feature-free baselines put these numbers in context. Predicting the training mean gives MAE 42.4 N/m. Predicting the mean of the training members of the same structure family
gives MAE 27.0 N/m (ln R2 0.46) under random and chemical-system splits and, by construction, 42.4 N/m for unseen families. Thus much of the apparent skill of random splits can be
obtained from the family label alone, but not for unseen families. [`robustness_summary.csv`]

## 3.4 What carries the signal
Feature-group ablations (LightGBM, five seeds; Table 3) show that composition and geometry contribute different information: either group alone gives MAE of about 19-20 N/m under random CV
(composition 19.2, structure 19.8), and together 13.8 N/m. Removing the space-group number from the full set changed the MAE by only +0.3 N/m (14.1 versus 13.8), and the space-group number alone was
nearly uninformative (38.0 N/m versus 42.4 for the mean), so the structural signal comes from the geometric descriptors rather than from the symmetry class. Offering the layer group and the
anonymous formula as one-hot features did not close the family gap: adding both to the full set changed the MAE by -0.2 N/m under random and family splits (13.6 and 19.7 N/m), within one to three seed
standard deviations, and the ratio stayed at 1.45. Used alone, these prototype descriptors reproduced the family-mean baseline under random CV (ln R2 0.46) and carried no information for unseen
families (ln R2 0.06). The family gap is therefore not an artefact of a missing descriptor: held-out prototypes are genuinely harder.
[`robustness_summary.csv`, `spg_ablation_summary.csv`, `proto_features_summary.csv`]

**Table 3. Feature ablation, LightGBM, Y2D MAE (N/m), five seeds.**

| Features | random | chemical system | family | family/random |
|---|---|---|---|---|
| all (141) | 13.8 | 13.9 | 19.9 | 1.44 |
| composition only | 19.2 | 18.6 | 25.5 | 1.33 |
| structure only | 19.8 | 20.0 | 26.6 | 1.35 |
| geometry without space-group number | 20.9 | - | 26.3 | 1.26 |
| space-group number only | 38.0 | - | 44.3 | 1.16 |
| all without space-group number | 14.1 | - | 20.2 | 1.44 |
| all + layer group | 13.6 | - | 19.8 | 1.45 |
| all + anonymous formula | 13.8 | - | 20.0 | 1.46 |
| all + both | 13.6 | - | 19.7 | 1.45 |

## 3.5 Poisson's ratio
The Poisson ratio nu = C12/C22 (restricted to |nu| <= 1, 7,229 materials) was much less predictable: LightGBM reached MAE 0.11 and R2 0.39 under random CV and 0.14 and 0.14 with families held out
(random forest: 0.11 and 0.41; 0.14 and 0.14). Ridge regression explained little (R2 0.11 and 0.03). [`task_A_comparison.md`]

## 3.6 External validation on JARVIS
Models trained on all C2DB materials were applied to the 186 usable JARVIS materials (Table 4; intervals are 95% bootstrap intervals over JARVIS materials for the seed-averaged prediction, i.e. they
reflect the size of the JARVIS sample, not the training data). Over all 186 materials LightGBM gave MAE 19.0 N/m [14.4, 24.5], ln R2 0.55 [0.34, 0.73] and Spearman rho 0.82 [0.72, 0.90].
For the 55 JARVIS materials whose chemical system never occurs in the (clean) C2DB set, performance was lower and much less certain: MAE 29.1 N/m [20.5, 39.6], ln R2 0.45 [-0.14, 0.75]
and rho 0.73 [0.51, 0.89]. Composition-only features performed worst on these materials (ln R2 0.06 [-0.65, 0.43]; rho 0.56 [0.29, 0.75]) and structure-only features better (0.41 [-0.06, 0.67]; rho 0.64 [0.42, 0.83]);
the intervals overlap, so this ranking should be read as suggestive. The ranking of materials by stiffness is thus transferred reasonably well, while absolute values are not,
and the point estimate for unseen chemistry has an interval that includes ln R2 = 0. Two caveats apply. First, even for the same material the two databases disagree (MAE 8.6 N/m on 106
matched materials; Spearman 0.92), which sets a floor on achievable agreement that is not a model error. Second, JARVIS and C2DB use different codes, functionals and structure sets, and
one JARVIS reference value (HgI2, 386 N/m) appears implausible, so part of the external error reflects reference noise. [`external_ci_report.md`]

**Table 4. External check on JARVIS (train: all C2DB; 95% bootstrap intervals over JARVIS materials).**

| Model / features | Subset (n) | MAE (N/m) | ln R2 | Spearman |
|---|---|---|---|---|
| LightGBM / all | all (186) | 19.0 [14.4, 24.5] | 0.55 [0.34, 0.73] | 0.82 [0.72, 0.90] |
| LightGBM / all | unseen chemistry (55) | 29.1 [20.5, 39.6] | 0.45 [-0.14, 0.75] | 0.73 [0.51, 0.89] |
| LightGBM / composition | unseen chemistry (55) | 37.5 [27.5, 49.0] | 0.06 [-0.65, 0.43] | 0.56 [0.29, 0.75] |
| LightGBM / structure | unseen chemistry (55) | 32.6 [21.8, 45.4] | 0.41 [-0.06, 0.67] | 0.64 [0.42, 0.83] |
| Random forest / all | unseen chemistry (55) | 35.2 [26.0, 47.1] | 0.27 [-0.48, 0.62] | 0.69 [0.45, 0.86] |

## 3.7 Limitations
(i) A single training database (C2DB, one code and functional) and a small external set. (ii) Standard deviations over seeds understate the total uncertainty; no resampling of the materials was done for the cross-validation results.
(iii) Family folds are uneven in size because a few families are very large. (iv) Polymorphs are retained, which is appropriate for a screening setting but means that identical compositions can occur in train and test under random splits.
(v) Hyper-parameters were not tuned and only tabular models were tested; whether graph neural networks would narrow the family gap is untested. (vi) The mechanical-stability filter removes about 14% of materials with tensors,
so the models describe stable materials only. (vii) All results concern monolayers; nothing here addresses bilayers or stacking.

## Suggested one-paragraph summary for the abstract (draft)
On 7,258 mechanically stable C2DB monolayers, a gradient-boosted model predicts the 2D Young's modulus with MAE 13.8 N/m under random cross-validation but 19.9 N/m (a factor 1.44) when entire structure
families are held out; holding out chemical systems has no effect. Layer-group and anonymous-formula descriptors do not remove the gap. On an independent set of 186 JARVIS materials the ranking is
reproduced (Spearman 0.82), but errors roughly double for chemistries absent from the training set (MAE 29 N/m, 95% CI 20-40).
