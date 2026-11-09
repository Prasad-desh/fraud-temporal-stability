# Does Feature Selection Survive Temporal Drift?

A stability-aware, cost-sensitive evaluation of feature selection for credit-card fraud detection.

**Prasad Deshpande** — PRN 1032240177

---

## Problem

Credit-card fraud detection under extreme class imbalance: only 492 of 284,807 transactions are fraudulent (0.173%). A classifier that labels everything legitimate scores 99.83% accuracy and catches nothing, so accuracy cannot be used to choose a model.

The dataset's features `V1`–`V28` are PCA components. PCA ranks directions by variance in the input space, which says nothing about their usefulness for predicting the label. Supervised feature selection is what answers that question — and this study asks whether the answer holds up over time.

## Approach

One method from each feature-selection family, paired with a classifier, each run with and without selection:

| # | Feature selection | Family | Classifier |
|---|---|---|---|
| 1 | Mutual Information (`SelectKBest`, k=15) | Filter | Logistic Regression |
| 2 | Recursive Feature Elimination (k=15) | Wrapper | Random Forest |
| 3 | XGBoost gain importance (top 15) | Embedded | XGBoost |

## Research questions

- **RQ1** — How much optimism does a random split introduce versus a temporal split?
- **RQ2** — Is the effect of feature selection distinguishable from fold-to-fold noise?
- **RQ3** — How stable are the three selector families across resamples and across time?
- **RQ4** — Under an amount-weighted cost model, does selection change the money saved?

## Method

```
load -> drop Time -> two evaluation protocols
     |
     +-- RANDOM:    15 folds (5-fold x 3 repeats), stratified
     +-- TEMPORAL:  train on day 1 (144,786 txns) -> test on day 2 (140,021 txns)

per fold / per protocol:
   1. fit StandardScaler on Amount      (training data only)
   2. fit feature selector              (original, still-imbalanced training data)
   3. apply SMOTE                       (training data only)
   4. train classifier
   5. evaluate on untouched data at its natural fraud rate
```

**Leakage control.** The scaler and selector are fitted on training data only, before SMOTE. SMOTE never touches evaluation data. The decision threshold is tuned on a held-out 25% slice of day 1, never on day 2.

**Cost model.** A missed fraud costs the transaction amount; every alert raised costs a fixed review fee. The baseline is doing nothing, which loses the full value of all fraud on day 2. Results are reported as percentage of fraud value saved, swept over review fees of 1 / 3 / 10 / 30.

**Statistics.** Wilcoxon signed-rank tests on 15 paired folds; selection stability by the Kuncheva index (chance-corrected) and Jaccard overlap.

## Key results

Full run: 15 CV folds plus the temporal protocol, about 24 hours on an Apple Silicon MacBook Air.

**RQ1 — random splits are optimistic.** Every one of the 29 features differs significantly between day 1 and day 2 (KS test, p < 0.01; V3 reaches KS = 0.52), and fraud prevalence falls from 0.194% to 0.151%.

| Model | PR-AUC (random CV) | PR-AUC (temporal) | Optimism |
|---|---|---|---|
| Logistic Regression | 0.7331 ± 0.0320 | 0.7818 | −6.2% |
| Random Forest | 0.8239 ± 0.0312 | 0.7351 | **+12.1%** |
| XGBoost | 0.8550 ± 0.0258 | 0.7563 | **+13.0%** |

**RQ2 — feature selection costs precision, not recall.** PR-AUC fell significantly for all three models; recall was statistically unchanged for the tree models.

| Model | Metric | Δ (selected − all) | Wilcoxon p | Significant |
|---|---|---|---|---|
| Logistic Regression | PR-AUC | −0.0113 | 0.030 | yes |
| Random Forest | PR-AUC | −0.0067 | 0.030 | yes |
| XGBoost | PR-AUC | −0.0188 | 0.0004 | yes |
| Random Forest | recall | −0.0007 | 0.600 | no |
| XGBoost | recall | −0.0040 | 0.561 | no |
| Random Forest | F1 | −0.1007 | 0.0001 | yes |
| XGBoost | F1 | −0.1026 | 0.0001 | yes |

**RQ3 — the cheapest selector is also the most reproducible.**

| Selector | Kuncheva (across folds) | Kuncheva (day 1 vs day 2) | Distinct features touched |
|---|---|---|---|
| MI (filter) | **0.926** | **0.862** | 16 |
| RFE (wrapper) | 0.636 | 0.310 | 25 |
| XGB (embedded) | 0.563 | 0.448 | 24 |

**RQ4 — threshold tuning matters more than model choice.** The cost-optimal threshold for Logistic Regression is 0.9999, not 0.5, and moving to it lifts savings from 23.1% to 60.4%. Once tuned, all six configurations land in a narrow 55–60% band.

## Repository contents

```
fraud_temporal_stability_study.py     full study (the code that produced results/)
results/                              full run: 9 tables (CSV) + 4 figures (PNG)
requirements-lock.txt                 exact package versions used
```

### Result files

| File | Contents |
|---|---|
| `drift_ks.csv` | KS statistic per feature, day 1 vs day 2 |
| `cv_raw.csv` | Per-fold raw metrics (90 evaluation rows) |
| `t1_cv_summary.csv` | CV metrics, mean ± SD |
| `t2_wilcoxon.csv` | Signed-rank tests (RQ2) |
| `t3_stability.csv` | Selection stability across folds (RQ3) |
| `t4_selection_frequency.csv` | How often each feature was chosen |
| `t5_temporal.csv` | Temporal protocol, default vs cost-tuned threshold |
| `t6_protocol_gap.csv` | Random vs temporal optimism (RQ1) |
| `t7_cross_day_stability.csv` | Selection stability across the day boundary |
| `t8_cost_sensitivity.csv` | Savings vs review fee (RQ4) |
| `t9_headline.csv` | Summary table |
| `f1_drift.png` … `f4_cost.png` | Figures |

## Reproducing

The dataset is **not** in this repository (144 MB, above GitHub's file size limit). Download `creditcard.csv` from [Kaggle — Credit Card Fraud Detection (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the repository root.

```bash
python -m venv .venv && source .venv/bin/activate
pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib scipy

python fraud_temporal_stability_study.py --quick    # ~10 min smoke test
python fraud_temporal_stability_study.py            # full run (hours)
```

The script verifies the CSV has all 284,807 rows and exits if it is truncated — a partial download loads without error and silently produces meaningless results.

`random_state=42` throughout, so the run is reproducible.

## Limitations

Two days of transactions from one European issuer in 2013. "Day 2" is a single drift observation, not a trend. PCA features cannot be interpreted individually. The cost model assumes a flat review fee and total loss on a missed fraud.

## References

1. A. Dal Pozzolo, O. Caelen, R. A. Johnson, and G. Bontempi, "Calibrating probability with undersampling for unbalanced classification," *Proc. IEEE SSCI*, 2015, pp. 159–166.
2. A. Dal Pozzolo, G. Boracchi, O. Caelen, C. Alippi, and G. Bontempi, "Credit card fraud detection: A realistic modeling and a novel learning strategy," *IEEE Trans. Neural Netw. Learn. Syst.*, vol. 29, no. 8, pp. 3784–3797, 2018.
3. N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic minority over-sampling technique," *J. Artif. Intell. Res.*, vol. 16, pp. 321–357, 2002.
4. L. I. Kuncheva, "A stability index for feature selection," *Proc. IASTED Int. Conf. Artif. Intell. Appl.*, 2007, pp. 390–395.
5. S. Nogueira, K. Sechidis, and G. Brown, "On the stability of feature selection algorithms," *J. Mach. Learn. Res.*, vol. 18, no. 174, pp. 1–54, 2018.
6. L. Breiman, "Random forests," *Mach. Learn.*, vol. 45, no. 1, pp. 5–32, 2001.
7. T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," *Proc. ACM SIGKDD*, 2016, pp. 785–794.
