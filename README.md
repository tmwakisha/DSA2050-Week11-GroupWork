# DSA 2050A — Week 9: Model Selection & Defensible Evaluation

**Repository:** [tmwakisha/DSA2050-Week11-GroupWork](https://github.com/tmwakisha/DSA2050-Week11-GroupWork)
**Task:** Predict customer churn from a synthetic teaching dataset; compare models fairly; inspect errors, subgroup behavior, and model reliance; freeze the choice before a one-time final evaluation.

> **Submission-format note:** The course handout specifies existing groups and one repository per group. This work has been completed as a solo working copy at the user's request. Confirm with the instructor that an individual submission is acceptable before submitting. Do not invent a group number or student ID: add the correct student ID below if required.

## Author and contribution

| Name | Student ID | Contribution |
|---|---|---|
| Tamara Mwakisha | **Add student ID before submission** | Development-data inspection; leakage prevention; pipeline and CV; model comparison; error and subgroup analysis; permutation importance; rationale; final evaluation; notebook and documentation. |

## Goal and data

The target is `churn` (`1` = cancelled, `0` = stayed). One row represents one customer. The data are synthetic and for teaching only; they do not represent real Kenyan telecom customers. The development dataset contains **2,720 rows and 11 columns**, with **910 churners (33.46%)** and **1,810 retained customers (66.54%)**. Missing values occur in `monthly_charges_kes` (156), `support_tickets_90d` (120), and `region` (93). The development data contain no exact duplicate rows.

The final dataset was kept sealed during exploration, preprocessing, cross-validation, candidate selection, and rationale writing. The selection was recorded in `reports/frozen_selection.json` before the final evaluation. The final set was then used once to evaluate that fixed choice.

## Prediction features and leakage exclusions

The model uses these eight candidate features: `tenure_months`, `monthly_charges_kes`, `support_tickets_90d`, `service_calls_90d`, `plan_type`, `region`, `autopay`, and `customer_segment`. Their availability at a real operational prediction time would need to be verified.

- `customer_id` is excluded because it is an identifier, not a defensible generalizable predictor; including it risks memorization.
- `cancellation_confirmed` is excluded because it is post-outcome leakage. In the development data it exactly maps `Active` to `churn=0` and `Cancelled` to `churn=1`; using it would reveal the target rather than predict it.
- `churn` is the outcome and is not included among predictors.

## Preprocessing and evaluation design

Numeric variables use median imputation followed by standard scaling. Categorical variables use most-frequent imputation followed by one-hot encoding. These steps are inside a scikit-learn `Pipeline`/`ColumnTransformer`, so transformations are fitted using each training fold only and cannot learn from that fold's validation observations.

All candidates used the same `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` folds and F1-based scoring. The dummy baseline predicts the majority class: its five-fold churn F1 scores are all **0.000**. Predicting “stayed” for everyone could yield accuracy near the 66.5% stay rate, while catching no churners; this demonstrates why accuracy alone is inadequate here.

## Cross-validation comparison

The selected configuration is the best mean-F1 configuration in each model family. `reports/experiment_log.csv` contains every tested configuration and its five fold scores.

| Model | Setting | Mean F1 | F1 SD | Fold F1 scores (1–5) | Mean precision | Mean recall |
|---|---:|---:|---:|---|---:|---:|
| Logistic regression | `C=1` | 0.4701 | 0.0527 | 0.3915, 0.5342, 0.4718, 0.4345, 0.5185 | 0.6243 | 0.3780 |
| Decision tree | `max_depth=3` | 0.3985 | 0.0487 | 0.3920, 0.4571, 0.3158, 0.4367, 0.3906 | 0.5656 | 0.3176 |

Logistic regression with `C=1` is selected. Its mean F1 exceeds the best tree's by about **0.0716** (0.072 rounded), materially more than 0.005. Its fold SD is slightly higher (0.0527 vs 0.0487), so it is not the more stable candidate by this estimate; the higher mean F1 and recall still support it as the strongest of the tested candidates. Logistic regression at `C=10` has the same displayed mean, SD, precision, recall, and fold results; grid search selects `C=1`, and the results provide no evidence that the less-regularized setting is better. This is a choice among the small prescribed grids, not proof of deployment readiness.

## Error analysis and customer segments

The selected model's out-of-fold development classification report gives churn recall of approximately **0.378**. It produces **566 false negatives** (actual churners predicted to stay) and **205 false positives** (retained customers flagged as churn risks), out of 910 churners and 1,810 non-churners respectively. False negatives are operationally important because the provider would miss an opportunity to intervene; false positives can consume retention resources unnecessarily. The out-of-fold confusion matrix is shown in [`reports/development_confusion_matrix.png`](reports/development_confusion_matrix.png).

| Customer segment | Customers | Actual churners | Recall | Precision | False negatives |
|---|---:|---:|---:|---:|---:|
| Established | 2,268 | 649 | 0.2157 | 0.5932 | 509 |
| New | 452 | 261 | 0.7816 | 0.6518 | 57 |

Recall is much lower for Established than New customers in these out-of-fold predictions. The difference is concerning even though both groups have substantial observed counts; it must be interpreted as a finding in this synthetic dataset and analysis, not a guaranteed real-population disparity. Do not hide this aggregate-performance limitation or generalize beyond the data.

## Permutation importance

On the held-out 25% development slice, permutation importance was measured by the mean decrease in F1 after shuffling a feature (five repeats). The two largest reductions are:

| Feature | Mean F1 reduction | SD |
|---|---:|---:|
| `tenure_months` | 0.0787 | 0.0237 |
| `support_tickets_90d` | 0.0350 | 0.0157 |

These are measures of predictive reliance for this fitted model and validation slice, not causal effects. Correlated predictors may share or mask importance, and the estimates can vary with the sample and fitted model. See [`reports/permutation_importance.csv`](reports/permutation_importance.csv) and [`reports/permutation_importance.png`](reports/permutation_importance.png).

## Frozen selection rationale

Before opening the final test set, the selected model and hyperparameters were frozen as **logistic regression, `C=1`**, with F1 as the primary metric. Its CV mean F1 is **0.4701 (SD 0.0527)** versus **0.3985 (SD 0.0487)** for the best decision tree. The stronger average performance supports the selection, but the fold variation is slightly greater and out-of-fold churn recall is weak: 566 of 910 churners are missed. Established customers have particularly low recall (0.2157; 509 of 649 actual churners missed), versus 0.7816 for New customers. Logistic regression is comparatively simple and regularized, but should not be treated as deployment-ready. The freeze record, including full rationale, selected parameters, seed, and scikit-learn version, is [`reports/frozen_selection.json`](reports/frozen_selection.json).

## One-time final evaluation

After freezing the decision, the chosen pipeline was refit on all development data and evaluated once on **680** final-test observations.

| Metric | Final-test result |
|---|---:|
| Accuracy | 0.7000 |
| F1 | 0.4486 |
| Precision | 0.5845 |
| Recall | 0.3640 |

Confusion matrix uses labels `[0, 1]` (rows = actual, columns = predicted):

|  | Predicted stayed (0) | Predicted churn (1) |
|---|---:|---:|
| Actual stayed (0) | 393 | 59 |
| Actual churn (1) | 145 | 83 |

Final-test F1 (0.4486) is 0.0215 below mean CV F1 (0.4701), a modest absolute difference that is broadly consistent with the CV estimate, while final recall remains low (0.3640). The test result is reported as an evaluation of the frozen model, not used to revise the hyperparameters or model selection. Machine-readable results are in [`reports/final_metrics.json`](reports/final_metrics.json).

## Written reflection

1. **Why share folds?** Comparing models on identical stratified folds holds the validation partitions constant, reducing the chance that a model appears better merely because it received easier validation data. Stratification preserves approximate class proportions in each fold.
2. **What does F1 measure?** F1 is the harmonic mean of precision and recall. It rewards finding churners while penalizing excessive false alarms; it does not encode the business costs of either error and can hide subgroup differences.
3. **Where do errors concentrate?** In development out-of-fold predictions, 566 churners are missed versus 205 false alarms. Established customers account for 509 of the 566 false negatives and have recall 0.2157. On the final test, there are 145 false negatives and 59 false positives.
4. **What is the highest-reliance feature?** `tenure_months` has the greatest mean F1 reduction when shuffled (0.0787), followed by `support_tickets_90d` (0.0350). This predictive reliance is not evidence of causation.
5. **Why is the final test not a tuning tool?** Repeated model or hyperparameter choices based on the final score adapt the model to that test set and make its score an optimistic, no-longer-independent estimate. The final score is for evaluating the frozen choice; issues found should be documented as limitations.
6. **What are two limitations, and what should happen before deployment?** The dataset is synthetic and may not represent real Kenyan customer behavior or measurement processes; the lab also does not specify a real intervention, error-cost trade-off, prediction-time data contract, or temporal change. Before deployment, validate on representative, time-separated real data; verify feature timing and leakage; assess subgroup performance, calibration, privacy, and intervention costs; and choose an operating threshold using development data only.

## Files and how to reproduce

```text
.
├── README.md
├── requirements.txt
├── data/
│   ├── churn_development.csv
│   └── churn_final_test.csv
├── notebooks/
│   └── Week9_Model_Selection_Lab.ipynb
└── reports/
    ├── development_confusion_matrix.png
    ├── experiment_log.csv
    ├── final_metrics.json
    ├── frozen_selection.json
    ├── permutation_importance.csv
    ├── permutation_importance.png
    └── subgroup_metrics.csv
```

Use Python 3.10–3.13 if available. From the repository root, create and activate a virtual environment, install dependencies, and run the notebook top to bottom:

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
jupyter notebook notebooks/Week9_Model_Selection_Lab.ipynb
```

The notebook has been executed end-to-end with visible results. `reports/final_metrics.json` is an intentional guard: if it already exists, the final cell displays saved results instead of evaluating the test data again. Do not delete it to obtain a new score unless the instructor explicitly directs a fresh, separately versioned final evaluation.
