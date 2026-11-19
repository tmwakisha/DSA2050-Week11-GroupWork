# DSA2050A - Week 11 Lab

**Submission:** Individual work

| Member name | Student ID | Contribution |
|---|---:|---|
| Tamara Mwakisha | 660146 | Data inspection, leakage prevention, modelling, evaluation, error and subgroup analysis, permutation importance, model rationale, final testing, and documentation. |

## Model Selection & Defensible Evaluation

A reproducible customer-churn modelling practical using a synthetic teaching dataset. The work compares models fairly, examines errors and limitations, freezes a selection, and evaluates it once on a held-out final test set.

## Goal

The goal is to predict customer cancellation (`churn=1`; stayed is `churn=0`) using information that could be available before cancellation. The task is not simply to maximize a score: it is to select and defend a suitable model using fair validation, error analysis, subgroup results, and practical considerations.

The synthetic development dataset contains **2,720 customer records**, including **910 churners (33.46%)** and **1,810 retained customers (66.54%)**. Missing values occur in monthly charges (156), support tickets (120), and region (93). The development data contain no exact duplicate rows. The records are for teaching and do not represent real telecom customers.

## Features and leakage exclusions

Eight candidate prediction features are used: `tenure_months`, `monthly_charges_kes`, `support_tickets_90d`, `service_calls_90d`, `plan_type`, `region`, `autopay`, and `customer_segment`. Their availability at an actual operational prediction time would need to be verified.

- **`customer_id` — excluded:** it is an identifier, not a defensible generalizable predictor, and may encourage memorization.
- **`cancellation_confirmed` — excluded:** it is a post-outcome leakage field. In this dataset it maps `Active` to `churn=0` and `Cancelled` to `churn=1`; using it would reveal the outcome.
- **`churn` — excluded from predictors:** it is the target being predicted.

Numeric features use median imputation and standard scaling. Categorical features use most-frequent imputation and one-hot encoding. All transformations are within the model pipeline, so they are refitted on each training fold and do not learn from that fold’s validation data.

## Cross-validation comparison

All candidates used identical five-fold stratified cross-validation (`n_splits=5`, `shuffle=True`, `random_state=42`) and F1 as the primary selection metric. The prescribed grids were logistic regression (`C = 0.1, 1, 10`) and decision tree (`max_depth = 3, 5, 8`). The dummy majority-class baseline had **0.000 churn F1 in every fold**: predicting everyone as retained might yield about 66.5% accuracy, but detects no churners.

The table reports the best mean-F1 setting within each model family. The full log, including all six settings and fold scores, is [`reports/experiment_log.csv`](reports/experiment_log.csv).

| Model | Best setting | Mean F1 | F1 SD | Fold F1 scores (1–5) | Mean precision | Mean recall |
|---|---:|---:|---:|---|---:|---:|
| Logistic regression | `C=1` | 0.4701 | 0.0527 | 0.3915, 0.5342, 0.4718, 0.4345, 0.5185 | 0.6243 | 0.3780 |
| Decision tree | `max_depth=3` | 0.3985 | 0.0487 | 0.3920, 0.4571, 0.3158, 0.4367, 0.3906 | 0.5656 | 0.3176 |

## Selected model and rationale

**Selected:** logistic regression with `C=1`.

Its mean F1 exceeds the best decision tree by **0.0716**, and its mean recall is higher (0.3780 vs. 0.3176). Its fold SD is slightly higher (0.0527 vs. 0.0487), so it is not the more stable candidate by this estimate; however, the mean-F1 advantage is substantial under the prespecified comparison. Logistic regression at `C=10` produced the same displayed scores as `C=1`, so the tested results do not demonstrate a benefit from the less-regularized setting. This is the strongest tested candidate, not a deployment-ready model.

The choice and rationale were saved in [`reports/frozen_selection.json`](reports/frozen_selection.json) **before** the final-test evaluation.

## Errors

Out-of-fold development predictions give churn recall of **0.378**. There are **566 false negatives** (actual churners predicted to stay) and **205 false positives** (retained customers flagged as churn risks). False negatives can mean missed opportunities for retention action; false positives can use retention resources unnecessarily. The confusion matrix is [`reports/development_confusion_matrix.png`](reports/development_confusion_matrix.png).

![Out-of-fold development confusion matrix](reports/development_confusion_matrix.png)

## Subgroup findings

| Customer segment | Customers | Actual churners | Recall | Precision | False negatives |
|---|---:|---:|---:|---:|---:|
| Established | 2,268 | 649 | 0.2157 | 0.5932 | 509 |
| New | 452 | 261 | 0.7816 | 0.6518 | 57 |

Established-customer recall is markedly lower than New-customer recall in the out-of-fold development predictions. Treat this as a concerning result in this synthetic dataset and analysis, not a guaranteed real-population difference. The metrics are available in [`reports/subgroup_metrics.csv`](reports/subgroup_metrics.csv).

## Permutation importance

On a held-out 25% slice of development data, importance was measured as mean F1 reduction after shuffling a feature (five repeats).

| Feature | Mean F1 reduction | SD |
|---|---:|---:|
| `tenure_months` | 0.0787 | 0.0237 |
| `support_tickets_90d` | 0.0350 | 0.0157 |

`tenure_months` is the feature with the highest measured predictive reliance, followed by `support_tickets_90d`. Permutation importance is **not evidence of causation**. Correlated features may share or mask importance, and estimates can vary with the validation sample and fitted model. See [`reports/permutation_importance.csv`](reports/permutation_importance.csv) and [`reports/permutation_importance.png`](reports/permutation_importance.png).

![Permutation importance plot](reports/permutation_importance.png)

## Final metrics

The final-test file remained untouched during development analysis and model selection. After the model and rationale were frozen, the selected pipeline was refit on all development data and evaluated **once** on 680 final-test records.

| Metric | Final-test result |
|---|---:|
| Accuracy | 0.7000 |
| F1 | 0.4486 |
| Precision | 0.5845 |
| Recall | 0.3640 |

Confusion matrix (rows = actual, columns = predicted; class order = stayed, churn):

|  | Predicted stayed | Predicted churn |
|---|---:|---:|
| **Actual stayed** | 393 | 59 |
| **Actual churn** | 145 | 83 |

Final-test F1 is 0.0215 below the development CV mean of 0.4701, broadly consistent with the CV estimate. Recall remains low: **145 of 228 actual churners were missed** on the final test. The result was not used to retune or select another model. Machine-readable metrics are in [`reports/final_metrics.json`](reports/final_metrics.json).

## Limitations

1. **Synthetic data:** the records may not represent real Kenyan customer behaviour, data collection, missingness, or churn prevalence.
2. **Prediction context and costs are unspecified:** the exercise does not establish when each feature is available, the business costs of false positives and false negatives, or how retention interventions would work. The weak recall and Established/New performance difference also warrant investigation.
3. **Limited model search and validation context:** only two model families and a small prescribed grid were tested. Cross-validation and the final score do not guarantee performance under temporal or population changes.

Before real deployment, validate on representative, time-separated data; verify feature availability and prevent leakage; assess subgroup performance and calibration; and define error and intervention costs before selecting an operating threshold using development data only. This exercise is not evidence that the model is ready for use on real customers.

## Reflection

1. **Which model was selected and why was it defensible?** Logistic regression with `C=1` was selected. Under the prespecified, identical five-fold stratified cross-validation, it achieved mean F1 **0.4701** with fold standard deviation **0.0527**, compared with mean F1 0.3985 (SD 0.0487) for the best decision tree. The 0.0716 mean-F1 advantage and higher mean recall (0.3780 versus 0.3176) support the choice. Logistic regression's fold variation was slightly higher, which is acknowledged; its comparatively simple, regularized form and stronger average performance make it defensible among the tested candidates.
2. **What did false negatives mean for this provider? Which subgroup struggled most?** A false negative is a customer who actually churned but was predicted to stay. The provider could miss a chance to offer that customer a retention intervention. Development out-of-fold predictions contained **566 false negatives**. Established customers struggled most: recall was **0.2157**, with 509 of 649 actual churners missed, compared with 0.7816 recall and 57 of 261 missed for New customers.
3. **Which feature was highest in permutation importance, and why does that not imply causation?** `tenure_months` ranked highest, with a mean F1 reduction of **0.0787** when shuffled. This indicates that this fitted model relied on tenure for predictions on the held-out development slice; it does not show that tenure causes churn. Correlated predictors may share or mask importance, and the result is specific to the model and validation sample.
4. **Compare final-test F1 with mean CV F1. Did your model generalize reasonably?** Final-test F1 was **0.4486**, compared with mean development CV F1 **0.4701**, a decrease of **0.0215**. This is a modest difference and suggests broadly reasonable generalization to this synthetic test set. However, final recall was only 0.3640 and the model missed 145 of 228 actual churners, so performance is not sufficient evidence of deployment readiness.
5. **State two limitations of this synthetic dataset and one step you would take before deploying a real churn model.** First, synthetic records may not represent real Kenyan customer behaviour, data collection, missingness, or churn prevalence. Second, the exercise does not establish real prediction-time feature availability or the business costs and outcomes of retention interventions. Before deployment, I would validate the frozen modelling approach on representative, time-separated real customer data, while verifying feature timing and measuring subgroup performance and intervention costs.

## Reproduce

Use Python 3.10–3.13 if available. From the repository root:

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
jupyter notebook notebooks/Week9_Model_Selection_Lab.ipynb
```

The notebook has been run end to end. If `reports/final_metrics.json` already exists, the final evaluation cell displays the saved results instead of evaluating the test data again. Preserve this guard; do not delete that file to obtain a new score unless instructed to conduct a separately versioned evaluation.
