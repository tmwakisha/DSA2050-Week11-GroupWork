# DSA2050A - Week 11 Lab

**Submission:** Individual work

| Member name | Student ID | Contribution |
|---|---:|---|
| Tamara Mwakisha | 660146 | Data inspection, leakage prevention, modelling, evaluation, error and subgroup analysis, permutation importance, model selection, final testing, and documentation. |

## Model Selection & Defensible Evaluation

This project explores how machine-learning models can help identify customers who may cancel their internet service. Rather than focusing only on the highest score, the analysis compares candidates fairly, examines their errors, and explains the final choice before evaluating it on held-out test data.

## Goal

The goal is to predict whether a customer cancels (`churn=1`) or stays (`churn=0`) from information that could be available before cancellation. Each row represents one customer.

The synthetic development dataset contains **2,720 customer records**: **910 churners (33.46%)** and **1,810 customers who stayed (66.54%)**. Values are missing for monthly charges (156), support tickets (120), and region (93). There are no exact duplicate rows. These records were created for teaching and do not represent actual customer behaviour.

## Features and leakage exclusions

The model uses eight candidate features: `tenure_months`, `monthly_charges_kes`, `support_tickets_90d`, `service_calls_90d`, `plan_type`, `region`, `autopay`, and `customer_segment`. In a real prediction setting, the availability of each feature at the prediction time would need to be confirmed.

- **`customer_id` is excluded** because it identifies a record rather than describing customer behaviour. Using it could encourage memorization instead of learning patterns that generalize.
- **`cancellation_confirmed` is excluded** because it is only known after the outcome. In this dataset, `Active` always corresponds to `churn=0` and `Cancelled` to `churn=1`; including it would give the model the answer.
- **`churn` is the target**, so it is not used as an input feature.

Numeric features are median-imputed and scaled. Categorical features are imputed with the most frequent value and one-hot encoded. These steps are inside each model pipeline, so they are fitted on each training fold without using that fold's validation data.

## Cross-validation comparison

All candidates were evaluated on the same five stratified folds (`n_splits=5`, `shuffle=True`, `random_state=42`), with F1 as the primary selection metric. The search tested logistic regression with `C = 0.1, 1, 10` and decision trees with `max_depth = 3, 5, 8`.

The dummy baseline predicts that every customer stays. Its churn F1 was **0.000 in all five folds**. Because about 66.5% of customers stayed, that prediction rule could still appear reasonably accurate while failing to identify any churners. The results below show the best setting in each model family.

| Model | Best setting | Mean F1 | F1 SD | Fold F1 scores (1–5) | Mean precision | Mean recall |
|---|---:|---:|---:|---|---:|---:|
| Logistic regression | `C=1` | 0.4701 | 0.0527 | 0.3915, 0.5342, 0.4718, 0.4345, 0.5185 | 0.6243 | 0.3780 |
| Decision tree | `max_depth=3` | 0.3985 | 0.0487 | 0.3920, 0.4571, 0.3158, 0.4367, 0.3906 | 0.5656 | 0.3176 |

## Selected model and rationale

**Selected model: logistic regression with `C=1`.** Its mean F1 of 0.4701 is 0.0716 higher than the best decision tree's 0.3985. It also has higher mean recall (0.3780 versus 0.3176). The logistic model's fold standard deviation is slightly higher (0.0527 versus 0.0487), so it is not the more consistent candidate by that measure. However, the mean-F1 improvement is substantial under the same validation folds, making it the strongest of the tested options.

The `C=10` logistic-regression setting produced the same reported scores as `C=1`, so the results do not show a clear advantage to using the less-regularized setting. This selection is limited to the small grid in the lab and does not establish that the model is ready for real-world use. The model and rationale were recorded before the final-test evaluation.

## Errors

The development out-of-fold predictions include **566 false negatives** and **205 false positives**. A false negative is a customer who churned but was predicted to stay, which could mean a missed opportunity to offer a retention option. A false positive is a customer flagged as a churn risk who actually stayed; responding to too many such flags could use time and resources unnecessarily.

## Subgroup findings

| Customer segment | Customers | Actual churners | Recall | Precision | False negatives |
|---|---:|---:|---:|---:|---:|
| Established | 2,268 | 649 | 0.2157 | 0.5932 | 509 |
| New | 452 | 261 | 0.7816 | 0.6518 | 57 |

Recall is much lower for Established customers than for New customers in the development out-of-fold results. The gap deserves attention, but because the data are synthetic it should not be treated as proof that the same difference exists among real customers.

## Permutation importance

Permutation importance was measured on a held-out 25% slice of development data as the mean F1 decrease after shuffling a feature, averaged over five repeats.

| Feature | Mean F1 reduction | SD |
|---|---:|---:|
| `tenure_months` | 0.0787 | 0.0237 |
| `support_tickets_90d` | 0.0350 | 0.0157 |

`tenure_months` had the highest measured importance, followed by `support_tickets_90d`. This describes how much the fitted model relied on each feature for predictions in this validation slice; it does not show that either feature causes churn. Correlated features can share or mask importance, and results can vary across samples and fitted models.

## Final metrics

After the model choice was frozen, logistic regression (`C=1`) was refit on all development records and evaluated once on the **680-record final test set**.

| Final-test measure | Score |
|---|---:|
| Accuracy | 0.7000 |
| F1 | 0.4486 |
| Precision | 0.5845 |
| Recall | 0.3640 |

The final F1 score is **0.0215 lower** than the development cross-validation mean of 0.4701. Recall of 0.3640 means the model identified about 36 out of every 100 customers who churned; **145 of the 228 actual churners were missed**. The final confusion matrix is shown in the completed notebook. These test results were not used to change the selected model or its settings.

## Limitations

1. **Synthetic data may not reflect real customers.** The records may differ from actual customer behaviour, how information is collected, missing-data patterns, or real churn rates.
2. **The modelling setup leaves important real-world questions unanswered.** The exercise does not confirm when each feature would be available, how costly false positives and false negatives would be, or how effective retention actions might be. The comparison also covers only two model families and a small parameter grid, so performance may change across populations or over time. Before real use, the model would need validation on representative, time-separated customer data, checks of feature timing and subgroup performance, and an assessment of intervention costs.

## Reflection

1. **Which model was selected and why was it defensible? Include mean F1 and fold standard deviation.**

   Logistic regression with `C=1` was selected. It achieved mean F1 **0.4701** and fold standard deviation **0.0527**, compared with 0.3985 (SD 0.0487) for the best decision tree. Its mean F1 and recall were higher, and the gain in average F1 was substantial. Although the fold variation was slightly higher, the results support logistic regression as the stronger of the tested candidates.

2. **What did false negatives mean for this provider? Which subgroup struggled most?**

   False negatives are customers who actually churned but were predicted to stay. The provider could therefore miss the chance to offer them a retention option. There were **566 false negatives** in the development out-of-fold results. Established customers struggled most: recall was **0.2157**, with 509 of 649 actual churners missed.

3. **Which feature was highest in permutation importance, and why does that not imply causation?**

   `tenure_months` ranked highest, with a mean F1 decrease of **0.0787** when shuffled. This indicates predictive reliance by this fitted model on the held-out development slice; it does not show that tenure causes churn. Correlated features may share or mask importance, and the estimate may differ with another sample or model.

4. **Compare final-test F1 with mean CV F1. Did your model generalize reasonably?**

   Final-test F1 was **0.4486**, compared with mean CV F1 of **0.4701**, a decrease of 0.0215. That modest gap suggests broadly reasonable generalization to this synthetic test set. However, final recall was 0.3640, with 145 churners missed, so the result alone is not enough to support real-world use.

5. **State two limitations of this synthetic dataset and one step you would take before deploying a real churn model.**

   First, synthetic records may not reflect real customer behaviour or how customer information is collected. Second, the assignment does not establish real-world feature availability or the costs and effectiveness of retention actions. Before deployment, the model should be validated on representative, time-separated real customer data, with feature timing and performance across customer groups checked.
