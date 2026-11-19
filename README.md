# DSA2050A - Week 11 Lab

**Submission:** Individual work

| Member name | Student ID | Contribution |
|---|---:|---|
| Tamara Mwakisha | 660146 | Data inspection, leakage prevention, modelling, evaluation, error and subgroup analysis, permutation importance, model selection, final testing, and documentation. |

## Model Selection & Defensible Evaluation

This project looks at how machine-learning models can help identify customers who may cancel their internet service. The focus is not just on getting the highest score. I compare the models fairly, look at the mistakes they make, and explain why I chose one before evaluating it on the final test data.

## Goal

The goal is to predict whether a customer cancels (`churn=1`) or stays (`churn=0`) using information that could be available before cancellation. Each row represents one customer.

The synthetic development dataset has **2,720 customer records**: **910 churners (33.46%)** and **1,810 customers who stayed (66.54%)**. Some values are missing in monthly charges (156), support tickets (120), and region (93). There are no exact duplicate rows. These records were created for teaching and should not be treated as real customer data.

## Features and leakage exclusions

The model uses eight candidate features: `tenure_months`, `monthly_charges_kes`, `support_tickets_90d`, `service_calls_90d`, `plan_type`, `region`, `autopay`, and `customer_segment`. In a real setting, I would first confirm that each feature is available at the moment a prediction is made.

- **`customer_id` is left out** because it identifies a record rather than describing customer behaviour. Including it could let a model memorize IDs instead of learning patterns that generalize.
- **`cancellation_confirmed` is left out** because it is only known after the outcome. In this dataset, `Active` always matches `churn=0` and `Cancelled` always matches `churn=1`, so including it would give the model the answer.
- **`churn` is the target**, so it is not used as an input feature.

For numeric features, the pipeline fills missing values with the median and then scales the values. For categorical features, it fills missing values with the most common category and applies one-hot encoding. These steps are part of each model pipeline, so they are learned from the training portion of each fold—not from its validation portion.

## Cross-validation comparison

I used the same five stratified folds for every candidate (`n_splits=5`, `shuffle=True`, `random_state=42`) and used F1 as the main selection metric. The search tested logistic regression with `C = 0.1, 1, 10` and decision trees with `max_depth = 3, 5, 8`.

The dummy baseline predicts that everyone stays. Its F1 score for churn was **0.000 in all five folds**. It could still look reasonably accurate because about 66.5% of customers stayed, but it would fail to catch any churners. The table below shows the best setting from each model family; the full log includes all six settings and fold scores.

| Model | Best setting | Mean F1 | F1 SD | Fold F1 scores (1–5) | Mean precision | Mean recall |
|---|---:|---:|---:|---|---:|---:|
| Logistic regression | `C=1` | 0.4701 | 0.0527 | 0.3915, 0.5342, 0.4718, 0.4345, 0.5185 | 0.6243 | 0.3780 |
| Decision tree | `max_depth=3` | 0.3985 | 0.0487 | 0.3920, 0.4571, 0.3158, 0.4367, 0.3906 | 0.5656 | 0.3176 |

## Selected model and rationale

I selected **logistic regression with `C=1`**. Its mean F1 was 0.4701, which is 0.0716 higher than the best decision tree's 0.3985. It also had higher mean recall: 0.3780 compared with 0.3176. Logistic regression's fold-to-fold standard deviation was slightly higher (0.0527 compared with 0.0487), so it was not the more consistent model by that measure. Even so, the improvement in average F1 was large enough to support the choice among the tested options.

The `C=10` logistic-regression setting had the same reported scores as `C=1`, so the results do not show a clear reason to use the less-regularized setting. This is the strongest option from the small grid in this lab, not proof that the model is ready to use with real customers. I recorded the choice and rationale before evaluating the final test data.

## Errors

The out-of-fold development predictions found **566 false negatives** and **205 false positives**. A false negative is a customer who actually churned but was predicted to stay. For the provider, that could mean missing a chance to offer the customer a retention option. A false positive is a customer flagged as a churn risk who actually stayed; following up with too many of these customers could use time and resources unnecessarily.

## Subgroup findings

| Customer segment | Customers | Actual churners | Recall | Precision | False negatives |
|---|---:|---:|---:|---:|---:|
| Established | 2,268 | 649 | 0.2157 | 0.5932 | 509 |
| New | 452 | 261 | 0.7816 | 0.6518 | 57 |

The model's recall was much lower for Established customers than for New customers in the out-of-fold development predictions. This gap is worth investigating. Since the data are synthetic, it should not be taken as proof that the same difference would appear among real customers.

## Permutation importance

I measured permutation importance on a held-out 25% slice of the development data. The measure is the mean drop in F1 after shuffling one feature, averaged over five repeats.

| Feature | Mean F1 reduction | SD |
|---|---:|---:|
| `tenure_months` | 0.0787 | 0.0237 |
| `support_tickets_90d` | 0.0350 | 0.0157 |

`tenure_months` had the largest measured effect on the model's predictions, followed by `support_tickets_90d`. This does not mean either feature causes churn. It only shows how much this particular model's F1 changed when a feature was shuffled in this validation slice. If features are correlated, they can share or hide some of each other's importance, and the estimates may change with a different sample or fitted model.

## Final metrics

I kept the final-test data aside while inspecting the development data, comparing models, and making the selection. After recording the choice, I refit the selected pipeline on all development data and evaluated it once on the **680 final-test records**.

| Metric | Final-test result |
|---|---:|
| Accuracy | 0.7000 |
| F1 | 0.4486 |
| Precision | 0.5845 |
| Recall | 0.3640 |

The confusion matrix below uses rows for the actual class and columns for the predicted class.

|  | Predicted stayed | Predicted churn |
|---|---:|---:|
| **Actual stayed** | 393 | 59 |
| **Actual churn** | 145 | 83 |

Final-test F1 was **0.4486**, which is 0.0215 lower than the mean cross-validation F1 of 0.4701. That is a fairly small difference, so the result is broadly in line with the cross-validation estimate. Still, recall on the final test was only 0.3640, and the model missed 145 of the 228 customers who actually churned. I did not use the final-test score to change the model or its settings.

## Limitations

There are a few important limits to keep in mind:

1. The data are synthetic, so they may not reflect real customer behaviour, how information is collected, patterns of missing data, or the actual rate of churn.
2. The exercise does not confirm when each feature would be available in a real prediction workflow, or what false positives, false negatives, and retention actions would cost in practice.
3. The comparison only covers two model families and a small grid. Even a good cross-validation and final-test result cannot guarantee performance on a different population or at a later time.

Before using a churn model with real customers, I would test it on representative, time-separated data, confirm that all features are available before the prediction point, and check performance across customer groups. I would also work out the costs and likely outcomes of retention actions before deciding what prediction threshold to use. The results here are not enough to justify real-world deployment.

## Reflection

1. **Which model was selected and why was it defensible? Include mean F1 and fold standard deviation.**

   I selected logistic regression with `C=1`. It had a mean F1 of **0.4701** and a fold standard deviation of **0.0527**, compared with 0.3985 (SD 0.0487) for the best decision tree. Its average F1 and recall were higher, and the gain in mean F1 was substantial. The slightly higher fold variation is a trade-off, but the evidence supports logistic regression as the stronger of the tested options.

2. **What did false negatives mean for this provider? Which subgroup struggled most?**

   False negatives are customers who churned but were predicted to stay, so the provider could miss an opportunity to try to retain them. There were **566** such cases in the development out-of-fold results. Established customers had the lowest recall (**0.2157**); 509 of their 649 actual churners were missed.

3. **Which feature was highest in permutation importance, and why does that not imply causation?**

   `tenure_months` was highest, with a mean F1 drop of **0.0787** when shuffled. This shows that the fitted model relied on tenure when making predictions in the validation slice. It does not show that tenure causes churn. Other correlated features can share or mask importance, and the result may differ with another sample or model.

4. **Compare final-test F1 with mean CV F1. Did your model generalize reasonably?**

   Final-test F1 was **0.4486**, compared with mean CV F1 of **0.4701**—a difference of 0.0215. That is a fairly small drop, so generalization to this synthetic test set looks broadly reasonable. However, final-test recall was 0.3640, with 145 churners missed, so I would not consider the model ready for real use based on this result alone.

5. **State two limitations of this synthetic dataset and one step you would take before deploying a real churn model.**

   First, synthetic data may not reflect real customer behaviour or the way customer information is recorded. Second, the assignment does not establish the real-world timing of features or the cost and effectiveness of retention actions. Before deployment, I would validate the approach on representative, time-separated real customer data and check feature timing and performance across customer groups.
