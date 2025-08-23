---
title: Model Evaluation Methods
---

# Model Evaluation Methods

## Overview

In ML algorithm design, we usually split the samples into training and test data set, where the training set is used to training the model and the test set is used to evaluate the model. In sample split and model evaluation process, we could use different sampling or evaluation methods.

## Main Evaluation Methods

### Holdout Evaluation

Holdout evaluation is the easiest way as it randomly split the original sample set into training and evaluation. For example, for a clickthrough rate prediction algorithm, we split the samples into 70 - 30%. We use the 70% data for model training and the 30% for evaluation, including ROC curve, accuracy calculation and recall rate metric evaluation.

**Significant downside:** The calculated final evaluation metric is highly correlated with the original data split. In order to eliminate this randomness, researchers started to use the "cross validation" idea.

### Cross-Validation

k-fold cross validation would always split the data set into k different sets that are of same counts. The method goes through all the k sample sets and always use the current subset as the evaluation set whereas the other ones are training set. Usually we use k = 10.

### Bootstrap

- Make a **fake test set** by randomly picking the same number of rows from your real test set **with replacement** (so rows can repeat and some are left out).
  - Suppose the test set has **n rows**.
  - Pick **n indices at random WITH replacement** from `0..n-1`. (Duplicates allowed; some rows won't be picked.)
  - Those picked rows form one **fake test set**.
- On that fake set, compute your metric (accuracy, F1, AUC, RMSE whatever you care about).
- Repeat steps 1-2 a lot (like **1,000 times**).
- Now you have 1,000 metric values.
  - The **average** is your central estimate.
  - The **middle 95% range** (ignore the lowest 2.5% and highest 2.5%) is your **95% confidence interval**.

As $n$ gets large, about **36.8%** of items are not in the set (never selected) and **63.2%** appear at least once. This is the source of the bootstrap terminology.

## Related Topics

- **[Metrics & Validation](metrics_and_validation.md)** - Understanding evaluation metrics
- **[Hyperparameter Tuning](hyperparameter_tuning.md)** - Using evaluation methods for tuning
- **[Feature Engineering](../feature_engineering/data_types_and_normalization.md)** - How evaluation affects feature selection
