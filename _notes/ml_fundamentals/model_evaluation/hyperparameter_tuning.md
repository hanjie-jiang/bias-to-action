---
title: Hyperparameter Tuning
---

# Hyperparameter Tuning

For a lot of algorithm engineers, hyperparameter tuning can be really of headache, as there is no other way other than empirically tune the parameters to a reasonable range, while it is really important for the algorithm to be effective.

## Common Ways of Hyperparameter Tuning

### Grid Search

Exhaustive on a small, **low-dimensional** space. Deterministic but expensive; scales poorly. In reality, it tend to be used as a bigger search space and larger step size to find the possible range of optimal results, then to shrink the search space and find more accurate optimal solution.

### Random Search

Sample hyperparams at random (often **log-uniform** for learning rates). Much better than grid when only a few dims matter but cannot guarantee for a optimal solution.

### Bayesian Optimization

Model config -> score to pick promising next trials. Unlike random/grid search **do not learn** from past trials, BO **uses what you have learned so far** to place the next (expensive) trial where it is most likely to pay off.

## Related Topics

- **[Evaluation Methods](evaluation_methods.md)** - Using evaluation methods for tuning
- **[Metrics & Validation](metrics_and_validation.md)** - Using metrics to guide tuning
- **[Regularization](../regularization/l1_l2_regularization.md)** - Tuning regularization parameters
