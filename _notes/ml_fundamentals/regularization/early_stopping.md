---
title: Early Stopping
---

# Early Stopping

- Early stopping watches **validation** loss/metric and halts training when it stops improving, and is a **stopping rule** driven by the **validation metric's change**, not a pre-fixed iteration count
- It **reduces overfitting** (lower variance) by not letting the model memorize noise; acts like **implicit L2** regularization.

Train while checking performance on a validation set. Whenever the validation score improves, remember those weights. If it doesn't improve for a while (patience), stop and roll back to the best checkpoint. This caps model complexity at the point where it generalized best, preventing the later epochs from fitting noise.

## Related Topics

- **[Overfitting & Underfitting](overfitting_underfitting.md)** - Why early stopping helps
- **[L1/L2 Regularization](l1_l2_regularization.md)** - Alternative regularization techniques
- **[Model Evaluation](../model_evaluation/evaluation_methods.md)** - Using validation for early stopping
