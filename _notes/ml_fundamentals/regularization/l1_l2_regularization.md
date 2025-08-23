---
title: L1/L2 Regularization
---

# L2 / L1 Regularization

## Setup

Model (no intercept for simplicity):

$$\hat y_i = w\,x_i$$

**Data loss** (sum of squared errors):

$$\sum_i (y_i - w x_i)^2$$

**L2-regularized loss** (ridge):

$$\underbrace{\sum_i (y_i - w x_i)^2}_{\text{fit the data}} \;+\; \underbrace{\lambda\, w^2}_{\text{penalize big weights}}$$

- $\lambda>0$ controls the strength of the penalty (larger $\lambda$ stronger shrinkage).
- In practice, we usually **don't penalize the bias/intercept**.

## How L2 Penalizes the Parameter

Take derivative w.r.t. $w$ and set to 0:

$$\frac{\partial}{\partial w}\Big[\sum_i (y_i - w x_i)^2 + \lambda w^2\Big] = -2\sum_i x_i(y_i - w x_i) + 2\lambda w = 0$$

Rearrange:

$$w\big(\sum_i x_i^2 + \lambda\big) = \sum_i x_i y_i \quad\Rightarrow\quad \boxed{\,w_{\text{ridge}} = \dfrac{\sum_i x_i y_i}{\sum_i x_i^2 + \lambda}\,}$$

Compare to **unregularized** OLS:

$$w_{\text{OLS}} = \dfrac{\sum_i x_i y_i}{\sum_i x_i^2}$$

L2 adds $\lambda$ to the denominator and **shrinks $w$ toward 0**.

## Why L2 Decrease Variance and Increase Bias?

L2 regularization constrains how large the parameters can get. Constraining parameters makes the fitted function smoother/less wiggly, so predictions don't swing wildly when the training sample changes—this cuts variance. The tradeoff is that the constrained model can't perfectly adapt to the true signal, so estimates are pulled toward zero (or toward simpler shapes), which introduces bias.

## Tiny Numeric Example

Data: $x=[0,1,2,3]$, $y=[0,1,2,60]$ (last point is an outlier)
- $\sum x_i^2 = 14, \sum x_i y_i = 185$

Weights:
- **OLS (no L2):** $185/14 \approx 13.214$
- **L2, $\lambda=10$:** $185/(14+10) = 185/24 \approx 7.708185$
- **L2, $\lambda=100$:** $185/(14+100) = 185/114 \approx 1.623$

As $\lambda$ grows, $w$ is **pulled toward 0**, limiting the impact of the outlier.

## Gradient-Descent View (Weight Decay)

With learning rate $\eta$:

$$w_{\text{new}} = w_{\text{old}} - \eta\Big(\underbrace{-2\sum_i x_i(y_i - w_{\text{old}} x_i)}_{\text{data gradient}} \;+\; \underbrace{2\lambda w_{\text{old}}}_{\text{L2 shrink}}\Big)$$

The $+2\lambda w$ term is the **shrinkage** that steadily decays weights.

## Multi-Feature Form (for reference)

For features $X\in \mathbb{R}^{n\times d}$, target $\mathbf{y}$:

$$\mathbf{w}_{\text{ridge}} = (X^\top X + \lambda I)^{-1} X^\top \mathbf{y}$$

## Copy-Paste Python

```python
import numpy as np

x = np.array([0,1,2,3], dtype=float)
y = np.array([0,1,2,60], dtype=float)

Sxx = np.sum(x**2)
Sxy = np.sum(x*y)

def ridge_weight(lmbda):
    return Sxy / (Sxx + lmbda)

print("w_OLS        =", Sxy / Sxx)
for lmbda in [10, 100]:
    print(f"w_ridge", ridge_weight(lmbda))
```

**Notes**
- Standardize features before using L2/L1 (esp. linear/logistic).
- Tune $\lambda$ via cross-validation.
- Do **not** penalize the bias term.

## Related Topics

- **[Overfitting & Underfitting](overfitting_underfitting.md)** - Why regularization helps
- **[Early Stopping](early_stopping.md)** - Alternative regularization technique
- **[Model Evaluation](../model_evaluation/hyperparameter_tuning.md)** - Tuning regularization parameters
