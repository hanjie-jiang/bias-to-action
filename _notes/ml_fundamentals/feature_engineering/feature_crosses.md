---
title: Feature Crosses & Dimensionality
---

# Feature Crosses & Dimensionality

## What are Feature Crosses?

Feature crosses combine single features together via dot-product or inner-product to help represent nonlinear relationships. This is particularly useful when individual features don't capture complex interactions in the data.

## High-Dimensional Feature Crosses

### The Problem

Using logistic regression as an example, when a dataset contains feature vector $X=(x_1, x_2, ..., x_k)$, the model would have:

$$Y = \text{sigmoid}(\sum_i \sum_j w_{ij} \langle x_i, x_j \rangle)$$

Where $w_{ij}$ is of dimension $n_{x_i} \cdot n_{x_j}$. When $n_{x_i} \times n_{x_j}$ is huge (especially in use cases like website customers and number of goods), this creates an extremely high-dimensional problem.

### The Solution: Dimensionality Reduction

**One way to get around this is to use a k-dimensional low-dimension vector (k << m, k << n).**

Now, $w_{ij} = x_i' \cdot x_j'$ and the number of parameters to tune becomes $m \times k + n \times k$.

This can also be viewed as **matrix factorization**, which has been widely used in recommendation systems.

### Matrix Factorization Example

```python
import numpy as np

# Original high-dimensional features
n_users = 1000
n_items = 5000
k = 50  # Low-dimensional representation

# Create low-dimensional embeddings
user_embeddings = np.random.randn(n_users, k)
item_embeddings = np.random.randn(n_items, k)

# Instead of n_users * n_items parameters,
# we now have n_users * k + n_items * k parameters
total_params = n_users * k + n_items * k
print(f"Parameters reduced from {n_users * n_items:,} to {total_params:,}")
```

## Feature Cross Selection

### The Challenge

In reality, we face a variety of high-dimensional features. A single feature cross of all different pairs would induce:
1. **Too many parameters**
2. **Overfitting issues**

### Effective Feature Combination Selection

We introduce feature cross selection based on decision tree models. Taking CTR (Click-Through Rate) prediction as an example:

**Input features:** age, gender, user type (free vs paid), searched item type (skincare vs foods)

**Decision tree approach:**
1. Make a decision tree from the original input and their labels
2. View the feature crosses from the tree
3. Extract meaningful feature combinations

**Example feature crosses from tree:**
1. age + gender
2. age + searched item type
3. paid user + search item type
4. paid user + age

### Gradient Boosting Decision Trees (GBDT)

**How to best construct the decision trees?**

One can use **Gradient Boosting Decision Trees (GBDT)**. The idea behind this is that before constructing a decision tree, we first calculate the error from the true value and iteratively construct the tree from the error.

```python
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd

# Example implementation
def extract_feature_crosses(X, y, n_estimators=100):
    """
    Extract feature crosses using GBDT
    """
    gbdt = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=3)
    gbdt.fit(X, y)
    
    # Extract feature importance
    feature_importance = gbdt.feature_importances_
    
    # Get feature crosses from tree structure
    feature_crosses = []
    for tree in gbdt.estimators_:
        # Extract decision paths and identify feature combinations
        # This is a simplified version - actual implementation would be more complex
        pass
    
    return feature_crosses
```

## Implementation Strategies

### 1. **Manual Feature Engineering**

```python
import pandas as pd

# Create feature crosses manually
def create_feature_crosses(df):
    # Age + Gender cross
    df['age_gender'] = df['age'].astype(str) + '_' + df['gender']
    
    # User type + Item type cross
    df['user_item_cross'] = df['user_type'] + '_' + df['item_type']
    
    return df
```

### 2. **Polynomial Features**

```python
from sklearn.preprocessing import PolynomialFeatures

# Create polynomial features (degree 2)
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)

print(f"Original features: {X.shape[1]}")
print(f"Polynomial features: {X_poly.shape[1]}")
```

### 3. **Factorization Machines**

```python
# Using a library like fastFM or similar
from fastFM import als

# Factorization Machine for feature interactions
fm = als.FMRegression(n_iter=1000, init_stdev=0.1, rank=8, l2_reg_w=0.1, l2_reg_V=0.1)
fm.fit(X_train, y_train)
```

## Best Practices

### **When to Use Feature Crosses:**

1. **Domain knowledge suggests interactions exist**
2. **Linear models need to capture nonlinear relationships**
3. **High-cardinality categorical features**
4. **Recommendation systems and collaborative filtering**

### **Implementation Guidelines:**

1. **Start with domain knowledge** - Don't cross everything
2. **Use tree-based methods** to identify important interactions
3. **Monitor for overfitting** - Cross-validation is crucial
4. **Consider computational cost** - High-dimensional crosses are expensive
5. **Use regularization** - L1/L2 regularization helps with sparse crosses

### **Common Pitfalls:**

- **Curse of dimensionality** - Too many crosses lead to sparse data
- **Overfitting** - Complex crosses without enough data
- **Computational expense** - High-dimensional crosses are slow
- **Loss of interpretability** - Complex crosses are hard to explain

## Related Topics

- **[Data Types & Normalization](data_types_and_normalization.md)** - Preparing features for crosses
- **[Categorical Encoding](categorical_encoding.md)** - Encoding categorical features for crosses
- **[Model Evaluation](../model_evaluation/evaluation_methods.md)** - Evaluating models with feature crosses
- **[Regularization](../regularization/l1_l2_regularization.md)** - Regularizing high-dimensional feature crosses
