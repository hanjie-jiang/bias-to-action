---
title: Data Types & Normalization
---

# Data Types & Normalization

## Data Types in Machine Learning

### Structured vs Unstructured Data

Machine learning deals with two main types of data:

#### **Structured / Tabular Data**
- Can be viewed as a data table from a relational database
- Every column has a clear definition
- Includes **numerical** and **categorical** data types
- Examples: CSV files, database tables, spreadsheets

#### **Unstructured Data**
- Includes **text, image, audio, video data**
- Information cannot be easily represented as numerical values
- No clear categorical definition
- Size of data is not identical
- Examples: documents, images, audio recordings

## Feature Normalization

### Why Normalize Numerical Features?

In order to eliminate the magnitude impact between features, we should always normalize the features that we use. This means uniformly normalizing all features to a similar range, which **helps compare between different metrics**.

### Types of Normalization

#### **1. Min-Max Scaling**
Linearly changes the original data so that it can be projected to [0, 1] range. This is an equal ratio transformation of the original data:

$$X_{\text{norm}} = \frac{X-X_{\text{min}}}{X_{\text{max}-X_{\text{min}}}}$$

**Characteristics:**
- Scales data to a fixed range [0, 1]
- Preserves zero entries in sparse data
- Sensitive to outliers

#### **2. Z-Score Normalization (Standardization)**
Projects the original data to a mean of 0 and variance = 1 distribution. If the original feature has mean $\mu$ and variance $\sigma$, the normalization equation is:

$$Z = \frac{x-\mu}{\sigma}$$

**Characteristics:**
- Centers data around mean = 0
- Scales to standard deviation = 1
- Less sensitive to outliers than min-max scaling
- Assumes data follows normal distribution

### When to Use Normalization

#### **SGD-Based Models (Require Normalization)**
- Linear regression
- Logistic regression
- Support vector machines
- Neural networks

**Example:** When two numerical features, $x_1$ of range [0,10] and $x_2$ of range [0,3], are not normalized, the ![Screenshot](<../resources/Screenshot 2025-08-05 at 8.22.11 PM.png>) gradient descent would not be as efficient as when normalization is applied.

#### **Tree-Based Models (Don't Require Normalization)**
- Decision trees
- Random forests
- Gradient boosting

**Reason:** Tree models split based on data and information gain ratio, which is not impacted by whether features have been normalized.

### Implementation Example

```python
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Sample data
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Min-Max Scaling
minmax_scaler = MinMaxScaler()
minmax_scaled = minmax_scaler.fit_transform(data)

# Z-Score Normalization
standard_scaler = StandardScaler()
standard_scaled = standard_scaler.fit_transform(data)

print("Original data:")
print(data)
print("\nMin-Max scaled:")
print(minmax_scaled)
print("\nZ-Score normalized:")
print(standard_scaled)
```

### Best Practices

1. **Fit scalers on training data only** - Never fit on test data to avoid data leakage
2. **Apply same scaling to test data** - Use the fitted scaler to transform test data
3. **Choose based on your model** - Use standardization for linear models, min-max for neural networks
4. **Handle outliers** - Consider robust scaling methods if outliers are present
5. **Preserve interpretability** - Keep track of scaling parameters for inverse transformation

### Related Topics

- **[Categorical Encoding](categorical_encoding.md)** - How to handle categorical features
- **[Feature Crosses](feature_crosses.md)** - Combining features for better performance
- **[Model Evaluation](../model_evaluation/evaluation_methods.md)** - How normalization affects model performance
