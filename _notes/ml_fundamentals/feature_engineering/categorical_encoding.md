---
title: Categorical Feature Encoding
---

# Categorical Feature Encoding

## Understanding Categorical Features

Categorical features include data like male/female, blood type (A,B,AB,O), and other variables that can only select values from a finite set of choices. Categorical features are originally input as strings.

**Important Note:** While decision trees and some other models can directly take in strings, for logistic regression or SVM models, categorical features need to be translated to numerical form to work properly.

## Encoding Methods

### 1. **Ordinal Encoding**

**Use Case:** Treats data that has ordinal sequence (e.g., high > middle > low)

**Method:** Assigns numerical IDs that retain the high-to-low relationship

**Example:**
- High → 3
- Middle → 2  
- Low → 1

**Characteristics:**
- Preserves ordinal relationships
- Simple and interpretable
- Assumes meaningful order exists

```python
from sklearn.preprocessing import OrdinalEncoder

# Example data
categories = \[\[['high'], ['low'], ['middle'], ['high'], ['low']\]

# Create encoder
encoder = OrdinalEncoder(categories=\[\[['low', 'middle', 'high']\]\])

# Fit and transform
encoded = encoder.fit_transform(categories)
print(encoded)  # \[\[[2], [0], [1], [2], [0]\]
```

### 2. **One-Hot Encoding**

**Use Case:** Treats features that do not have ordinal relationships (e.g., blood type)

**Method:** Creates binary vectors for each category

**Example for Blood Type:**
- Type A → [1, 0, 0, 0]
- Type B → [0, 1, 0, 0]
- Type AB → [0, 0, 1, 0]
- Type O → [0, 0, 0, 1]

**Characteristics:**
- No ordinal relationship assumed
- Creates sparse vectors
- Increases dimensionality significantly

**Challenges:**
1. **High-dimensional features** can be difficult in:
   - K-nearest neighbors: Distance between high-dimensional vectors is hard to measure
   - Logistic regression: Parameters increase with higher dimensions, causing overfitting
   - Clustering: Only some dimensions may be helpful

2. **Sparse vectors** for saving space

```python
import pandas as pd

# Example data
data = pd.DataFrame({'blood_type': ['A', 'B', 'AB', 'O', 'A']})

# One-hot encoding
one_hot = pd.get_dummies(data, columns=['blood_type'])
print(one_hot)
```

### 3. **Binary Encoding**

**Use Case:** Alternative to one-hot encoding for space efficiency

**Method:** Uses binary representation to do a hash mapping on the original category ID

**Characteristics:**
- Saves space compared to one-hot encoding
- Usually fewer dimensions
- Maintains some category information

```python
import category_encoders as ce

# Example data
data = pd.DataFrame({'category': ['A', 'B', 'C', 'D', 'A']})

# Binary encoding
encoder = ce.BinaryEncoder(cols=['category'])
binary_encoded = encoder.fit_transform(data)
print(binary_encoded)
```

## Advanced Encoding Techniques

### **Target Encoding (Mean Encoding)**

**Method:** Replaces categories with the mean of the target variable for that category

**Advantages:**
- Captures relationship with target
- Reduces dimensionality
- Handles high-cardinality features

**Disadvantages:**
- Risk of overfitting
- Requires careful cross-validation

```python
from category_encoders import TargetEncoder

# Example with target variable
X = pd.DataFrame({'category': ['A', 'B', 'A', 'C', 'B']})
y = pd.Series([1, 0, 1, 0, 1])

# Target encoding
encoder = TargetEncoder(cols=['category'])
encoded = encoder.fit_transform(X, y)
print(encoded)
```

### **Hash Encoding**

**Method:** Uses hash functions to map categories to a fixed number of features

**Advantages:**
- Handles high-cardinality features
- Fixed output dimensionality
- Memory efficient

**Disadvantages:**
- Potential hash collisions
- Less interpretable

## Best Practices

### **When to Use Each Method:**

1. **Ordinal Encoding:**
   - Clear ordinal relationship exists
   - Categories have meaningful order
   - Tree-based models

2. **One-Hot Encoding:**
   - No ordinal relationship
   - Small number of categories (< 10)
   - Linear models

3. **Binary Encoding:**
   - Medium number of categories (10-100)
   - Memory constraints
   - Want to reduce dimensionality

4. **Target Encoding:**
   - High-cardinality features
   - Clear relationship with target
   - Proper cross-validation setup

### **Implementation Guidelines:**

1. **Handle missing values** before encoding
2. **Fit encoders on training data only**
3. **Apply same encoding to test data**
4. **Consider feature interactions** after encoding
5. **Monitor for overfitting** with high-cardinality features

### **Common Pitfalls:**

- **Data leakage:** Fitting encoders on test data
- **Overfitting:** Using target encoding without proper validation
- **Dimensionality explosion:** One-hot encoding high-cardinality features
- **Losing information:** Using ordinal encoding for non-ordinal data

## Related Topics

- **[Data Types & Normalization](data_types_and_normalization.md)** - Understanding different data types
- **[Feature Crosses](feature_crosses.md)** - Combining encoded features
- **[Model Evaluation](../model_evaluation/evaluation_methods.md)** - How encoding affects model performance
