---
title: K-Nearest Neighbors
---

# K-Nearest Neighbors (k-NN) Algorithm

The kNN algorithm works on a basic principle: a data point is likely to be in the same category as the data points it is closest to. Note that choosing 'k' significantly impacts our model. A low 'k' might capture more noise in the data, whereas a high 'k' is computationally expensive.

## Euclidean Distance Calculation

In k-NN, classification is determined by weighing the distance between data points. Euclidean distance is a frequently used metric that calculates the shortest straight-line distance $\sqrt{(x_1-x_2)^2 + (y_1 - y_2)^2}$ between two data points $(x_1, y_1)$ and $(x_2, y_2)$ in a Euclidean space.

```python
import math

# The 'euclidean_distance' function computes the Euclidean distance between two points
def euclidean_distance(point1, point2):
    squares = [(p - q) ** 2 for p, q in zip(point1, point2)] # Calculate squared distance for each dimension
    return math.sqrt(sum(squares)) # Return the square root of the sum of squares

# Test it
point1 = (1, 2) # The coordinates of the first point
point2 = (4, 6) # The coordinates of the second point
print(euclidean_distance(point1, point2)) # 5.0
```

## Actual KNN Algorithm

```python
from collections import Counter
import numpy as np

def k_nearest_neighbors(data, query, k, distance_fn):
    neighbor_distances_and_indices = []
    # Compute distance from each training data point
    for idx, label in enumerate(data):
        distance = euclidean_distance(label[0], query)
        neighbor_distances_and_indices.append((distance, idx))
    # Sort array by distance
    sorted_neighbor_distances_and_indices = sorted(neighbor_distances_and_indices)
    # Select k closest data points
    k_nearest_distances_and_indices = sorted_neighbor_distances_and_indices[:k]
    # Obtain class labels for those k data points
    k_nearest_labels = [data[i][1] for distance, i in k_nearest_distances_and_indices]
    # Majority vote
    most_common = Counter(k_nearest_labels).most_common(1)
    return most_common[0][0] # Return the label of the class that receives the majority vote

def euclidean_distance(point1, point2):
    distance = sum((p - q) ** 2 for p, q in zip(point1, point2))
    return np.sqrt(distance)
    
def mannhattan_distance(point1, point2):
    return np.sum(np.abs(p - q) for p, q in zip(point1, point2))

data = [
    ((2, 3), 0),
    ((5, 4), 0),
    ((9, 6), 1),
    ((4, 7), 0),
    ((8, 1), 1),
    ((7, 2), 1)
]
query = (7,6)
k=2

class_label = k_nearest_neighbors(data, query, k, distance_fn)
print(class_label)
```

## Related Topics

- **[K-Means Clustering](k_means_clustering.md)** - Alternative unsupervised learning algorithm
- **[Model Evaluation](../model_evaluation/metrics_and_validation.md)** - Evaluating KNN performance
- **[Feature Engineering](../feature_engineering/data_types_and_normalization.md)** - How normalization affects KNN
