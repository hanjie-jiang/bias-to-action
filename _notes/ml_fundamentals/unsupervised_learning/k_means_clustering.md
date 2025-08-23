---
title: K-Means Clustering
---

# K-means Clustering

Algorithms such as SVM, logistic regression, decision trees are more for the categorization, i.e. based on the known labelled samples, classifiers are training so that it could apply the same logic on unlabeled samples. Unlike the classification problems, clustering is directly categorize the samples without any previously known labelling.

Classification belongs to supervised learning whereas clustering is a type of unsupervised learning algorithm. K-means clustering, as one type of the most basic and fundamental clustering algorithm, has the main idea of iteratively finding the way of cutting the space into K clusters, so that the loss function is the lowest. The loss function can be defined as the sum of squared error distance of each sample from their clustered centers:

$$J(c,\mu) = \sum_{i=1}^M ||x_i - \mu_{c_i}||^2$$

where $x_i$ represents the samples, $c_i$ represents the cluster that $x_i$ belongs to, $\mu_{c_i}$ corresponds to the center of the cluster that $x_i$'s located in and $M$ is the total number of samples.

## K-means Clustering Algorithm in Steps

The goal of K-means clustering is to categorize the dataset of interest into K-clusters, and also provides the cluster center corresponding to each data points:

1. **Data engineering and cleaning**: normalization and outlier removal.
2. **Randomly pick K-cluster centers**, labelled as $\mu_1^{(0)}, \mu_2^{(0)}, ..., \mu_K^{(0)}$
3. **Define the loss function** to be $J(c,\mu) = \min_{\mu} \min_{c} \sum_{i=1}^M ||x_i - \mu_{c_i}||^2$
4. **Iterate through the process below by t times**, where t denotes the number of iterations:
   1. for every sample $x_i$, categorize it to the cluster that has shortest distance $$c_i^{(t)} \leftarrow {\arg\min}_k ||x_i - \mu_k^{(t)}||^2$$
   2. for every cluster k, recalculate the center: $$\mu_k^{(t+1)}\leftarrow {\arg\min}_\mu \sum_{i:c_i^{(t)}=k} ||x_i - \mu||^2$$

```python
# k-Means algorithm
def k_means(data, centers, k):
    while True:
        clusters = [[] for _ in range(k)] 

        # Assign data points to the closest center
        for point in data:
            distances = [distance(point, center) for center in centers]
            index = distances.index(min(distances)) 
            clusters[index].append(point)

        # Update centers to be the mean of points in a cluster
        new_centers = []
        for cluster in clusters:
            center = (sum([point[0] for point in cluster])/len(cluster), 
                      sum([point[1] for point in cluster])/len(cluster)) 
            new_centers.append(center)

        # Break loop if centers don't change significantly
        if max([distance(new, old) for new, old in zip(new_centers, centers)]) < 0.0001:
            break
        else:
            centers = new_centers
    return clusters, centers
```

## Related Topics

- **[K-Nearest Neighbors](k_nearest_neighbors.md)** - Alternative unsupervised learning algorithm
- **[Model Evaluation](../model_evaluation/metrics_and_validation.md)** - Evaluating clustering performance
- **[Feature Engineering](../feature_engineering/data_types_and_normalization.md)** - How normalization affects clustering
