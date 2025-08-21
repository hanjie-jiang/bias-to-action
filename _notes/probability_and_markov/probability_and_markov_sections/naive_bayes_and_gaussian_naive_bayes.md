---
title: Naive Bayes & Gaussian Naive Bayes
---
**Resources**
- [StatQuest: Conditional Probability (YouTube)](https://www.youtube.com/watch?v=_IgyaD7vOOA)
- [StatQuest: Bayes' Rule](https://www.youtube.com/watch?v=9wCnvr7Xw4E)
- [3Blue1Brown: Bayes theorem, the geometry of changing beliefs](https://www.youtube.com/watch?v=HZGCoVF3YvM)

## Implementing Naive Bayes Classifier from Scratch in Python
We approach the implementation of the Naive Bayes Classifier by first calculating the prior probabilities of each class, and then the likelihood of each feature given a class.

```
import pandas as pd

def calculate_prior_probabilities(y):
    # Calculate prior probabilities for each class
    return y.value_counts(normalize=True)

def calculate_likelihoods(X, y, smoothing = False):
    likelihoods = {}
    for column in X.columns:
        likelihoods[column] = {}
        for class_ in y.unique():
            # Filter feature column data for each class
            class_data = X[y == class_][column]
            counts = class_data.value_counts()
            if not smoothing:
	            total_count = len(class_data)  # Total count of instances for current class
	            likelihoods[column][class_] = counts / total_count  # Direct likelihoods without smoothing
	        else:
		        total_count = len(class_data) + len(X[column].unique()) # total count with smoothing 
		        likelihoods[column][class_] = (counts + 1) / total_count # add-1 smoothing
    return likelihoods
```

Armed with these utility functions, we can implement the Naive Bayes Classifier function:

```
def naive_bayes_classifier(X_test, priors, likelihoods):
    predictions = []
    for _, data_point in X_test.iterrows():
        class_probabilities = {}
        for class_ in priors.index:
            class_probabilities[class_] = priors[class_]
            for feature in X_test.columns:
                # Use .get to safely retrieve probability and get a default of 1/total to handle unseen values
                feature_probs = likelihoods[feature][class_]
                class_probabilities[class_] *= feature_probs.get(data_point[feature], 1 / (len(feature_probs) + 1))

        # Predict class with maximum posterior probability
        predictions.append(max(class_probabilities, key=class_probabilities.get))

    return predictions
```