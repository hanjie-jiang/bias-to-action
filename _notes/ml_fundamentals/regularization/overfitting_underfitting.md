---
title: Overfitting & Underfitting
---

# Overfitting & Underfitting

This section tells how one could efficiently recognize overfit and underfit scenarios and do model improvements based on what has been identified.

## What is Overfit and What is Underfit?

- **Overfit** means that a model can be overfitting on its training data whereas on the test and new data sets, it's performing worse.
- **Underfit** means that the model is performing poorly on both training and test data sets.

## Ways to Mitigate Overfit and Underfit

### Avoid Overfitting

- **Data**: obtaining more data is one primitive way of solving overfit problem as more data can help the model to learn more efficient features to mitigate the impact from noise. Using rotation or expansion for image or GAN for getting more new training data.
- **Model**: one could use less complicated / complex model to avoid overfitting. For example, in NN one could reduce the number of layers or neurons in each layer; or in decision tree, one could reduce the depth of the tree or cut the tree.
- **Regularization**: one could use L2 regularization in model parameters to constraint the model.
- **Ensemble method**: ensemble method is to integrate multiple models together to avoid a single model overfitting issue, such as bagging methods.

### Avoid Underfitting

- **Add more features**: when there is not enough features or the features are not relevant with the sample labels, there would be a underfit. We could dig into contextual features / ID features / combination of features to obtain better results. In deep learning, factor decomposition / gradient-boosted decision tree / deep-crossing can all be used for get more features.
- **Increase the complexity of model**.
- **Decrease regularization parameters**.

## Related Topics

- **[L1/L2 Regularization](l1_l2_regularization.md)** - Mathematical foundations of regularization
- **[Early Stopping](early_stopping.md)** - Training control techniques
- **[Model Evaluation](../model_evaluation/evaluation_methods.md)** - How to detect overfitting/underfitting
