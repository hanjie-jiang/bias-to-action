---
title: Logistic Regression
---

# Logistic Regression

Logistic regression is the most widely used and most fundamental model that one could use in the ML industry. One should always understand the deduction of logistic regression and application of it, as it is used in medical diagnosis, credit evaluation, email junk categorization, etc.

## Formulation behind Logistic Regression

Logistic Regression calculates a raw model output, then transforms it using the sigmoid function, mapping it to a range between 0 and 1, thus making it a probability. The sigmoid function can be defined as $S(x) = \frac{1}{1+e^{-x}}$. This can thus be implemented as:

```python
def sigmoid(z):
    return 1 / (1+np.exp(-z))
```

The mathematical form of logistic regression can be expressed as follows:
$$P(Y=1|x) = \frac{1}{1+e^{-(\beta_0+\beta_1x)}}$$
where $P(Y=1|x)$ is the probability of event $Y=1$ given $x$, $\beta_0$ and $\beta_1$ are parameters of the model, $x$ is the input variable and $\beta_0+\beta_1x$ is the linear combination of parameters and features.

_Log-Likelihood_ in Logistic Regression plays a similar role to the _Least Squares method_ in Linear Regression. A maximum likelihood estimation method estimates parameters that maximize the likelihood of making the observations we collected. In Logistic Regression, we seek to maximize the log-likelihood.

The cost function for a single training instance in logistic regression can be expressed as $-[y\log{(\hat p)+(1-y)\log{(1-\hat p)}}]$ where $\hat p$ denotes the predicted probability.

```python
def cost_function(h, y): # h = sigmoid(z) where z = X @ theta
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

def logistic_regression(X, y, num_iterations, learning_rate): 
    # Add intercept to X 
    intercept = np.ones((X.shape[0], 1)) 
    X = np.concatenate((intercept, X), axis=1) 
    
    # Weights initialization 
    theta = np.zeros(X.shape[1]) 
    for i in range(num_iterations): 
        z = np.dot(X, theta) 
        h = sigmoid(z) 
        gradient = np.dot(X.T, (h - y)) / y.size 
        theta -= learning_rate * gradient 
        
        z = np.dot(X, theta) 
        h = sigmoid(z) 
        loss = cost_function(h, y) 

        if i % 10000 == 0:
            print(f'Loss: {loss}\t') 
    
    return theta

def predict_prob(X, theta):
    # Add intercept to X
    intercept = np.ones((X.shape[0], 1))
    X = np.concatenate((intercept, X), axis=1)
    return sigmoid(np.dot(X, theta))

def predict(X, theta, threshold=0.5):
    return predict_prob(X, theta) >= threshold
```

## Differences Between Logistic and Linear Regression

### Key Differences

- **logistic regression is used for categorization whereas linear regression is used for regression problems**. This is the most significant difference between the two. In logistic regression, when given x and hyperparameter $\theta$, we could get the expectation value of the $y$ values to predict the categorization of the values. On the other hand, in linear regression, one is solving $y' = \theta^Tx$, which is the approximate of the real relationship of $y = \theta^Tx+\epsilon$ where $\epsilon$ corresponds to the system error.

- The actual logistic regression equation can be formulated via $\log{\frac{p}{1-p}}=\theta^Tx$, where $p=P(y=1|x)$, corresponding to given x the probability of y being positive. **Thus the most important difference between logistic regression and linear regression would be that the logistic regression $y$s are discretized whereas the linear regression $y$s are continuous.** When $x$ and $\theta$ are given, logistic regression can also be seen as generalized linear models where $y$ follows the binary distribution, whereas when using least-squares for linear regression we view $y$ follows the normal distribution.

### Similarities

- They both used maximum likelihood estimation for modeling the training data.
- They both could use gradient descent for getting the hyperparameters, and it is also a common strategy that all the supervised learning methods use.

## General Logic Behind Regression

```python
Inputs: X (N×d), y (N,), model ∈ {"linear","logistic"}
Hyperparams: learning_rate (lr), lambda (L2), max_iters, tol, patience
Prep:
  Xb = concat([ones(N,1), X])        # add bias column
  w = zeros(d+1)                     # includes bias at index 0
  mask = [0, 1, 1, ..., 1]           # no L2 on bias

For t in 1..max_iters:
  z = Xb @ w
  if model == "linear":
      pred = z
      loss_data = (1/(2N)) * sum((pred - y)^2)
  else:  # logistic
      pred = sigmoid(z)              # clip to [eps, 1-eps] for stability
      loss_data = -(1/N) * sum(y*log(pred) + (1-y)*log(1-pred))

  loss = loss_data + lambda * sum((w*mask)^2)
  grad = (1/N) * (Xb.T @ (pred - y)) + 2*lambda*(w*mask)
  w = w - learning_rate * grad
  if norm(grad) < tol or early_stopping_on_val(loss): break

Return w
```

## Binomial vs Normal Distribution

The main difference between a binomial distribution and a normal distribution lies in the type of data they describe: **binomial distributions deal with discrete data from a fixed number of trials, while normal distributions describe continuous data that tends to cluster around a mean**. Binomial distributions are characterized by a fixed number of trials, each with two possible outcomes (success or failure), while normal distributions are continuous, symmetric, and have a bell-shaped curve.

## Related Topics

- **[Linear Regression](linear_regression.md)** - Understanding the differences and similarities
- **[Decision Trees](decision_trees.md)** - Alternative classification algorithm
- **[Model Evaluation](../model_evaluation/metrics_and_validation.md)** - Evaluating classification performance
- **[Regularization](../regularization/l1_l2_regularization.md)** - Regularizing logistic regression
