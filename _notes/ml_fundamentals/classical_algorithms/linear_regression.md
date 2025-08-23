---
title: Linear Regression
---

# Linear Regression

There are two central provinces in the world of regression: simple linear regression and multiple linear regression.

## Formula of Simple Linear Regression

The formula of linear regression can be represented as $$y=c+m\cdot x$$

The formula revolves around minimizing residuals. Imagine residuals as the distance between the actual and predicted values of the dependent variable $y$:

$$m = \frac{\sum_{i=1}^N{(x_i-\bar x)(y_i-\bar y)}}{\sum_{i=1}^N(x_i-\bar x)^2}$$

and the constant corresponds to $c=\bar y - m \cdot\bar x$.

```python
import numpy as np

# Step 1: Get the data set
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Step 2: Compute the mean of the X and y
mean_x = np.mean(x)
mean_y = np.mean(y)

# Step 3: Calculate the coefficients
m = np.sum((x - mean_x) * (y - mean_y)) / np.sum((x - mean_x) ** 2)
c = mean_y - m * mean_x

# Voila! We have our model
print(f"Model: y = {c} + {m}*x")  # Output: Model: y= 2.2 + 0.6*x
```

## Formula of Multiple Linear Regression

For simple linear regression formula, we have $y=\beta_0 + \beta_1x$, for multiple linear regression, we add multiple independent variables $x_1, x_2, ... , x_m$. Suppose we had n data points, each with m features, then X would be like:

$$\mathbf{X}=\begin{bmatrix}  
1 & x_{1,1} & x_{1,2} & ... & x_{1,m} \\  
1 & x_{2,1} & x_{2,2} & ... & x_{2,m} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_{n,1} & x_{n,2} & ... & x_{n,m} \\
\end{bmatrix} \in \mathbb{R^{n\times (m+1)}}, \mathbf{y} = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{bmatrix} \in \mathbb{R^{n\times 1}}, \mathbf{\beta} = \begin{bmatrix} \beta_0 \\ \beta_1 \\ \vdots \\ \beta_m \end{bmatrix} \in  \mathbb{R^{(m+1)\times 1}}$$

Each row represents the m features for a single data point. The first column with $\mathbf{1}$s are the bias / intercept of each equation. The normal equation would be of form

$$\beta = (X^T X)^{-1}X^Ty$$

The predicted $\hat y$ values can be represented as

$$\hat y = (1 \cdot \beta_0)+(\beta_1 \cdot x_1) + (\beta_2 \cdot x_2) + \dots + (\beta_m \cdot x_m)$$

To calculate all the predictions at once, we take the dot product of $X$ and $\beta$:

$$\mathbf{y} = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{bmatrix} = X\cdot \beta =\begin{bmatrix}  
1 & x_{1,1} & x_{1,2} & ... & x_{1,m} \\  
1 & x_{2,1} & x_{2,2} & ... & x_{2,m} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_{n,1} & x_{n,2} & ... & x_{n,m} \\
\end{bmatrix} \begin{bmatrix} \beta_0 \\ \beta_1 \\ \vdots \\ \beta_m \end{bmatrix}$$

## Linear Regression Model Evaluation

### Coefficient of Determination ($R^2$ Score)

$$R^2=1-\frac{SS_\text{residuals}}{SS_\text{total}} = 1 - \frac{\sum_{i=1}^n(y_i - \hat y_i)^2}{\sum_{i=1}^n(y_i - \bar y_i)^2}$$

Where $SS_\text{residuals}$ denotes the residual sum of squares for predictions and $SS_\text{total}$ denotes the total sum of squares from actual values. A higher R-squared value / closer to 1 indicates a good model fit.

```python
import numpy as np
# given data
housing_data = np.array(\[\[[1800, 3], [2400, 4], [1416, 2], [3000, 5]\])
prices = np.array([350000, 475000, 230000, 640000])

# adding 1s to our matrix
# ones = np.ones(shape=(len(housing_data), 1))
# X = np.append(ones, housing_data, axis=1)
X = np.c_[np.ones((len(housing_data),1)),X] # add bias parameter to X

# calculating coefficients
coefficients = np.linalg.inv(X.T @ X) @ X.T @ prices

# predicting prices
predicted_prices = X @ coefficients

# calculating residuals
residuals = prices - predicted_prices

# calculating total sum of squares
sst = np.sum((prices - np.mean(prices)) ** 2)

# calculating residual sum of squares
ssr = np.sum(residuals ** 2)

# calculating R^2
r2 = 1 - (ssr/sst)

print("Coefficients:", coefficients)
print("Predicted prices:", predicted_prices)
print("R^2:", r2)
```

## Gradient Descent

**Gradient descent** is an iterative optimization algorithm for minimizing a function, usually a loss function, quantifying the disparity between predicted and actual results. The goal of gradient descent is to find the parameters that minimize the value of the loss function.

Gradient descent derives its name from its working mechanism: taking _descents_ along the _gradient_. It operates in several iterative steps as follows:

1. Choose random values for initial parameters.
2. Calculate the cost (the difference between actual and predicted value).
3. Compute the gradient (the steepest slope of the function around that point).
4. Update the parameters using the gradient.
5. Repeat steps 2 to 4 until we reach an acceptable error rate or exhaust the maximum iterations.

A vital component of gradient descent is the learning rate, which determines the size of the descent towards the optimum solution.

The first step is to calculate the cost function, which takes the form of $$J(X, y, \theta) = \frac{1}{m}\sum_{i=1}^m(X\cdot \theta - y_i)^2$$ where J is the cost, X is the data, y is the actual values and $\theta$ is the parameters, $m$ is the length of $y$. It is calculating the mean square error.

```python
import numpy as np

def cost(X, y, theta):
	m = len(y)
	predictions = X @ theta
	cost = (1/m) * np.sum((predictions - y) ** 2)
	return cost
```

The second step is to compute the gradient descent function, which will be updated in the iterative loop:

$$\theta:=\theta-\alpha\frac{1}{m}X^T\cdot(X\cdot \theta - y)$$

Here $\alpha$ is the learning rate, which determines the size of steps in the descent and $X^T$ is the transpose of data, which should have been multiplied by 2 but as we take the derivative of the mean squared error we could also consider it to be included as part of the learning rate $\alpha$.

```python
def gradient_descent(X, y, theta, alpha, threshold=0.01):
    m = len(y)
    cost_history = []
    prev_cost = float('inf')
    iterations = 0
    while True:
        prediction = X.dot(theta)
        theta = theta - (alpha / m) * X.T.dot(prediction - y)
        cost = (1/(2*m)) * np.sum((prediction - y) ** 2)
        cost_history.append(cost)
        if abs(prev_cost - cost) < threshold:
            break
        prev_cost = cost
        iterations += 1
    return theta, cost_history, iterations
```

## Related Topics

- **[Decision Trees](decision_trees.md)** - Alternative regression algorithm
- **[Model Evaluation](../model_evaluation/metrics_and_validation.md)** - Evaluating regression models
- **[Regularization](../regularization/l1_l2_regularization.md)** - Regularizing linear regression
