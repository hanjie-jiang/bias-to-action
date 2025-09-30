---
title: Calculus & Gradient Descent
---
# Introduction
Calculus provides the machinery to compute derivatives, which are essential for optimizing neural networks. Understanding gradients and how gradient descent navigates a loss landscape is critical for debugging and improving models.

# Log Computations

In computer science and algorithm analysis, logarithms are commonly used in complexity analysis. While mathematicians often use the natural logarithm $\ln(x) = \log_e(x)$, computer scientists frequently work with $\log_2(x)$ due to binary operations. However, for asymptotic analysis, the base doesn't matter since logarithms of different bases differ only by a constant factor.

## Examples of Logarithmic Properties

### Example 1: Logarithm of Powers

$$\log(n^{100}) = 100 \cdot \log(n) = \Theta(\log(n))$$

**Explanation:** The constant factor 100 doesn't affect the asymptotic complexity class, so $100 \log(n)$ is still $\Theta(\log(n))$.

### Example 2: Change of Base Formula

$$\log_5(n) = \frac{\log(n)}{\log(5)} = \Theta(\log(n))$$

**Explanation:** Using the change of base formula, we see that $\log_5(n)$ differs from $\log(n)$ only by the constant factor $\frac{1}{\log(5)}$, so they're in the same asymptotic class.

### Example 3: Nested Logarithms

$$\log((\log(n))^{100}) = 100 \cdot \log(\log(n)) = \Theta(\log(\log(n)))$$

**Explanation:** Even with the constant factor 100, this is still in the $\Theta(\log(\log(n)))$ complexity class. Note how nested logarithms create a different, slower-growing complexity class.


#### Knowledge Points
- Partial derivatives
- Chain rule
- Gradients as slopes in high dimensions
- Gradient descent algorithm & intuition
- Visualization of gradient descent in 2D
