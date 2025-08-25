---
title: Recursion Fundamentals
---

# Recursion Fundamentals

Recursion is a programming technique where a function calls itself to solve a problem by breaking it down into smaller, similar subproblems.

## Core Concepts

### What is Recursion?

Recursion occurs when a function calls itself with modified parameters until it reaches a base case. It's a fundamental problem-solving approach that mirrors mathematical induction.

### Essential Components

Every recursive function must have:

1. **Base Case**: A condition that stops the recursion
2. **Recursive Case**: The function calling itself with modified parameters
3. **Progress Toward Base Case**: Each recursive call must move closer to the base case

## Basic Structure

```python
def recursive_function(parameters):
    # Base case
    if base_condition:
        return base_value
    
    # Recursive case
    return recursive_function(modified_parameters)
```

## Simple Examples

### Example 1: Factorial

```python
def factorial(n):
    # Base case
    if n <= 1:
        return 1
    
    # Recursive case
    return n * factorial(n - 1)

# Usage
print(factorial(5))  # Output: 120
```

**How it works:**
- factorial(5) = 5 * factorial(4)
- factorial(4) = 4 * factorial(3)
- factorial(3) = 3 * factorial(2)
- factorial(2) = 2 * factorial(1)
- factorial(1) = 1 (base case)

### Example 2: Fibonacci Sequence

```python
def fibonacci(n):
    # Base cases
    if n <= 1:
        return n
    
    # Recursive case
    return fibonacci(n - 1) + fibonacci(n - 2)

# Usage
print(fibonacci(6))  # Output: 8
```

### Example 3: Sum of Array

```python
def array_sum(arr, index=0):
    # Base case
    if index >= len(arr):
        return 0
    
    # Recursive case
    return arr[index] + array_sum(arr, index + 1)

# Usage
numbers = [1, 2, 3, 4, 5]
print(array_sum(numbers))  # Output: 15
```

## Types of Recursion

### 1. Linear Recursion
Function makes one recursive call per execution.

```python
def countdown(n):
    if n <= 0:
        print("Done!")
        return
    print(n)
    countdown(n - 1)
```

### 2. Binary Recursion
Function makes two recursive calls per execution.

```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
```

### 3. Tail Recursion
Recursive call is the last operation in the function.

```python
def factorial_tail(n, accumulator=1):
    if n <= 1:
        return accumulator
    return factorial_tail(n - 1, n * accumulator)
```

### 4. Multiple Recursion
Function makes more than two recursive calls.

```python
def tree_traversal(node):
    if node is None:
        return
    
    # Process current node
    process(node)
    
    # Recursive calls for each child
    for child in node.children:
        tree_traversal(child)
```

## Recursion vs Mathematical Induction

Recursion closely follows the principle of mathematical induction:

1. **Base Case** = Prove for the smallest case
2. **Inductive Step** = If true for k, prove for k+1
3. **Recursive Step** = Assume solution works for smaller problems

## Memory and Stack

### Call Stack
Each recursive call adds a new frame to the call stack:

```python
def print_stack_depth(n, depth=0):
    print(f"Depth {depth}: n = {n}")
    if n <= 0:
        return
    print_stack_depth(n - 1, depth + 1)

print_stack_depth(3)
```

Output:
```
Depth 0: n = 3
Depth 1: n = 2
Depth 2: n = 1
Depth 3: n = 0
```

### Stack Overflow
Too many recursive calls can cause stack overflow:

```python
# This will cause stack overflow for large n
def bad_recursion(n):
    if n == 0:
        return 0
    return 1 + bad_recursion(n - 1)

# Python's default recursion limit is around 1000
import sys
print(sys.getrecursionlimit())  # Usually 1000
```

## Best Practices

### 1. Always Define Base Case
```python
# Good
def power(base, exp):
    if exp == 0:
        return 1
    return base * power(base, exp - 1)

# Bad - no base case leads to infinite recursion
def bad_power(base, exp):
    return base * bad_power(base, exp - 1)
```

### 2. Ensure Progress Toward Base Case
```python
# Good - exp decreases each call
def power(base, exp):
    if exp == 0:
        return 1
    return base * power(base, exp - 1)

# Bad - exp never changes
def bad_power(base, exp):
    if exp == 0:
        return 1
    return base * bad_power(base, exp)  # exp never decreases
```

### 3. Consider Iterative Alternatives
```python
# Recursive (can cause stack overflow)
def fibonacci_recursive(n):
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)

# Iterative (more efficient)
def fibonacci_iterative(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
```

## When to Use Recursion

### Good Use Cases:
- Tree and graph traversal
- Divide and conquer algorithms
- Problems with recursive structure (fractals, nested data)
- Backtracking problems

### Avoid Recursion When:
- Simple iterative solution exists
- Deep recursion expected (stack overflow risk)
- Performance is critical (iterative often faster)

## Common Pitfalls

### 1. Missing Base Case
```python
# Will run forever
def infinite_recursion(n):
    return infinite_recursion(n - 1)
```

### 2. Incorrect Base Case
```python
# Base case never reached for negative numbers
def factorial(n):
    if n == 1:  # Should be n <= 1
        return 1
    return n * factorial(n - 1)
```

### 3. No Progress Toward Base Case
```python
# n never decreases
def no_progress(n):
    if n == 0:
        return 0
    return no_progress(n)  # Should be no_progress(n - 1)
```

## Related Topics

- **[Recursive Algorithms](Recursive_Algorithms.md)** - Common recursive algorithms
- **[Recursion vs Iteration](Recursion_vs_Iteration.md)** - When to choose each approach
- **[Common Recursive Patterns](Common_Recursive_Patterns.md)** - Problem-solving patterns
