---
title: Recursion vs Iteration
---

# Recursion vs Iteration

Understanding when to use recursion versus iteration is crucial for writing efficient and maintainable code. This guide compares both approaches and provides decision-making criteria.

## Fundamental Differences

### Recursion
- Function calls itself with modified parameters
- Uses the call stack to maintain state
- Natural for problems with recursive structure
- Can be more elegant and readable

### Iteration
- Uses loops (for, while) to repeat operations
- Uses variables to maintain state
- Generally more memory efficient
- Often faster due to no function call overhead

## Side-by-Side Comparisons

### Example 1: Factorial

**Recursive Implementation:**
```python
def factorial_recursive(n):
    if n <= 1:
        return 1
    return n * factorial_recursive(n - 1)

# Time: O(n), Space: O(n) - due to call stack
```

**Iterative Implementation:**
```python
def factorial_iterative(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

# Time: O(n), Space: O(1) - constant space
```

### Example 2: Fibonacci Sequence

**Naive Recursive (Inefficient):**
```python
def fibonacci_recursive(n):
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)

# Time: O(2^n), Space: O(n) - exponential time!
```

**Iterative Implementation:**
```python
def fibonacci_iterative(n):
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# Time: O(n), Space: O(1) - much more efficient
```

**Optimized Recursive (with memoization):**
```python
def fibonacci_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    
    memo[n] = fibonacci_memo(n - 1, memo) + fibonacci_memo(n - 2, memo)
    return memo[n]

# Time: O(n), Space: O(n) - efficient but uses more memory
```

### Example 3: Array Sum

**Recursive Implementation:**
```python
def sum_recursive(arr, index=0):
    if index >= len(arr):
        return 0
    return arr[index] + sum_recursive(arr, index + 1)

# Time: O(n), Space: O(n)
```

**Iterative Implementation:**
```python
def sum_iterative(arr):
    total = 0
    for num in arr:
        total += num
    return total

# Time: O(n), Space: O(1)
```

### Example 4: Binary Tree Traversal

**Recursive Implementation (Natural):**
```python
def inorder_recursive(root):
    if root is None:
        return []
    
    result = []
    result.extend(inorder_recursive(root.left))
    result.append(root.val)
    result.extend(inorder_recursive(root.right))
    return result

# Time: O(n), Space: O(h) where h is tree height
```

**Iterative Implementation (Using Stack):**
```python
def inorder_iterative(root):
    result = []
    stack = []
    current = root
    
    while stack or current:
        # Go to leftmost node
        while current:
            stack.append(current)
            current = current.left
        
        # Process current node
        current = stack.pop()
        result.append(current.val)
        
        # Move to right subtree
        current = current.right
    
    return result

# Time: O(n), Space: O(h) - similar space complexity
```

## Performance Comparison

### Time Complexity
| Problem | Recursive | Iterative |
|---------|-----------|-----------|
| Factorial | O(n) | O(n) |
| Fibonacci (naive) | O(2ⁿ) | O(n) |
| Fibonacci (memo) | O(n) | O(n) |
| Array Sum | O(n) | O(n) |
| Binary Search | O(log n) | O(log n) |
| Tree Traversal | O(n) | O(n) |

### Space Complexity
| Problem | Recursive | Iterative |
|---------|-----------|-----------|
| Factorial | O(n) | O(1) |
| Fibonacci (naive) | O(n) | O(1) |
| Fibonacci (memo) | O(n) | O(1) |
| Array Sum | O(n) | O(1) |
| Binary Search | O(log n) | O(1) |
| Tree Traversal | O(h) | O(h) |

## When to Use Recursion

### Ideal Use Cases:

1. **Tree and Graph Problems**
   ```python
   def tree_height(root):
       if root is None:
           return 0
       return 1 + max(tree_height(root.left), tree_height(root.right))
   ```

2. **Divide and Conquer Algorithms**
   ```python
   def merge_sort(arr):
       if len(arr) <= 1:
           return arr
       
       mid = len(arr) // 2
       left = merge_sort(arr[:mid])
       right = merge_sort(arr[mid:])
       return merge(left, right)
   ```

3. **Backtracking Problems**
   ```python
   def solve_n_queens(n, row=0, positions=[]):
       if row == n:
           return [positions[:]]
       
       solutions = []
       for col in range(n):
           if is_safe(positions, row, col):
               positions.append(col)
               solutions.extend(solve_n_queens(n, row + 1, positions))
               positions.pop()
       
       return solutions
   ```

4. **Mathematical Definitions**
   ```python
   def gcd(a, b):
       if b == 0:
           return a
       return gcd(b, a % b)
   ```

5. **Nested Data Structures**
   ```python
   def flatten_nested_list(nested_list):
       result = []
       for item in nested_list:
           if isinstance(item, list):
               result.extend(flatten_nested_list(item))
           else:
               result.append(item)
       return result
   ```

## When to Use Iteration

### Ideal Use Cases:

1. **Simple Counting/Accumulation**
   ```python
   def count_even_numbers(arr):
       count = 0
       for num in arr:
           if num % 2 == 0:
               count += 1
       return count
   ```

2. **Linear Data Processing**
   ```python
   def find_max(arr):
       if not arr:
           return None
       
       max_val = arr[0]
       for num in arr[1:]:
           if num > max_val:
               max_val = num
       return max_val
   ```

3. **Performance-Critical Code**
   ```python
   def fibonacci_fast(n):
       if n <= 1:
           return n
       
       a, b = 0, 1
       for _ in range(2, n + 1):
           a, b = b, a + b
       return b
   ```

4. **Memory-Constrained Environments**
   ```python
   def sum_large_array(arr):
       total = 0
       for num in arr:
           total += num
       return total
   ```

## Converting Between Approaches

### Recursion to Iteration

Many recursive algorithms can be converted to iterative using a stack:

```python
# Recursive factorial
def factorial_recursive(n):
    if n <= 1:
        return 1
    return n * factorial_recursive(n - 1)

# Convert to iterative using explicit stack
def factorial_with_stack(n):
    if n <= 1:
        return 1
    
    stack = []
    while n > 1:
        stack.append(n)
        n -= 1
    
    result = 1
    while stack:
        result *= stack.pop()
    
    return result

# Or simple iterative version
def factorial_iterative(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
```

### Iteration to Recursion

Sometimes iterative solutions can be made recursive (though not always beneficial):

```python
# Iterative array search
def find_iterative(arr, target):
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1

# Convert to recursive
def find_recursive(arr, target, index=0):
    if index >= len(arr):
        return -1
    if arr[index] == target:
        return index
    return find_recursive(arr, target, index + 1)
```

## Decision Framework

### Choose Recursion When:
- Problem has natural recursive structure (trees, graphs)
- Divide and conquer approach fits well
- Backtracking is needed
- Mathematical definition is recursive
- Code clarity is more important than performance
- Working with nested/hierarchical data

### Choose Iteration When:
- Simple linear processing is needed
- Performance is critical
- Memory usage must be minimized
- Working with large datasets
- Stack overflow is a concern
- The iterative solution is simpler

### Hybrid Approach:
Sometimes combining both approaches works best:

```python
def tree_paths_iterative(root):
    """Find all root-to-leaf paths using iteration with stack"""
    if not root:
        return []
    
    paths = []
    stack = [(root, [root.val])]
    
    while stack:
        node, path = stack.pop()
        
        # If leaf node, add path to results
        if not node.left and not node.right:
            paths.append(path)
        
        # Add children to stack
        if node.right:
            stack.append((node.right, path + [node.right.val]))
        if node.left:
            stack.append((node.left, path + [node.left.val]))
    
    return paths
```

## Best Practices

### For Recursion:
1. Always define clear base cases
2. Ensure progress toward base case
3. Consider memoization for overlapping subproblems
4. Be aware of stack depth limitations
5. Test with edge cases (empty inputs, single elements)

### For Iteration:
1. Initialize variables properly
2. Ensure loop termination conditions
3. Handle edge cases explicitly
4. Use descriptive variable names
5. Consider loop invariants

### General Guidelines:
1. Profile your code when performance matters
2. Consider readability and maintainability
3. Document complex algorithms
4. Use the approach that best fits the problem domain
5. Don't convert just for the sake of converting

## Common Pitfalls

### Recursion Pitfalls:
```python
# Missing base case
def infinite_recursion(n):
    return infinite_recursion(n - 1)  # Will crash

# Incorrect base case
def factorial_wrong(n):
    if n == 1:  # Fails for n <= 0
        return 1
    return n * factorial_wrong(n - 1)

# No progress toward base case
def no_progress(n):
    if n == 0:
        return 0
    return no_progress(n)  # n never changes
```

### Iteration Pitfalls:
```python
# Off-by-one errors
def sum_array_wrong(arr):
    total = 0
    for i in range(len(arr) - 1):  # Misses last element
        total += arr[i]
    return total

# Infinite loops
def infinite_loop():
    i = 0
    while i < 10:
        print(i)
        # Forgot to increment i
```

## Related Topics

- **[Recursion Fundamentals](Recursion_Fundamentals.md)** - Basic recursion concepts
- **[Recursive Algorithms](Recursive_Algorithms.md)** - Common recursive algorithms
- **[Common Recursive Patterns](Common_Recursive_Patterns.md)** - Problem-solving patterns
