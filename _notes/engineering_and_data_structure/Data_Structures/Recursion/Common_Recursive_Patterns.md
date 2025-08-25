---
title: Common Recursive Patterns
---

# Common Recursive Patterns

This guide covers the most common recursive patterns you'll encounter in programming interviews and real-world applications. Understanding these patterns helps you recognize when and how to apply recursion effectively.

## Pattern 1: Linear Recursion

Process elements one by one, making a single recursive call.

### Template:
```python
def linear_recursion(data, index=0):
    # Base case
    if index >= len(data):
        return base_value
    
    # Process current element
    current_result = process(data[index])
    
    # Recursive call
    rest_result = linear_recursion(data, index + 1)
    
    # Combine results
    return combine(current_result, rest_result)
```

### Examples:

**Array Sum:**
```python
def array_sum(arr, index=0):
    if index >= len(arr):
        return 0
    return arr[index] + array_sum(arr, index + 1)
```

**String Length:**
```python
def string_length(s, index=0):
    if index >= len(s):
        return 0
    return 1 + string_length(s, index + 1)
```

**Find Maximum:**
```python
def find_max(arr, index=0):
    if index >= len(arr):
        return float('-inf')
    
    current = arr[index]
    rest_max = find_max(arr, index + 1)
    
    return max(current, rest_max)
```

## Pattern 2: Binary Recursion

Make two recursive calls, typically dividing the problem in half.

### Template:
```python
def binary_recursion(data, start, end):
    # Base case
    if start > end:
        return base_value
    
    # Divide
    mid = (start + end) // 2
    
    # Conquer
    left_result = binary_recursion(data, start, mid)
    right_result = binary_recursion(data, mid + 1, end)
    
    # Combine
    return combine(left_result, right_result)
```

### Examples:

**Binary Search:**
```python
def binary_search(arr, target, left=0, right=None):
    if right is None:
        right = len(arr) - 1
    
    if left > right:
        return -1
    
    mid = (left + right) // 2
    
    if arr[mid] == target:
        return mid
    elif arr[mid] > target:
        return binary_search(arr, target, left, mid - 1)
    else:
        return binary_search(arr, target, mid + 1, right)
```

**Merge Sort:**
```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)
```

**Maximum Subarray (Divide & Conquer):**
```python
def max_subarray(arr, left=0, right=None):
    if right is None:
        right = len(arr) - 1
    
    if left == right:
        return arr[left]
    
    mid = (left + right) // 2
    
    left_max = max_subarray(arr, left, mid)
    right_max = max_subarray(arr, mid + 1, right)
    cross_max = max_crossing_sum(arr, left, mid, right)
    
    return max(left_max, right_max, cross_max)
```

## Pattern 3: Multiple Recursion

Make multiple recursive calls (more than two), often for tree-like structures.

### Template:
```python
def multiple_recursion(node):
    # Base case
    if node is None:
        return base_value
    
    # Process current node
    current_result = process(node)
    
    # Recursive calls for each child
    child_results = []
    for child in node.children:
        child_results.append(multiple_recursion(child))
    
    # Combine all results
    return combine(current_result, child_results)
```

### Examples:

**Tree Traversal:**
```python
def preorder_traversal(root):
    if root is None:
        return []
    
    result = [root.val]
    result.extend(preorder_traversal(root.left))
    result.extend(preorder_traversal(root.right))
    
    return result
```

**Directory Size Calculation:**
```python
def calculate_directory_size(directory):
    if directory.is_file():
        return directory.size
    
    total_size = 0
    for item in directory.contents:
        total_size += calculate_directory_size(item)
    
    return total_size
```

## Pattern 4: Tail Recursion

The recursive call is the last operation in the function.

### Template:
```python
def tail_recursion(data, accumulator=initial_value):
    # Base case
    if termination_condition:
        return accumulator
    
    # Update accumulator and make recursive call
    new_accumulator = update(accumulator, current_data)
    return tail_recursion(remaining_data, new_accumulator)
```

### Examples:

**Factorial (Tail Recursive):**
```python
def factorial_tail(n, accumulator=1):
    if n <= 1:
        return accumulator
    return factorial_tail(n - 1, n * accumulator)
```

**Sum Array (Tail Recursive):**
```python
def sum_tail(arr, index=0, accumulator=0):
    if index >= len(arr):
        return accumulator
    return sum_tail(arr, index + 1, accumulator + arr[index])
```

**List Reversal (Tail Recursive):**
```python
def reverse_tail(arr, index=0, result=None):
    if result is None:
        result = []
    
    if index >= len(arr):
        return result
    
    return reverse_tail(arr, index + 1, [arr[index]] + result)
```

## Pattern 5: Backtracking

Explore all possible solutions by trying each option and backtracking when needed.

### Template:
```python
def backtrack(current_solution, remaining_choices):
    # Base case: found complete solution
    if is_complete(current_solution):
        return [current_solution[:]]  # Return copy
    
    solutions = []
    
    # Try each possible choice
    for choice in remaining_choices:
        if is_valid(current_solution, choice):
            # Make choice
            current_solution.append(choice)
            
            # Recursively explore
            new_remaining = get_remaining_choices(remaining_choices, choice)
            solutions.extend(backtrack(current_solution, new_remaining))
            
            # Backtrack (undo choice)
            current_solution.pop()
    
    return solutions
```

### Examples:

**Generate All Permutations:**
```python
def permutations(arr):
    if len(arr) <= 1:
        return [arr]
    
    result = []
    for i in range(len(arr)):
        current = arr[i]
        remaining = arr[:i] + arr[i+1:]
        
        for perm in permutations(remaining):
            result.append([current] + perm)
    
    return result
```

**N-Queens Problem:**
```python
def solve_n_queens(n):
    def is_safe(positions, row, col):
        for r, c in enumerate(positions):
            if c == col or abs(r - row) == abs(c - col):
                return False
        return True
    
    def backtrack(row, positions):
        if row == n:
            return [positions[:]]
        
        solutions = []
        for col in range(n):
            if is_safe(positions, row, col):
                positions.append(col)
                solutions.extend(backtrack(row + 1, positions))
                positions.pop()
        
        return solutions
    
    return backtrack(0, [])
```

**Subset Generation:**
```python
def generate_subsets(arr):
    def backtrack(index, current_subset):
        if index >= len(arr):
            return [current_subset[:]]
        
        # Include current element
        current_subset.append(arr[index])
        with_current = backtrack(index + 1, current_subset)
        current_subset.pop()
        
        # Exclude current element
        without_current = backtrack(index + 1, current_subset)
        
        return with_current + without_current
    
    return backtrack(0, [])
```

## Pattern 6: Memoization (Dynamic Programming)

Store results of subproblems to avoid redundant calculations.

### Template:
```python
def memoized_recursion(problem, memo={}):
    # Check if already computed
    if problem in memo:
        return memo[problem]
    
    # Base case
    if base_condition:
        return base_value
    
    # Recursive case
    result = compute_result(problem)
    
    # Store result
    memo[problem] = result
    return result
```

### Examples:

**Fibonacci with Memoization:**
```python
def fibonacci_memo(n, memo={}):
    if n in memo:
        return memo[n]
    
    if n <= 1:
        return n
    
    memo[n] = fibonacci_memo(n - 1, memo) + fibonacci_memo(n - 2, memo)
    return memo[n]

# Using decorator
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci_cached(n):
    if n <= 1:
        return n
    return fibonacci_cached(n - 1) + fibonacci_cached(n - 2)
```

**Longest Common Subsequence:**
```python
def lcs(s1, s2, i=0, j=0, memo={}):
    if (i, j) in memo:
        return memo[(i, j)]
    
    if i >= len(s1) or j >= len(s2):
        return 0
    
    if s1[i] == s2[j]:
        result = 1 + lcs(s1, s2, i + 1, j + 1, memo)
    else:
        result = max(lcs(s1, s2, i + 1, j, memo), 
                    lcs(s1, s2, i, j + 1, memo))
    
    memo[(i, j)] = result
    return result
```

## Pattern 7: Tree Recursion

Specifically for tree data structures.

### Template:
```python
def tree_recursion(root):
    # Base case: empty tree
    if root is None:
        return base_value
    
    # Process current node
    current_result = process(root)
    
    # Recursive calls for children
    left_result = tree_recursion(root.left)
    right_result = tree_recursion(root.right)
    
    # Combine results
    return combine(current_result, left_result, right_result)
```

### Examples:

**Tree Height:**
```python
def tree_height(root):
    if root is None:
        return 0
    
    left_height = tree_height(root.left)
    right_height = tree_height(root.right)
    
    return 1 + max(left_height, right_height)
```

**Tree Sum:**
```python
def tree_sum(root):
    if root is None:
        return 0
    
    return root.val + tree_sum(root.left) + tree_sum(root.right)
```

**Validate Binary Search Tree:**
```python
def is_valid_bst(root, min_val=float('-inf'), max_val=float('inf')):
    if root is None:
        return True
    
    if root.val <= min_val or root.val >= max_val:
        return False
    
    return (is_valid_bst(root.left, min_val, root.val) and 
            is_valid_bst(root.right, root.val, max_val))
```

## Pattern 8: String Recursion

Common patterns for string processing.

### Examples:

**Palindrome Check:**
```python
def is_palindrome(s, left=0, right=None):
    if right is None:
        right = len(s) - 1
    
    if left >= right:
        return True
    
    if s[left] != s[right]:
        return False
    
    return is_palindrome(s, left + 1, right - 1)
```

**String Permutations:**
```python
def string_permutations(s):
    if len(s) <= 1:
        return [s]
    
    result = []
    for i, char in enumerate(s):
        remaining = s[:i] + s[i+1:]
        for perm in string_permutations(remaining):
            result.append(char + perm)
    
    return result
```

**Edit Distance:**
```python
def edit_distance(s1, s2, i=0, j=0, memo={}):
    if (i, j) in memo:
        return memo[(i, j)]
    
    if i >= len(s1):
        return len(s2) - j
    if j >= len(s2):
        return len(s1) - i
    
    if s1[i] == s2[j]:
        result = edit_distance(s1, s2, i + 1, j + 1, memo)
    else:
        insert = 1 + edit_distance(s1, s2, i, j + 1, memo)
        delete = 1 + edit_distance(s1, s2, i + 1, j, memo)
        replace = 1 + edit_distance(s1, s2, i + 1, j + 1, memo)
        result = min(insert, delete, replace)
    
    memo[(i, j)] = result
    return result
```

## Pattern Recognition Guide

### How to Identify Patterns:

1. **Linear Recursion**: Processing elements sequentially, one recursive call
2. **Binary Recursion**: Dividing problem in half, two recursive calls
3. **Multiple Recursion**: Tree-like structure, multiple recursive calls
4. **Tail Recursion**: Accumulating result, recursive call is last operation
5. **Backtracking**: Exploring all possibilities, need to undo choices
6. **Memoization**: Overlapping subproblems, repeated calculations
7. **Tree Recursion**: Working with tree data structures
8. **String Recursion**: Character-by-character or substring processing

### Decision Framework:

```python
# Ask these questions:
# 1. Can the problem be broken into smaller similar problems?
# 2. Is there a natural base case?
# 3. Are there overlapping subproblems? (use memoization)
# 4. Do I need to explore all possibilities? (use backtracking)
# 5. Am I working with tree/graph structures? (use tree recursion)
# 6. Can I accumulate the result? (use tail recursion)
```

## Common Mistakes to Avoid

1. **Missing Base Case:**
   ```python
   # Wrong
   def factorial(n):
       return n * factorial(n - 1)  # No base case
   
   # Correct
   def factorial(n):
       if n <= 1:
           return 1
       return n * factorial(n - 1)
   ```

2. **Incorrect Progress:**
   ```python
   # Wrong - infinite recursion
   def countdown(n):
       print(n)
       countdown(n)  # n doesn't change
   
   # Correct
   def countdown(n):
       if n <= 0:
           return
       print(n)
       countdown(n - 1)
   ```

3. **Inefficient Recursion:**
   ```python
   # Inefficient - exponential time
   def fibonacci(n):
       if n <= 1:
           return n
       return fibonacci(n - 1) + fibonacci(n - 2)
   
   # Better - with memoization
   @lru_cache(maxsize=None)
   def fibonacci(n):
       if n <= 1:
           return n
       return fibonacci(n - 1) + fibonacci(n - 2)
   ```

## Practice Problems by Pattern

### Linear Recursion:
- Array sum, product, maximum
- String reversal, character counting
- Linked list operations

### Binary Recursion:
- Binary search
- Merge sort, quick sort
- Tree height, tree sum

### Backtracking:
- N-Queens, Sudoku solver
- Generate permutations, combinations
- Path finding in mazes

### Memoization:
- Fibonacci, climbing stairs
- Longest common subsequence
- Coin change problem

### Tree Recursion:
- Tree traversals
- Binary search tree validation
- Path sum problems

## Related Topics

- **[Recursion Fundamentals](Recursion_Fundamentals.md)** - Basic recursion concepts
- **[Recursive Algorithms](Recursive_Algorithms.md)** - Common recursive algorithms
- **[Recursion vs Iteration](Recursion_vs_Iteration.md)** - When to choose each approach
