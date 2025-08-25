---
title: Recursive Algorithms
---

# Recursive Algorithms

This section covers common algorithms that are naturally implemented using recursion, including searching, sorting, and mathematical computations.

## Searching Algorithms

### Binary Search

Efficiently search a sorted array by repeatedly dividing the search space in half.

```python
def binary_search(arr, target, left=0, right=None):
    if right is None:
        right = len(arr) - 1
    
    # Base case: element not found
    if left > right:
        return -1
    
    mid = (left + right) // 2
    
    # Base case: element found
    if arr[mid] == target:
        return mid
    
    # Recursive cases
    if arr[mid] > target:
        return binary_search(arr, target, left, mid - 1)
    else:
        return binary_search(arr, target, mid + 1, right)

# Usage
sorted_array = [1, 3, 5, 7, 9, 11, 13, 15]
print(binary_search(sorted_array, 7))  # Output: 3
print(binary_search(sorted_array, 4))  # Output: -1
```

**Time Complexity**: O(log n)  
**Space Complexity**: O(log n) due to recursion stack

### Linear Search in Linked List

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def search_linked_list(head, target):
    # Base case: reached end of list
    if head is None:
        return False
    
    # Base case: found target
    if head.val == target:
        return True
    
    # Recursive case
    return search_linked_list(head.next, target)
```

## Sorting Algorithms

### Merge Sort

Divide and conquer sorting algorithm that recursively divides the array and merges sorted subarrays.

```python
def merge_sort(arr):
    # Base case: array with 0 or 1 element is already sorted
    if len(arr) <= 1:
        return arr
    
    # Divide
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    # Conquer (merge)
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    
    # Merge the two sorted arrays
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # Add remaining elements
    result.extend(left[i:])
    result.extend(right[j:])
    
    return result

# Usage
numbers = [64, 34, 25, 12, 22, 11, 90]
sorted_numbers = merge_sort(numbers)
print(sorted_numbers)  # Output: [11, 12, 22, 25, 34, 64, 90]
```

**Time Complexity**: O(n log n)  
**Space Complexity**: O(n)

### Quick Sort

Another divide and conquer sorting algorithm using a pivot element.

```python
def quick_sort(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1
    
    # Base case
    if low < high:
        # Partition the array
        pivot_index = partition(arr, low, high)
        
        # Recursively sort elements before and after partition
        quick_sort(arr, low, pivot_index - 1)
        quick_sort(arr, pivot_index + 1, high)
    
    return arr

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

# Usage
numbers = [64, 34, 25, 12, 22, 11, 90]
sorted_numbers = quick_sort(numbers.copy())
print(sorted_numbers)  # Output: [11, 12, 22, 25, 34, 64, 90]
```

**Average Time Complexity**: O(n log n)  
**Worst Case Time Complexity**: O(n²)  
**Space Complexity**: O(log n)

## Mathematical Algorithms

### Greatest Common Divisor (GCD)

Using Euclidean algorithm:

```python
def gcd(a, b):
    # Base case
    if b == 0:
        return a
    
    # Recursive case
    return gcd(b, a % b)

# Usage
print(gcd(48, 18))  # Output: 6
print(gcd(17, 13))  # Output: 1
```

### Power Function

```python
def power(base, exp):
    # Base cases
    if exp == 0:
        return 1
    if exp == 1:
        return base
    
    # Recursive case for even exponent (optimization)
    if exp % 2 == 0:
        half_power = power(base, exp // 2)
        return half_power * half_power
    else:
        return base * power(base, exp - 1)

# Usage
print(power(2, 10))  # Output: 1024
print(power(3, 4))   # Output: 81
```

**Time Complexity**: O(log n) with optimization, O(n) without

### Tower of Hanoi

Classic recursive problem:

```python
def hanoi(n, source, destination, auxiliary):
    if n == 1:
        print(f"Move disk 1 from {source} to {destination}")
        return
    
    # Move n-1 disks from source to auxiliary
    hanoi(n - 1, source, auxiliary, destination)
    
    # Move the largest disk from source to destination
    print(f"Move disk {n} from {source} to {destination}")
    
    # Move n-1 disks from auxiliary to destination
    hanoi(n - 1, auxiliary, destination, source)

# Usage
hanoi(3, 'A', 'C', 'B')
```

**Time Complexity**: O(2ⁿ)

## Tree Algorithms

### Tree Traversals

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def inorder_traversal(root):
    if root is None:
        return []
    
    result = []
    result.extend(inorder_traversal(root.left))
    result.append(root.val)
    result.extend(inorder_traversal(root.right))
    
    return result

def preorder_traversal(root):
    if root is None:
        return []
    
    result = [root.val]
    result.extend(preorder_traversal(root.left))
    result.extend(preorder_traversal(root.right))
    
    return result

def postorder_traversal(root):
    if root is None:
        return []
    
    result = []
    result.extend(postorder_traversal(root.left))
    result.extend(postorder_traversal(root.right))
    result.append(root.val)
    
    return result
```

### Tree Height

```python
def tree_height(root):
    # Base case: empty tree
    if root is None:
        return 0
    
    # Recursive case
    left_height = tree_height(root.left)
    right_height = tree_height(root.right)
    
    return 1 + max(left_height, right_height)
```

### Tree Size (Node Count)

```python
def tree_size(root):
    # Base case: empty tree
    if root is None:
        return 0
    
    # Recursive case
    return 1 + tree_size(root.left) + tree_size(root.right)
```

## String Algorithms

### Palindrome Check

```python
def is_palindrome(s, start=0, end=None):
    if end is None:
        end = len(s) - 1
    
    # Base cases
    if start >= end:
        return True
    
    if s[start] != s[end]:
        return False
    
    # Recursive case
    return is_palindrome(s, start + 1, end - 1)

# Usage
print(is_palindrome("racecar"))  # Output: True
print(is_palindrome("hello"))    # Output: False
```

### String Reversal

```python
def reverse_string(s):
    # Base case
    if len(s) <= 1:
        return s
    
    # Recursive case
    return s[-1] + reverse_string(s[:-1])

# Alternative implementation
def reverse_string_alt(s, index=0):
    # Base case
    if index >= len(s):
        return ""
    
    # Recursive case
    return reverse_string_alt(s, index + 1) + s[index]

# Usage
print(reverse_string("hello"))  # Output: "olleh"
```

## Backtracking Algorithms

### Generate All Permutations

```python
def permutations(arr):
    # Base case
    if len(arr) <= 1:
        return [arr]
    
    result = []
    for i in range(len(arr)):
        # Choose current element
        current = arr[i]
        remaining = arr[:i] + arr[i+1:]
        
        # Generate permutations of remaining elements
        for perm in permutations(remaining):
            result.append([current] + perm)
    
    return result

# Usage
print(permutations([1, 2, 3]))
# Output: All permutations of [1, 2, 3]
```

### Generate All Subsets

```python
def subsets(arr, index=0):
    # Base case
    if index >= len(arr):
        return [[]]
    
    # Get subsets without current element
    subsets_without_current = subsets(arr, index + 1)
    
    # Get subsets with current element
    subsets_with_current = []
    for subset in subsets_without_current:
        subsets_with_current.append([arr[index]] + subset)
    
    return subsets_without_current + subsets_with_current

# Usage
print(subsets([1, 2, 3]))
# Output: All possible subsets of [1, 2, 3]
```

## Performance Considerations

### Memoization for Optimization

Many recursive algorithms can be optimized using memoization:

```python
# Inefficient recursive Fibonacci
def fibonacci_slow(n):
    if n <= 1:
        return n
    return fibonacci_slow(n - 1) + fibonacci_slow(n - 2)

# Optimized with memoization
def fibonacci_memo(n, memo={}):
    if n in memo:
        return memo[n]
    
    if n <= 1:
        return n
    
    memo[n] = fibonacci_memo(n - 1, memo) + fibonacci_memo(n - 2, memo)
    return memo[n]

# Using Python's lru_cache decorator
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci_cached(n):
    if n <= 1:
        return n
    return fibonacci_cached(n - 1) + fibonacci_cached(n - 2)
```

## Algorithm Complexity Summary

| Algorithm | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Binary Search | O(log n) | O(log n) |
| Merge Sort | O(n log n) | O(n) |
| Quick Sort | O(n log n) avg | O(log n) |
| Tree Traversal | O(n) | O(h) where h is height |
| Fibonacci (naive) | O(2ⁿ) | O(n) |
| Fibonacci (memo) | O(n) | O(n) |
| Tower of Hanoi | O(2ⁿ) | O(n) |

## Related Topics

- **[Recursion Fundamentals](Recursion_Fundamentals.md)** - Basic recursion concepts
- **[Recursion vs Iteration](Recursion_vs_Iteration.md)** - When to choose each approach
- **[Common Recursive Patterns](Common_Recursive_Patterns.md)** - Problem-solving patterns
