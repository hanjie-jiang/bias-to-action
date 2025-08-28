# Binary Search Fundamentals

Binary search is one of the most important and efficient search algorithms. It uses a divide-and-conquer approach to find elements in sorted arrays with O(log n) time complexity.

## How Binary Search Works

### Core Concept
Binary search works by repeatedly dividing the search space in half:
1. Compare the target with the middle element
2. If equal, we found the target
3. If target is smaller, search the left half
4. If target is larger, search the right half
5. Repeat until found or search space is empty

### Visual Example
```
Array: [1, 3, 5, 7, 9, 11, 13, 15]
Target: 7

Step 1: left=0, right=7, mid=3
[1, 3, 5, 7, 9, 11, 13, 15]
          ↑
arr[3] = 7 = target → Found at index 3!
```

## Implementation

### Iterative Approach (Recommended)
```python
def binary_search(arr, target):
    """
    Search for target in sorted array using binary search.
    
    Args:
        arr: Sorted array of comparable elements
        target: Element to search for
    
    Returns:
        Index of target if found, -1 otherwise
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        # Calculate middle index (avoid overflow)
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1  # Search right half
        else:
            right = mid - 1  # Search left half
	    if 0 <= low < len(data) and data[low] == target:
	        return low
	    else:
	        return -1
```

For classic binary search on a sorted list:
- If using `[low, high)` (right-exclusive), update `high = mid` and `low = mid + 1`
- If using `[low, high]` (both inclusive), update `high = mid - 1` and `low = mid + 1`
It's important to keep your interval logic and updates consistent.
### Recursive Approach
```python
def binary_search_recursive(arr, target, left=0, right=None):
    """
    Recursive implementation of binary search.
    """
    if right is None:
        right = len(arr) - 1
    
    # Base case: search space is empty
    if left > right:
        return -1
    
    mid = left + (right - left) // 2
    
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)
```

- If you use `[low, high)` (half-open), check when `high - low == 1` and compare `data[low]` to `target`.
- If you use `[left, right]` (closed), check when `left > right` and return `-1` (or `None`), and compare `arr[mid]` to `target` as usual.
## Time and Space Complexity

### Time Complexity: O(log n)
- Each comparison eliminates half of the remaining elements
- Maximum comparisons: log₂(n)
- Example: Array of 1000 elements needs at most 10 comparisons

### Space Complexity
- **Iterative**: O(1) - only uses a few variables
- **Recursive**: O(log n) - due to call stack

## Key Requirements

### 1. **Sorted Data**
Binary search only works on sorted arrays:
```python
# Correct: sorted array
sorted_arr = [1, 3, 5, 7, 9, 11]
result = binary_search(sorted_arr, 7)  # Works correctly

# Incorrect: unsorted array
unsorted_arr = [3, 1, 7, 5, 9, 11]
result = binary_search(unsorted_arr, 7)  # May give wrong result
```

### 2. **Comparable Elements**
Elements must be comparable with <, >, == operators:
```python
# Works with numbers
numbers = [1, 2, 3, 4, 5]

# Works with strings (lexicographical order)
words = ["apple", "banana", "cherry", "date"]

# Works with custom objects if comparison is defined
```

## Common Pitfalls and Solutions

### 1. **Integer Overflow**
```python
# Wrong: Can overflow in other languages
mid = (left + right) // 2

# Correct: Prevents overflow
mid = left + (right - left) // 2
```

### 2. **Infinite Loops**
```python
# Ensure loop terminates
while left <= right:  # Note: <= not <
    mid = left + (right - left) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        left = mid + 1   # Important: +1
    else:
        right = mid - 1  # Important: -1
```

### 3. **Off-by-One Errors**
```python
# Correct initialization
left, right = 0, len(arr) - 1  # right is last valid index

# Correct updates
left = mid + 1   # Exclude mid from next search
right = mid - 1  # Exclude mid from next search
```

## Variations and Extensions

### 1. **Find First Occurrence**
```python
def find_first_occurrence(arr, target):
    left, right = 0, len(arr) - 1
    result = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            result = mid
            right = mid - 1  # Continue searching left
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result
```

### 2. **Find Last Occurrence**
```python
def find_last_occurrence(arr, target):
    left, right = 0, len(arr) - 1
    result = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            result = mid
            left = mid + 1  # Continue searching right
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result
```

### 3. **Find Insertion Point**
```python
def find_insertion_point(arr, target):
    """Find index where target should be inserted to maintain sorted order."""
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return left  # Insertion point
```

## Practical Examples

### Example 1: Search in Library Catalog
```python
def search_book(catalog, isbn):
    """Search for book by ISBN in sorted catalog."""
    return binary_search([book.isbn for book in catalog], isbn)
```

### Example 2: Finding Square Root
```python
def sqrt_binary_search(x, precision=6):
    """Find square root using binary search."""
    if x < 0:
        return None
    
    left, right = 0, max(1, x)
    
    while right - left > 10**(-precision):
        mid = (left + right) / 2
        if mid * mid > x:
            right = mid
        else:
            left = mid
    
    return (left + right) / 2
```

## When to Use Binary Search

### ✅ Good for:
- Sorted arrays or lists
- Large datasets (> 100 elements)
- Frequent searches on static data
- Finding boundaries or ranges
- Mathematical problems (finding roots, etc.)

### ❌ Not suitable for:
- Unsorted data
- Linked lists (no random access)
- Very small datasets (< 10 elements)
- Data that changes frequently

## Performance Comparison

```python
import time
import random

# Generate test data
data = sorted(random.randint(1, 10000) for _ in range(10000))
target = random.choice(data)

# Linear search timing
start = time.time()
for _ in range(1000):
    linear_search(data, target)
linear_time = time.time() - start

# Binary search timing
start = time.time()
for _ in range(1000):
    binary_search(data, target)
binary_time = time.time() - start

print(f"Linear search: {linear_time:.4f}s")
print(f"Binary search: {binary_time:.4f}s")
print(f"Speedup: {linear_time/binary_time:.1f}x")
```

## Next Topics

- [[Binary_Search_Variations]] - Advanced patterns and applications
- [[Search_Problems]] - Practice problems using binary search
- [[Sorting_Algorithms_Overview]] - Preparing data for binary search
