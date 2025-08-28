# Binary Search Variations

While basic binary search finds if an element exists, many real-world problems require variations of binary search. These patterns extend the core concept to solve more complex problems. For example, binary search algorithm can be applied onto continuous functions too. which will provide new insight on how to determine a specific function value within a continuous interval. This approach broadens the application of binary search from discrete space to continuous functions. The mechanism of binary search remains much the same, but instead of comparing the middle element to the target, we compare the middle point x's function value f(x) to the target.

## Common Binary Search Patterns

### 1. **Find First Occurrence**
Find the leftmost occurrence of a target in a sorted array with duplicates.

```python
def find_first_occurrence(arr, target):
    """Find the first (leftmost) occurrence of target."""
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

# Example: [1, 2, 2, 2, 3], target = 2
# Returns: 1 (first occurrence at index 1)
```

### 2. **Find Last Occurrence**
Find the rightmost occurrence of a target.

```python
def find_last_occurrence(arr, target):
    """Find the last (rightmost) occurrence of target."""
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

# Example: [1, 2, 2, 2, 3], target = 2
# Returns: 3 (last occurrence at index 3)
```

### 3. **Find Insert Position**
Find where to insert target to maintain sorted order.

```python
def search_insert_position(arr, target):
    """Find insertion point for target in sorted array."""
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return left  # Insertion position

# Example: [1, 3, 5, 6], target = 5 → returns 2
# Example: [1, 3, 5, 6], target = 2 → returns 1
# Example: [1, 3, 5, 6], target = 7 → returns 4
```

### 4. **Find Range**
Find the range \[first, last\] of target occurrences.

```python
def find_range(arr, target):
    """Find range of target occurrences."""
    def find_boundary(arr, target, find_left):
        left, right = 0, len(arr) - 1
        result = -1
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if arr[mid] == target:
                result = mid
                if find_left:
                    right = mid - 1  # Search left
                else:
                    left = mid + 1   # Search right
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return result
    
    first = find_boundary(arr, target, True)
    if first == -1:
        return [-1, -1]
    
    last = find_boundary(arr, target, False)
    return [first, last]

# Example: [5, 7, 7, 8, 8, 10], target = 8
# Returns: [3, 4]
```

### 5. **Find the answer of function**
```Python
# Define the function
def f(x):
    return x * x - 2

# Define the binary search function 
def binary_search(target, left, right, precision):
    while right - left > precision:
        mid = (left + right) / 2
        if f(mid) < target: # If the midpoint value is less than the target...
            left = mid  # ...update the left boundary to be the midpoint.
        else:
            right = mid  # Otherwise, update the right boundary.
    return left # Return the left boundary of our final, narrow interval.

epsilon = 1e-6
result = binary_search(0, 1, 2, epsilon)
print("x for which f(x) is approximately 0:", result)

# Outputs:
# x for which f(x) is approximately 0: 1.4142131805419922
```

Binary search only works reliably if the function is **monotonic** (always increasing or always decreasing) in the interval you search.
## Advanced Binary Search Patterns

### 1. **Peak Finding**
Find a peak element (greater than its neighbors).

```python
def find_peak_element(arr):
    """Find any peak element in the array."""
    left, right = 0, len(arr) - 1
    
    while left < right:
        mid = left + (right - left) // 2
        
        if arr[mid] > arr[mid + 1]:
            # Peak is in left half (including mid)
            right = mid
        else:
            # Peak is in right half
            left = mid + 1
    
    return left

# Example: [1, 2, 3, 1] → returns 2 (element 3 at index 2)
# Example: [1, 2, 1, 3, 5, 6, 4] → returns 1 or 5 (multiple peaks)
```

### 2. **Search in Rotated Sorted Array**
Search in a sorted array that has been rotated.

```python
def search_rotated_array(arr, target):
    """Search in rotated sorted array."""
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            return mid
        
        # Check which half is sorted
        if arr[left] <= arr[mid]:  # Left half is sorted
            if arr[left] <= target < arr[mid]:
                right = mid - 1  # Target in left half
            else:
                left = mid + 1   # Target in right half
        else:  # Right half is sorted
            if arr[mid] < target <= arr[right]:
                left = mid + 1   # Target in right half
            else:
                right = mid - 1  # Target in left half
    
    return -1

# Example: [4, 5, 6, 7, 0, 1, 2], target = 0 → returns 4
# Example: [4, 5, 6, 7, 0, 1, 2], target = 3 → returns -1
```

### 3. **Find Minimum in Rotated Array**
Find the minimum element in a rotated sorted array.

Naive Approach: A straightforward solution involves scanning each element in the array until we find a match or exhaust the list. This linear search approach is simple but computationally expensive for large lists - its time complexity is O(n).

```python
def find_min_rotated(arr):
    """Find minimum element in rotated sorted array."""
    left, right = 0, len(arr) - 1
    
    while left < right:
        mid = left + (right - left) // 2
        
        if arr[mid] > arr[right]:
            # Minimum is in right half
            left = mid + 1
        else:
            # Minimum is in left half (including mid)
            right = mid
    
    return arr[left]

# Example: [3, 4, 5, 1, 2] → returns 1
# Example: [4, 5, 6, 7, 0, 1, 2] → returns 0
```

## Binary Search on Answer Space

### 1. **Square Root**
Find integer square root using binary search.

```python
def sqrt_binary_search(x):
    """Find integer square root using binary search."""
    if x < 0:
        return -1
    if x < 2:
        return x
    
    left, right = 1, x // 2
    
    while left <= right:
        mid = left + (right - left) // 2
        square = mid * mid
        
        if square == x:
            return mid
        elif square < x:
            left = mid + 1
        else:
            right = mid - 1
    
    return right  # Largest integer whose square <= x

# Example: sqrt(8) → returns 2 (since 2² = 4 ≤ 8 < 3² = 9)
```

### 2. **Find Kth Smallest Element**
Find kth smallest element in sorted matrix.

```python
def kth_smallest_in_matrix(matrix, k):
    """Find kth smallest element in row and column sorted matrix."""
    n = len(matrix)
    left, right = matrix[0][0], matrix[n-1][n-1]
    
    def count_less_equal(target):
        """Count elements <= target."""
        count = 0
        row, col = n - 1, 0
        
        while row >= 0 and col < n:
            if matrix[row][col] <= target:
                count += row + 1
                col += 1
            else:
                row -= 1
        
        return count
    
    while left < right:
        mid = left + (right - left) // 2
        
        if count_less_equal(mid) < k:
            left = mid + 1
        else:
            right = mid
    
    return left
```

## Template for Binary Search Variations

### General Template
```python
def binary_search_template(arr, target):
    """General template for binary search variations."""
    left, right = 0, len(arr) - 1
    
    while left <= right:  # or left < right for some variations
        mid = left + (right - left) // 2
        
        if condition_met(arr[mid], target):
            return mid  # or update result and continue
        elif arr[mid] < target:  # or custom condition
            left = mid + 1
        else:
            right = mid - 1  # or right = mid
    
    return left  # or right, or -1, depending on problem
```

### Key Decisions
1. **Loop condition**: `left <= right` vs `left < right`
2. **Update strategy**: `left = mid + 1` vs `left = mid`
3. **Return value**: `left`, `right`, `mid`, or `-1`

## Common Pitfalls

### 1. **Infinite Loops**
```python
# Wrong: Can cause infinite loop
while left < right:
    mid = left + (right - left) // 2
    if condition:
        left = mid  # Should be mid + 1
    else:
        right = mid - 1

# Correct: Ensure progress
while left < right:
    mid = left + (right - left) // 2
    if condition:
        left = mid + 1
    else:
        right = mid
```

### 2. **Off-by-One Errors**
```python
# Be careful with boundary updates
if arr[mid] == target:
    result = mid
    right = mid - 1  # For first occurrence
    # vs
    left = mid + 1   # For last occurrence
```

### 3. **Integer Overflow**
```python
# Wrong: Can overflow in other languages
mid = (left + right) // 2

# Correct: Prevents overflow
mid = left + (right - left) // 2
```

## Practice Problems

### Easy
1. First Bad Version
2. Search Insert Position
3. Find First and Last Position

### Medium
1. Search in Rotated Sorted Array
2. Find Peak Element
3. Find Minimum in Rotated Sorted Array
4. Kth Smallest Element in Sorted Matrix

### Hard
1. Median of Two Sorted Arrays
2. Split Array Largest Sum
3. Capacity to Ship Packages

## Next Topics

- [[Search_Problems]] - Practice problems using binary search variations
- [[Sorting_Algorithms_Overview]] - Preparing data for binary search
