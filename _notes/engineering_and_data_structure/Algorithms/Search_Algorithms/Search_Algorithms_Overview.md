# Search Algorithms Overview

Search algorithms are fundamental techniques used to find specific elements within data structures. The choice of search algorithm depends on the data organization, size, and performance requirements.

## Types of Search Algorithms

### 1. **Linear Search**
- **Time Complexity**: O(n)
- **Space Complexity**: O(1)
- **Use Case**: Unsorted data, small datasets
- **How it works**: Check each element sequentially

### 2. **Binary Search**
- **Time Complexity**: O(log n)
- **Space Complexity**: O(1) iterative, O(log n) recursive
- **Use Case**: Sorted data
- **How it works**: Divide and conquer approach

### 3. **Hash-based Search**
- **Time Complexity**: O(1) average, O(n) worst case
- **Space Complexity**: O(n)
- **Use Case**: When fast lookups are needed
- **How it works**: Direct access using hash function

## Search Algorithm Comparison

| Algorithm | Best Case | Average Case | Worst Case | Space | Prerequisites |
|-----------|-----------|--------------|------------|-------|---------------|
| Linear    | O(1)      | O(n)         | O(n)       | O(1)  | None          |
| Binary    | O(1)      | O(log n)     | O(log n)   | O(1)  | Sorted data   |
| Hash      | O(1)      | O(1)         | O(n)       | O(n)  | Hash function |

## When to Use Each Algorithm

### Linear Search
```python
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

# Use when:
# - Data is unsorted
# - Small datasets (< 100 elements)
# - Simple implementation needed
# - Memory is very limited
```

### Binary Search
```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

# Use when:
# - Data is sorted
# - Large datasets
# - Logarithmic performance needed
# - Memory usage should be minimal
```

### Hash-based Search
```python
# Using Python dictionary (hash table)
def create_hash_table(arr):
    hash_table = {}
    for i, value in enumerate(arr):
        hash_table[value] = i
    return hash_table

def hash_search(hash_table, target):
    return hash_table.get(target, -1)

# Use when:
# - Very frequent searches
# - Constant time lookup needed
# - Extra memory is available
# - Data doesn't change often
```

## Search in Different Data Structures

### Arrays/Lists
- **Unsorted**: Linear search O(n)
- **Sorted**: Binary search O(log n)

### Linked Lists
- **Always**: Linear search O(n)
- **No random access**: Binary search not applicable

### Trees
- **Binary Search Tree**: O(log n) average, O(n) worst
- **Balanced Trees**: O(log n) guaranteed

### Hash Tables
- **Direct access**: O(1) average

## Practical Considerations

### 1. **Data Size**
- Small data (< 100): Linear search is often fine
- Medium data (100-10000): Consider sorting + binary search
- Large data (> 10000): Hash tables or advanced data structures

### 2. **Search Frequency**
- **One-time search**: Linear search
- **Frequent searches**: Invest in sorting or hash tables
- **Real-time requirements**: Hash tables

### 3. **Memory Constraints**
- **Limited memory**: Linear or binary search
- **Abundant memory**: Hash tables

### 4. **Data Mutability**
- **Static data**: Sort once, use binary search
- **Frequently changing**: Hash tables or linear search
- **Append-only**: Consider keeping sorted order

## Advanced Search Techniques

### 1. **Interpolation Search**
- Better than binary search for uniformly distributed data
- O(log log n) average case

### 2. **Exponential Search**
- Good for unbounded/infinite arrays
- O(log n) time complexity

### 3. **Ternary Search**
- Divides array into three parts
- Useful for finding maximum/minimum in unimodal functions

## Implementation Tips

### 1. **Handle Edge Cases**
```python
def robust_binary_search(arr, target):
    if not arr:  # Empty array
        return -1
    
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2  # Avoid overflow
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
```

### 2. **Consider Return Values**
- Return index vs boolean vs element
- Handle duplicates appropriately
- Define behavior for not found cases

## Next Topics

- [[Binary_Search_Fundamentals]] - Deep dive into binary search
- [[Binary_Search_Variations]] - Advanced binary search patterns
- [[Search_Problems]] - Practice problems and applications
