# Sorting Algorithms Overview

Sorting algorithms arrange elements in a specific order (typically ascending or descending). They are fundamental to computer science and serve as building blocks for many other algorithms.

## Why Sorting Matters

### 1. **Enables Binary Search**
- Sorted data allows O(log n) search instead of O(n)
- Critical for performance in large datasets

### 2. **Data Organization**
- Makes data easier to understand and process
- Enables efficient algorithms for other problems

### 3. **Algorithm Foundation**
- Many algorithms assume sorted input
- Sorting is often a preprocessing step

## Classification of Sorting Algorithms

### By Stability
- **Stable**: Maintains relative order of equal elements
- **Unstable**: May change relative order of equal elements

### By Comparison
- **Comparison-based**: Compare elements to determine order
- **Non-comparison**: Use element properties (like digits)

### By Memory Usage
- **In-place**: Uses O(1) extra space
- **Out-of-place**: Uses O(n) or more extra space

## Common Sorting Algorithms

| Algorithm | Best Case | Average Case | Worst Case | Space | Stable | In-place |
|-----------|-----------|--------------|------------|-------|--------|----------|
| Bubble Sort | O(n) | O(n²) | O(n²) | O(1) | ✅ | ✅ |
| Selection Sort | O(n²) | O(n²) | O(n²) | O(1) | ❌ | ✅ |
| Insertion Sort | O(n) | O(n²) | O(n²) | O(1) | ✅ | ✅ |
| Merge Sort | O(n log n) | O(n log n) | O(n log n) | O(n) | ✅ | ❌ |
| Quick Sort | O(n log n) | O(n log n) | O(n²) | O(log n) | ❌ | ✅ |
| Heap Sort | O(n log n) | O(n log n) | O(n log n) | O(1) | ❌ | ✅ |
| Counting Sort | O(n+k) | O(n+k) | O(n+k) | O(k) | ✅ | ❌ |
| Radix Sort | O(d(n+k)) | O(d(n+k)) | O(d(n+k)) | O(n+k) | ✅ | ❌ |

*k = range of input, d = number of digits*

## Simple Sorting Algorithms (O(n²))

### 1. Bubble Sort
```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:  # Optimization: stop if no swaps
            break
    return arr

# Good for: Small datasets, educational purposes
# Bad for: Large datasets, performance-critical applications
```

### 2. Selection Sort
```python
def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

# Good for: Minimizing memory writes, small datasets
# Bad for: Large datasets, when stability is needed
```

### 3. Insertion Sort
```python
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

# Good for: Small datasets, nearly sorted data, online algorithms
# Bad for: Large datasets with random order
```

## Efficient Sorting Algorithms (O(n log n))

### 1. Merge Sort
```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    result = []
    i, j = 0, 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Good for: Guaranteed O(n log n), stable sorting, large datasets
# Bad for: Memory-constrained environments
```

### 2. Quick Sort
```python
def quick_sort(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        pivot_idx = partition(arr, low, high)
        quick_sort(arr, low, pivot_idx - 1)
        quick_sort(arr, pivot_idx + 1, high)
    
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

# Good for: Average case performance, in-place sorting
# Bad for: Worst-case guarantees, already sorted data (without optimization)
```

## Specialized Sorting Algorithms

### 1. Counting Sort (Non-comparison)
```python
def counting_sort(arr, max_val):
    # Only works for integers in known range
    count = [0] * (max_val + 1)
    
    # Count occurrences
    for num in arr:
        count[num] += 1
    
    # Reconstruct sorted array
    result = []
    for i, freq in enumerate(count):
        result.extend([i] * freq)
    
    return result

# Good for: Small range of integers, linear time needed
# Bad for: Large range, non-integer data
```

### 2. Radix Sort (Non-comparison)
```python
def radix_sort(arr):
    # Find maximum number to determine digits
    max_num = max(arr)
    exp = 1
    
    while max_num // exp > 0:
        counting_sort_by_digit(arr, exp)
        exp *= 10
    
    return arr

def counting_sort_by_digit(arr, exp):
    output = [0] * len(arr)
    count = [0] * 10
    
    # Count occurrences of each digit
    for num in arr:
        digit = (num // exp) % 10
        count[digit] += 1
    
    # Calculate positions
    for i in range(1, 10):
        count[i] += count[i - 1]
    
    # Build output array
    for i in range(len(arr) - 1, -1, -1):
        digit = (arr[i] // exp) % 10
        output[count[digit] - 1] = arr[i]
        count[digit] -= 1
    
    # Copy back to original array
    for i in range(len(arr)):
        arr[i] = output[i]

# Good for: Fixed-width integers, linear time needed
# Bad for: Variable-length data, small datasets
```

## Choosing the Right Sorting Algorithm

### For Small Arrays (< 50 elements)
- **Insertion Sort**: Simple, efficient for small data
- **Selection Sort**: Minimizes memory writes

### For Large Arrays
- **Merge Sort**: Guaranteed O(n log n), stable
- **Quick Sort**: Average O(n log n), in-place
- **Heap Sort**: Guaranteed O(n log n), in-place

### For Specific Data Types
- **Integers in small range**: Counting Sort
- **Integers with fixed digits**: Radix Sort
- **Strings**: Usually Quick Sort or Merge Sort

### For Specific Requirements
- **Stability needed**: Merge Sort, Insertion Sort
- **Memory constrained**: Heap Sort, Quick Sort
- **Nearly sorted data**: Insertion Sort
- **Online sorting**: Insertion Sort

## Hybrid Approaches

### Timsort (Python's Built-in)
```python
# Python's sorted() and list.sort() use Timsort
# Combines merge sort and insertion sort
# Optimized for real-world data patterns

arr = [3, 1, 4, 1, 5, 9, 2, 6]
sorted_arr = sorted(arr)  # Uses Timsort
arr.sort()  # In-place Timsort
```

### Introsort (C++ std::sort)
- Starts with Quick Sort
- Switches to Heap Sort if recursion depth exceeds limit
- Uses Insertion Sort for small subarrays

## Performance Testing

```python
import time
import random

def benchmark_sorting_algorithms():
    sizes = [100, 1000, 10000]
    algorithms = {
        'Bubble': bubble_sort,
        'Selection': selection_sort,
        'Insertion': insertion_sort,
        'Merge': merge_sort,
        'Quick': quick_sort,
        'Python Built-in': sorted
    }
    
    for size in sizes:
        print(f"\nArray size: {size}")
        data = [random.randint(1, 1000) for _ in range(size)]
        
        for name, func in algorithms.items():
            test_data = data.copy()
            start = time.time()
            func(test_data)
            end = time.time()
            print(f"{name}: {end - start:.4f}s")
```

## Next Topics

- [[Sorting_Problems]] - Practice problems using various sorting techniques
- [[Binary_Search_Fundamentals]] - Use sorting to enable binary search
- [[Two_Pointers_Overview]] - Techniques that work well with sorted data
