# Sorting Algorithms Overview

Sorting algorithms arrange elements in a specific order (typically ascending or descending). They are fundamental to computer science and serve as building blocks for many other algorithms.

Your inputs are usually an array of $n$ numbers or keys `A`, and outputs are the sorted arrays `B`.

## Sorting Vocabulary
- Destructive: Overwrites the input array
- In place: Uses $O(1)$ extra space

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
| Permutation Sort | O(n!n) | O(n!n) | O(n!n) | O(n) | ❌ | ❌ |
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

### 0. Permutation Sort
```python
def permutation_sort(arr):
    '''Sort arr'''
    for B in permutations(arr):
        if is_sorted(B):
            return B
```
**Key techniques:**
1. enumerate all the permutations: Best case scenario = $\Omega(n!)$
2. check if the particular permutation is sorted: Average case = $O(n)$
```python
    for i in range(1, n-1):
        if B[i] <= B[i+1]:
            continue
        else:
            break
```

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

```python
def selection_sort(arr):
    ''' Selecting the biggest element'''
    n = len(arr)
    for i in range(n-1, 0, -1): # index of the max index should be from last element to 1
        max_idx = 0 # find the first element in the unsorted portion of the list
        for j in range(1, i+1):
            if arr[j] > arr[max_idx]:
                max_idx = j
        arr[i], arr[max_idx] = arr[max_idx], arr[i]
    return arr

# Good for: Minimizing memory writes, small datasets
# Bad for: Large datasets, when stability is needed
```
**Key techniques:**
1. find the biggest with index <= 1
2. swap the biggest element to end of array
3. sort from 1 to i - 1 for the unsorted part

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

**Quicksort** is a divide-and-conquer algorithm that works by selecting a 'pivot' element and partitioning the array around it. 

Notes for the Quick Sort algorithm when implementing it:
- In Quick Sort, after partitioning, the pivot is at its correct position. You should not include the pivot index in both recursive calls.

#### Recursive Implementation
```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot_position = len(arr) // 2
    pivot_value = arr[pivot_position]

    mid = [x for x in arr if x == pivot_value]
    left = [x for x in arr if x < pivot_value]
    right = [x for x in arr if x > pivot_value]

    return quick_sort(left) + mid + quick_sort(right)
```

#### Basic Implementation
```python
def quick_sort_desc(arr, left, right):
    if left < right:
        # find a index
        pivot_index = partition(arr, left, right)
        # first sort left half
        # the pivot index should not be included in both recursive calls.
        quick_sort_desc(arr, left, pivot_index - 1)
        quick_sort_desc(arr, pivot_index + 1, right)
    return arr
        
def partition(arr, left, right):
    i = left # starting the position of which value > pivot (in this case arr[right]) from left index 
    for j in range(left, right):
        if arr[j] >= arr[right]: # arr[right] is always the pivot
            arr[i], arr[j] = arr[j], arr[i]
            i = i + 1
    
    arr[i], arr[right] = arr[right], arr[i] # at the end of comparison, swapping pivot value with index indicating the position of which values are larger than pivot
    return i

# Good for: Average case performance, in-place sorting
# Bad for: Worst-case guarantees, already sorted data (without optimization)
```

#### Detailed Step-by-Step Example

Let's sort `[8, 3, 5, 4, 7, 6, 1, 2]` to understand how quicksort works:

```python
def quicksort_detailed(arr, low, high):
    """Quicksort with detailed steps."""
    if low < high:
        # Step 1: Partition and get pivot position
        pivot_pos = partition(arr, low, high)
        print(f"After partition around {arr[pivot_pos]}: {arr}")
        
        # Step 2: Recursively sort left and right subarrays
        quicksort_detailed(arr, low, pivot_pos - 1)   # Left side
        quicksort_detailed(arr, pivot_pos + 1, high)  # Right side

def partition(arr, low, high):
    """Partition array around pivot (last element)."""
    pivot = arr[high]  # Choose last element as pivot
    print(f"Partitioning around pivot: {pivot}")
    
    i = low - 1  # Index of smaller element
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
            print(f"  Swap {arr[j]} and {arr[i]}: {arr}")
    
    # Place pivot in correct position
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    print(f"  Place pivot {pivot} at position {i + 1}: {arr}")
    return i + 1
```

**Detailed Walkthrough:**

**Initial Array**: `[8, 3, 5, 4, 7, 6, 1, 2]`

**Step 1: First Partition (pivot = 2)**

```
Array:  [8, 3, 5, 4, 7, 6, 1, 2]
Indices: 0  1  2  3  4  5  6  7
Pivot = 2 (at index 7, last element)
store_index = 0 (where next small element goes)
```

**Step-by-step partitioning:**
- Check 8: Is 8 ≤ 2? NO → no swap
- Check 3: Is 3 ≤ 2? NO → no swap  
- Check 5: Is 5 ≤ 2? NO → no swap
- Check 4: Is 4 ≤ 2? NO → no swap
- Check 7: Is 7 ≤ 2? NO → no swap
- Check 6: Is 6 ≤ 2? NO → no swap
- Check 1: Is 1 ≤ 2? **YES!** → Swap positions 0 and 6

```
Before: [8, 3, 5, 4, 7, 6, 1, 2]
After:  [1, 3, 5, 4, 7, 6, 8, 2]
```

**Place pivot:** Swap pivot with position 1
```
Final:  [1, 2, 5, 4, 7, 6, 8, 3]
        ↑  ↑  ↑─────────────↑
      ≤2  pivot    >2
```

**Key Insight: Partitioning ≠ Sorting!**

The partition step **ONLY** guarantees:
- All elements ≤ pivot go to LEFT side
- All elements > pivot go to RIGHT side  
- **It does NOT sort within each side**

**Why elements appear "jumbled":** Only elements that needed to move (1 and 2) actually moved. Elements 5, 4, 7, 6, 8, 3 stayed in their relative positions until recursive sorting happens later.

**Step 2: Recursively sort subarrays**
- Left: `[1]` - already sorted
- Right: `[5, 4, 7, 6, 8, 3]` - needs recursive sorting

This process continues until all subarrays are sorted.



#### Quickselect Algorithm

**Quickselect** uses quicksort's partitioning but only recurses on one side to find the kth element in O(n) average time. A good tutorial of QuickSelect can be found at:
- https://www.youtube.com/watch?v=wiNfjkMDl3A. 
- https://www.youtube.com/watch?v=XEmy13g1Qxc.

A pivot is the fundmanetal component of this algorithm, which determines whether a number in the array goes to the first half or the second half, i.e. based on partition. 

Easiest solution:

```python
def findKthLargest(self, nums: List[int], k:int) -> int:
    nums.sort()
    return nums[len(nums) - k]
```

```python
import random

def find_kth_largest(numbers, k):
    if numbers:
        # find pivot
        pivot = partition(numbers, 0, len(numbers)-1)
        # compare k with pivot index
        if k - 1 == pivot: # the k - 1 rank value in the list is the kth largest number
            return numbers[pivot]
        # k smaller than pivot
        elif k - 1 < pivot: # meaning rank k-1 value (k largest) is smaller in rank than pivot, i.e. rank k-1 value should be bigger than numbers[pivot], we should be searching in the left half where values are bigger when the list is descending order
            return find_kth_largest(numbers[:pivot], k)
        # k - 1 > pivot, meaning k - 1 ranks lower in magnitude, i.e. value k - 1 is smaller in value than pivot, hence with descending order, it should be in second / right half
        else:
            return find_kth_largest(numbers[pivot+1:], k - 1 - pivot)
        
        
def partition(nums, l, r):
    # assign the store point for comparison for how many values are smaller than pivot
    i = l
    print("old nums:", nums)
    # choose pivot, i.e. could also be random int
    pivot_index = random.randint(l, r)
    pivot_value = nums[pivot_index]
    print("pivot value:", pivot_value)
    
    # always make sure the pivot is at right end
    nums[r], nums[pivot_index] = nums[pivot_index], nums[r]

    # loop and compare with pivot
    for j in range(l, r):
        if nums[j] >= pivot_value: # only when value is bigger than pivot we swap, i.e. the new list will have smaller on left of pivot and bigger on right
            print("nums[", j, "]:", nums[j], ", swapping with nums[", i, "]:", nums[i])
            nums[i], nums[j] = nums[j], nums[i]
            i = i + 1
    
    nums[i], nums[r] = nums[r], nums[i]
    print("new nums:", nums, r, "pivot:", i)
    # partition always return store point
    return i

# Average: O(n), Worst: O(n²), Space: O(1)
```

**Example**: Find 3rd largest in `[3, 2, 1, 5, 6, 4]`
- Convert to finding (6-3) = 3rd smallest (0-indexed: index 3)
- Partition around pivot 4: `[3, 2, 1, 4, 6, 5]`
- Pivot 4 is at index 3 = target index → Answer is 4!

**Why Quickselect is Better than Sorting:**
- **Sorting**: O(n log n) - processes all elements
- **Quickselect**: O(n) average - only processes one partition each recursion

#### Time Complexity Analysis
- **Best/Average Case**: O(n log n) - balanced partitions
- **Worst Case**: O(n²) - unbalanced partitions (e.g., already sorted with poor pivot)
- **Space**: O(log n) - recursion stack for balanced partitions
- **Quickselect Average**: O(n) - only recurses on one side

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

## Built-in Sort Function in Python
### sorting values
`sorted` function sorts a given list without modifying the original one. Instead, it returns a new list with the elements of the original list in sorted order.

### sorting tuples
The `sorted()` function can sort complex data structures like tuples using the `key` parameter. This parameter defines a function that takes an input element and 
returns a key that Python will use for sorting purposes. Note that `.sort()` function does the same but sorts the list in place.

```Python
def sort_tuples(tuples):
    return sorted(tuples, key=lambda x: x[1])
```

The lambda function `x: x[1]` takes an element from `tuples` and returns its second element (i.e., `x[1]`). The `sorted()` function uses these second elements to sort the tuples.
On top of that, if the second element can include ties we need to eliminate, a tuple comes to the rescue, as tuples in Python are automatically comparable:

```Python
def sort_tuples_ties(values):
    return values.sort(key=lambda x: (x[1], x[0]))
```

Similarly, we could also osrt a dictionary based on values:

```Python
def sort_dict(dictionary):
    return sorted(dictionart.items(), key=lambda x: x[1])
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
