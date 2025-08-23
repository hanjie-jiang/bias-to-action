---
title: Common Patterns
---

# Common Patterns

Common patterns and techniques used in data structure and algorithm problems.

## Dictionary Operations

### How to Initiate a Set and Add Elements

```python
set_sample = set()
set_sample.add(1)
```

### How to Convert a String of Digits in Dictionary to a Number

```python
int("".join([value for _, value in dictionary.items()]))
```

### How to Sort a Dictionary in Python

#### Based on Key

```python
my_dict = {'apple': 3, 'orange': 1, 'banana': 2}  
sorted_by_key = dict(sorted(my_dict.items()))  
print(sorted_by_key)  
# Output: {'apple': 3, 'banana': 2, 'orange': 1}
```

#### Based on Value

```python
my_dict = {'apple': 3, 'orange': 1, 'banana': 2}  
sorted_by_value = dict(sorted(my_dict.items(), key=lambda item: item[1]))  
print(sorted_by_value)  
# Output: {'orange': 1, 'banana': 2, 'apple': 3}
```

## String Operations

### How to Ignore Capital vs Lower Cases in Python

```python
inventory1 = [string.upper() for string in inventory1] # .lower() for lower 
inventory2 = [string.upper() for string in inventory2] # .lower() for lower 
```

## Set Operations

### Basic Set Creation and Manipulation

```python
# Create empty set
empty_set = set()

# Create set from list
numbers = set([1, 2, 3, 4, 5])

# Add elements
my_set = set()
my_set.add(1)
my_set.update([2, 3, 4])

# Remove elements
my_set.remove(3)        # Raises KeyError if not found
my_set.discard(6)       # No error if not found
popped = my_set.pop()   # Remove and return arbitrary element
```

### Set Mathematical Operations

```python
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

# Union
union = set1 | set2  # or set1.union(set2)

# Intersection
intersection = set1 & set2  # or set1.intersection(set2)

# Difference
difference = set1 - set2  # or set1.difference(set2)

# Symmetric Difference
symmetric_diff = set1 ^ set2  # or set1.symmetric_difference(set2)
```

## List Operations

### List Comprehension Patterns

```python
# Basic list comprehension
squares = [x**2 for x in range(10)]

# List comprehension with condition
even_squares = [x**2 for x in range(10) if x % 2 == 0]

# Nested list comprehension
matrix = [[i+j for j in range(3)] for i in range(3)]
```

### List Manipulation

```python
# Remove duplicates while preserving order
def remove_duplicates_preserve_order(lst):
    seen = set()
    return [x for x in lst if not (x in seen or seen.add(x))]

# Flatten nested list
def flatten(lst):
    return [item for sublist in lst for item in sublist]
```

## Algorithmic Patterns

### Two Pointers Technique

```python
def two_pointers_example(arr):
    left, right = 0, len(arr) - 1
    while left < right:
        # Process elements at left and right pointers
        if some_condition:
            left += 1
        else:
            right -= 1
    return result
```

### Sliding Window Technique

```python
def sliding_window_example(arr, k):
    window_sum = sum(arr[:k])
    result = [window_sum]
    
    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i-k] + arr[i]
        result.append(window_sum)
    
    return result
```

### Prefix Sum Technique

```python
def prefix_sum_example(arr):
    prefix = [0] * (len(arr) + 1)
    for i in range(len(arr)):
        prefix[i + 1] = prefix[i] + arr[i]
    return prefix

def range_sum(prefix, left, right):
    return prefix[right + 1] - prefix[left]
```

## Time Complexity Cheat Sheet

### Common Operations

| Operation | List | Set | Dictionary |
|-----------|------|-----|------------|
| Access | O(1) | N/A | O(1) |
| Search | O(n) | O(1) | O(1) |
| Insert | O(1) | O(1) | O(1) |
| Delete | O(n) | O(1) | O(1) |

### Sorting Algorithms

| Algorithm | Time Complexity | Space Complexity | Stable |
|-----------|----------------|------------------|--------|
| Bubble Sort | O(n²) | O(1) | Yes |
| Selection Sort | O(n²) | O(1) | No |
| Insertion Sort | O(n²) | O(1) | Yes |
| Merge Sort | O(n log n) | O(n) | Yes |
| Quick Sort | O(n log n) | O(log n) | No |
| Heap Sort | O(n log n) | O(1) | No |

## Space Complexity Guidelines

- **O(1)**: Constant space - no extra space needed
- **O(n)**: Linear space - space grows with input size
- **O(n²)**: Quadratic space - space grows with square of input size
- **O(log n)**: Logarithmic space - space grows with log of input size

## Related Topics

- **[Time Complexity Guide](Time_Complexity_Guide.md)** - Detailed complexity analysis
- **[Interview Strategies](Interview_Strategies.md)** - Tips for technical interviews
- **[Set Operations](../Data_Structures/Hash_Tables/Python_Set_Operations.md)** - Detailed set operations
- **[Dictionary Operations](../Data_Structures/Hash_Tables/Python_Dictionary_Operations.md)** - Dictionary usage patterns
