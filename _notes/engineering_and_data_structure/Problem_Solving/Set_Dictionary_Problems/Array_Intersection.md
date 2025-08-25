---
title: Array Intersection
---
# Array Intersection

Find the intersection of two arrays using Python sets.

## Problem Description

Given two arrays, find all elements that appear in both arrays. The result should be sorted.

## Solution Using Sets

```python
def array_intersection(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = set1 & set2
    return sorted(list(intersection))
```

## Time Complexity Analysis

The set operations run at a time complexity of O(n), but the sorting step has a time complexity of O(n log n). Therefore, the overall time complexity of the solution is **O(n log n)**, dominated by the sorting step.

## Space Complexity

- **Space Complexity**: O(n) - We need to store the sets and the result list

## Example Usage

```python
# Example 1
list1 = [1, 2, 2, 1]
list2 = [2, 2]
result = array_intersection(list1, list2)
print(result)  # [2]

# Example 2
list1 = [4, 9, 5]
list2 = [9, 4, 9, 8, 4]
result = array_intersection(list1, list2)
print(result)  # [4, 9]
```

## Alternative Approaches

### Using List Comprehension (Less Efficient)

```python
def array_intersection_list(list1, list2):
    return sorted([x for x in list1 if x in list2])
```

**Time Complexity**: O(n²) - For each element in list1, we check if it's in list2
**Space Complexity**: O(n)

### Using Dictionary for Counting

```python
from collections import Counter

def array_intersection_counter(list1, list2):
    counter1 = Counter(list1)
    counter2 = Counter(list2)
    intersection = counter1 & counter2
    return sorted(list(intersection.elements()))
```

**Time Complexity**: O(n log n) - Due to sorting
**Space Complexity**: O(n)

## Key Insights

1. **Set Operations**: Using sets provides O(1) average time complexity for membership testing
2. **Sorting Requirement**: The final result needs to be sorted, which adds O(n log n) complexity
3. **Duplicate Handling**: Sets automatically handle duplicates, making the solution clean and efficient

## Related Problems

- **[Non-Repeating Elements](Non_Repeating_Elements.md)** - Finding elements that appear only once
- **[Unique Elements](Unique_Elements.md)** - Finding elements unique to each array
- **[Set Operations](../../Data_Structures/Hash_Tables/Python_Set_Operations.md)** - Understanding set operations in detail
