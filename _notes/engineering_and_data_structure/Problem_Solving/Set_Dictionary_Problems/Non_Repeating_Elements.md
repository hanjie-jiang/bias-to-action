---
title: Non-Repeating Elements
---
# Non-Repeating Elements

Find elements that appear only once in an array using Python sets.

## Problem Description

Given an array of integers, find all elements that appear only once (non-repeating elements).

## Solution Using Sets

```python
def find_non_repeating_elements(nums):
    seen, repeated = set(), set()
    for num in nums:
        if num in seen:
            repeated.add(num)
        else: 
            seen.add(num)
    return list(seen - repeated)
```

## How It Works

1. **First Pass**: Iterate through the array
   - If we see a number for the first time, add it to `seen`
   - If we see a number again, add it to `repeated`
2. **Result**: Return elements that are in `seen` but not in `repeated`

## Time Complexity Analysis

This approach results in a time complexity of **O(n)** and a memory complexity of **O(n)** due to the constant time operations provided by the Python `set`.

## Space Complexity

- **Space Complexity**: O(n) - We need to store the seen and repeated sets

## Example Usage

```python
# Example 1
nums = [1, 2, 3, 1, 4, 2]
result = find_non_repeating_elements(nums)
print(result)  # [3, 4]

# Example 2
nums = [1, 1, 2, 2, 3, 3]
result = find_non_repeating_elements(nums)
print(result)  # []

# Example 3
nums = [1, 2, 3, 4, 5]
result = find_non_repeating_elements(nums)
print(result)  # [1, 2, 3, 4, 5]
```

## Alternative Approaches

### Using Dictionary for Counting

```python
from collections import Counter

def find_non_repeating_elements_counter(nums):
    counter = Counter(nums)
    return [num for num, count in counter.items() if count == 1]
```

**Time Complexity**: O(n)
**Space Complexity**: O(n)

### Using List Comprehension (Less Efficient)

```python
def find_non_repeating_elements_list(nums):
    return [num for num in nums if nums.count(num) == 1]
```

**Time Complexity**: O(n²) - `count()` method is O(n) for each element
**Space Complexity**: O(n)

## Key Insights

1. **Two-Set Approach**: Using two sets to track seen and repeated elements
2. **Set Difference**: Using `seen - repeated` to find elements that appear only once
3. **Single Pass**: The solution requires only one pass through the array
4. **Efficient Lookups**: Set operations provide O(1) average time complexity

## Edge Cases

- **Empty Array**: Returns empty list
- **All Repeating**: Returns empty list
- **All Unique**: Returns all elements
- **Single Element**: Returns the element if it appears only once

## Related Problems

- **[Array Intersection](Array_Intersection.md)** - Finding common elements between arrays
- **[Unique Elements](Unique_Elements.md)** - Finding elements unique to each array
- **[Unique Strings](../String_Problems/Unique_Strings.md)** - Finding unique strings in a list
