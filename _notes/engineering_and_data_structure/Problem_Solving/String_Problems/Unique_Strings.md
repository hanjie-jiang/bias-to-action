---
title: Unique Strings
---

# Unique Strings

Find the first unique string in a list using Python sets and dictionaries.

## Problem Description

Given a list of strings, find the first string that appears only once (is unique).

## Solution Using Two Sets

```python
def find_unique_string(words):
    seen = set()
    duplicates = set()
    for word in words:
        if word in seen:
            duplicates.add(word)
        seen.add(word)
    for word in words:
        if word not in duplicates:
            return word
    return ""
```

## How It Works

1. **First Pass**: Track seen words and duplicates
   - If we see a word for the first time, add it to `seen`
   - If we see a word again, add it to `duplicates`
2. **Second Pass**: Find the first word not in duplicates
   - Return the first word that appears only once

## Solution Using Dictionary

```python
def find_unique_string_dict(words):
    count_dict = {}
    for word in words:
        if word in count_dict:
            count_dict[word] = count_dict[word] + 1
        else:
            count_dict[word] = 1
    for word in words:
        if count_dict[word] == 1:
            return word
    return ""
```

## Time Complexity Analysis

- **Two Sets Approach**: O(n) - Two passes through the array
- **Dictionary Approach**: O(n) - Two passes through the array
- **Space Complexity**: O(n) - We need to store the sets/dictionary

## Example Usage

```python
# Example 1
words = ["hello", "world", "hello", "python"]
result = find_unique_string(words)
print(result)  # "world"

# Example 2
words = ["a", "b", "a", "b", "c"]
result = find_unique_string(words)
print(result)  # "c"

# Example 3
words = ["a", "a", "b", "b"]
result = find_unique_string(words)
print(result)  # ""

# Example 4
words = ["unique"]
result = find_unique_string(words)
print(result)  # "unique"
```

## Comparison of Approaches

### Two Sets Approach
- **Pros**: Simple logic, easy to understand
- **Cons**: Requires two passes through the array
- **Best for**: When you need to track both seen and duplicate elements

### Dictionary Approach
- **Pros**: More explicit counting, can be extended for other counting problems
- **Cons**: Slightly more complex logic
- **Best for**: When you need exact counts or might need to extend the solution

## Key Insights

1. **Two-Pass Solution**: First pass to identify duplicates, second pass to find first unique
2. **Order Preservation**: The second pass maintains the original order to find the "first" unique
3. **Efficient Lookups**: Set/dictionary operations provide O(1) average time complexity
4. **Edge Case Handling**: Returns empty string if no unique element exists

## Edge Cases

- **Empty List**: Returns empty string
- **All Duplicates**: Returns empty string
- **All Unique**: Returns the first element
- **Single Element**: Returns the element
- **Case Sensitivity**: "Hello" and "hello" are considered different

## Related Problems

- **[Non-Repeating Elements](../Set_Dictionary_Problems/Non_Repeating_Elements.md)** - Finding non-repeating numbers
- **[Array Intersection](../Set_Dictionary_Problems/Array_Intersection.md)** - Finding common elements
- **[String Operations](String_Operations.md)** - Other string manipulation techniques
