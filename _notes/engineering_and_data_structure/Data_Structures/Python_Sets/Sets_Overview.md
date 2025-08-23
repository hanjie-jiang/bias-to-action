---
title: Python Sets Overview
---

# Python Sets Overview

A set in Python is an unordered collection of unique objects, ensuring the absence of duplicate values. Furthermore, it allows us to perform several operations on such collections, **such as intersection (identifying common elements), union (combining all unique elements), and difference (detecting unique items in a set).**

## Key Characteristics

- **Unordered**: Elements are not stored in any specific order
- **Unique**: No duplicate elements allowed
- **Mutable**: Can add or remove elements
- **Hashable**: Elements must be hashable (immutable)

## Basic Operations

### Creating Sets

```python
# Empty set
empty_set = set()

# Set from list
numbers = set([1, 2, 3, 4, 5])

# Set literal
fruits = {'apple', 'banana', 'orange'}

# Set from string (creates set of characters)
char_set = set('hello')  # {'h', 'e', 'l', 'o'}
```

### Adding Elements

```python
my_set = set()
my_set.add(1)           # Add single element
my_set.update([2, 3, 4]) # Add multiple elements
```

### Removing Elements

```python
my_set = {1, 2, 3, 4, 5}
my_set.remove(3)        # Raises KeyError if element doesn't exist
my_set.discard(6)       # No error if element doesn't exist
popped = my_set.pop()   # Remove and return arbitrary element
my_set.clear()          # Remove all elements
```

## Set Operations

### Mathematical Operations

```python
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

# Union
union = set1 | set2  # or set1.union(set2)
# Result: {1, 2, 3, 4, 5, 6}

# Intersection
intersection = set1 & set2  # or set1.intersection(set2)
# Result: {3, 4}

# Difference
difference = set1 - set2  # or set1.difference(set2)
# Result: {1, 2}

# Symmetric Difference
symmetric_diff = set1 ^ set2  # or set1.symmetric_difference(set2)
# Result: {1, 2, 5, 6}
```

### Membership Testing

```python
my_set = {1, 2, 3, 4, 5}
print(3 in my_set)      # True
print(6 not in my_set)  # True
```

## Common Use Cases

1. **Removing Duplicates**: Convert list to set and back
2. **Finding Unique Elements**: Set operations for uniqueness
3. **Membership Testing**: Fast O(1) lookups
4. **Mathematical Operations**: Union, intersection, difference

## Time Complexity

- **Add/Remove**: O(1) average case
- **Membership**: O(1) average case
- **Union/Intersection**: O(len(s1) + len(s2))
- **Iteration**: O(n)

## Related Topics

- **[Set Operations](Set_Operations.md)** - Detailed set operations and methods
- **[Array Intersection](../../Problem_Solving/Set_Dictionary_Problems/Array_Intersection.md)** - Using sets for array intersection
- **[Non-Repeating Elements](../../Problem_Solving/Set_Dictionary_Problems/Non_Repeating_Elements.md)** - Finding unique elements
- **[Unique Elements](../../Problem_Solving/Set_Dictionary_Problems/Unique_Elements.md)** - Finding elements unique to each array
