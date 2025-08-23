---
title: Set Operations
---

# Set Operations

Detailed set operations and methods in Python.

## Basic Set Methods

### Adding Elements

```python
my_set = set()
my_set.add(1)           # Add single element
my_set.update([2, 3, 4]) # Add multiple elements from iterable
```

### Removing Elements

```python
my_set = {1, 2, 3, 4, 5}
my_set.remove(3)        # Raises KeyError if element doesn't exist
my_set.discard(6)       # No error if element doesn't exist
popped = my_set.pop()   # Remove and return arbitrary element
my_set.clear()          # Remove all elements
```

## Mathematical Operations

### Union

```python
set1 = {1, 2, 3}
set2 = {3, 4, 5}

# Using | operator
union = set1 | set2  # {1, 2, 3, 4, 5}

# Using union() method
union = set1.union(set2)  # {1, 2, 3, 4, 5}

# Union with multiple sets
union = set1.union(set2, {6, 7})  # {1, 2, 3, 4, 5, 6, 7}
```

### Intersection

```python
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

# Using & operator
intersection = set1 & set2  # {3, 4}

# Using intersection() method
intersection = set1.intersection(set2)  # {3, 4}
```

### Difference

```python
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

# Using - operator
difference = set1 - set2  # {1, 2}

# Using difference() method
difference = set1.difference(set2)  # {1, 2}
```

### Symmetric Difference

```python
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

# Using ^ operator
symmetric_diff = set1 ^ set2  # {1, 2, 5, 6}

# Using symmetric_difference() method
symmetric_diff = set1.symmetric_difference(set2)  # {1, 2, 5, 6}
```

## Set Comparison Methods

### Subset and Superset

```python
set1 = {1, 2, 3}
set2 = {1, 2, 3, 4, 5}

# Check if set1 is subset of set2
is_subset = set1.issubset(set2)  # True
is_subset = set1 <= set2         # True

# Check if set2 is superset of set1
is_superset = set2.issuperset(set1)  # True
is_superset = set2 >= set1           # True

# Proper subset/superset (excludes equality)
proper_subset = set1 < set2      # True
proper_superset = set2 > set1    # True
```

### Disjoint Sets

```python
set1 = {1, 2, 3}
set2 = {4, 5, 6}
set3 = {3, 4, 5}

# Check if sets have no common elements
is_disjoint = set1.isdisjoint(set2)  # True
is_disjoint = set1.isdisjoint(set3)  # False
```

## Set Comprehension

```python
# Create set from list with condition
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_numbers = {x for x in numbers if x % 2 == 0}  # {2, 4, 6, 8, 10}

# Create set from string (unique characters)
unique_chars = {char for char in "hello world"}  # {'h', 'e', 'l', 'o', ' ', 'w', 'r', 'd'}
```

## Performance Considerations

- **Membership testing**: O(1) average case
- **Add/Remove**: O(1) average case
- **Union/Intersection**: O(len(s1) + len(s2))
- **Iteration**: O(n)

## Related Topics

- **[Sets Overview](Sets_Overview.md)** - Basic set concepts and characteristics
- **[Array Intersection](../../Problem_Solving/Set_Dictionary_Problems/Array_Intersection.md)** - Using set intersection
- **[Non-Repeating Elements](../../Problem_Solving/Set_Dictionary_Problems/Non_Repeating_Elements.md)** - Using sets for uniqueness
- **[Unique Elements](../../Problem_Solving/Set_Dictionary_Problems/Unique_Elements.md)** - Using set difference operations
