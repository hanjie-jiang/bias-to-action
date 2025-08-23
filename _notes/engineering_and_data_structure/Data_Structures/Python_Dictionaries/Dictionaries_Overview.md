---
title: Python Dictionaries Overview
---

# Python Dictionaries Overview

Python dictionaries are mutable, unordered collections of key-value pairs. They provide fast O(1) average time complexity for lookups, insertions, and deletions.

## Key Characteristics

- **Key-Value Pairs**: Each element is a key-value pair
- **Unordered**: Elements are not stored in any specific order (Python 3.7+ maintains insertion order)
- **Mutable**: Can add, modify, or remove key-value pairs
- **Hashable Keys**: Keys must be hashable (immutable)
- **Unique Keys**: No duplicate keys allowed

## Basic Operations

### Creating Dictionaries

```python
# Empty dictionary
empty_dict = {}

# Dictionary with initial values
person = {'name': 'John', 'age': 30, 'city': 'New York'}

# Using dict() constructor
numbers = dict(one=1, two=2, three=3)

# From list of tuples
pairs = [('a', 1), ('b', 2), ('c', 3)]
dict_from_pairs = dict(pairs)
```

### Accessing and Modifying

```python
person = {'name': 'John', 'age': 30}

# Accessing values
name = person['name']  # Raises KeyError if key doesn't exist
age = person.get('age', 0)  # Returns default value if key doesn't exist

# Adding/Modifying
person['city'] = 'New York'  # Add new key-value pair
person['age'] = 31  # Modify existing value

# Removing
del person['age']  # Raises KeyError if key doesn't exist
city = person.pop('city', 'Unknown')  # Returns default if key doesn't exist
person.clear()  # Remove all items
```

## Common Methods

```python
person = {'name': 'John', 'age': 30, 'city': 'New York'}

# Keys, values, and items
keys = person.keys()  # dict_keys(['name', 'age', 'city'])
values = person.values()  # dict_values(['John', 30, 'New York'])
items = person.items()  # dict_items([('name', 'John'), ('age', 30), ('city', 'New York')])

# Membership testing
'name' in person  # True
'phone' not in person  # True

# Length
len(person)  # 3
```

## Dictionary Operations

### Merging Dictionaries

```python
dict1 = {'a': 1, 'b': 2}
dict2 = {'c': 3, 'd': 4}

# Using update() method
dict1.update(dict2)  # Modifies dict1 in-place

# Using | operator (Python 3.9+)
merged = dict1 | dict2  # Creates new dictionary

# Using ** unpacking
merged = {**dict1, **dict2}
```

### Dictionary Comprehension

```python
# Create dictionary from list
numbers = [1, 2, 3, 4, 5]
squares = {x: x**2 for x in numbers}

# Filter dictionary
person = {'name': 'John', 'age': 30, 'city': 'New York', 'phone': '123-456'}
filtered = {k: v for k, v in person.items() if isinstance(v, str)}
```

## Time Complexity

- **Access**: O(1) average case
- **Insert/Update**: O(1) average case
- **Delete**: O(1) average case
- **Search**: O(1) average case
- **Iteration**: O(n)

## Common Use Cases

1. **Caching/Memoization**: Store computed results
2. **Counting**: Track frequency of elements
3. **Grouping**: Organize data by categories
4. **Configuration**: Store settings and parameters
5. **JSON-like Data**: Represent structured data

## Related Topics

- **[Dictionary Operations](Dictionary_Operations.md)** - Detailed dictionary methods and techniques
- **[Array Intersection](../../Problem_Solving/Set_Dictionary_Problems/Array_Intersection.md)** - Using dictionaries for counting
- **[Anagram Pairs](../../Problem_Solving/Set_Dictionary_Problems/Anagram_Pairs.md)** - Dictionary-based anagram detection
- **[Unique Strings](../../Problem_Solving/String_Problems/Unique_Strings.md)** - Dictionary-based string problems
