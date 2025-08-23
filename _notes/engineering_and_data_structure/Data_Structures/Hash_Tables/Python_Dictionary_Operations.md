---
title: Dictionary Operations
---

# Dictionary Operations

Detailed dictionary operations and methods in Python.

## Basic Dictionary Methods

### Creating and Modifying

```python
# Create dictionary
my_dict = {}

# Add/Update items
my_dict['key'] = 'value'
my_dict.update({'key2': 'value2', 'key3': 'value3'})

# Get value with default
value = my_dict.get('key', 'default_value')

# Remove items
del my_dict['key']  # Raises KeyError if key doesn't exist
value = my_dict.pop('key', 'default')  # Returns default if key doesn't exist
my_dict.clear()  # Remove all items
```

### Accessing Dictionary Data

```python
person = {'name': 'John', 'age': 30, 'city': 'New York'}

# Get keys, values, and items
keys = person.keys()      # dict_keys(['name', 'age', 'city'])
values = person.values()  # dict_values(['John', 30, 'New York'])
items = person.items()    # dict_items([('name', 'John'), ('age', 30), ('city', 'New York')])

# Iterate through dictionary
for key in person:
    print(key, person[key])

for key, value in person.items():
    print(key, value)
```

## Advanced Dictionary Operations

### Dictionary Comprehension

```python
# Create dictionary from list
numbers = [1, 2, 3, 4, 5]
squares = {x: x**2 for x in numbers}  # {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

# Filter dictionary
person = {'name': 'John', 'age': 30, 'city': 'New York', 'phone': '123-456'}
string_values = {k: v for k, v in person.items() if isinstance(v, str)}

# Conditional comprehension
grades = {'Alice': 85, 'Bob': 92, 'Charlie': 78, 'David': 95}
passed = {name: grade for name, grade in grades.items() if grade >= 80}
```

### Merging Dictionaries

```python
dict1 = {'a': 1, 'b': 2}
dict2 = {'c': 3, 'd': 4}

# Using update() method (modifies dict1)
dict1.update(dict2)

# Using | operator (Python 3.9+)
merged = dict1 | dict2

# Using ** unpacking
merged = {**dict1, **dict2}

# Using dict() constructor
merged = dict(dict1, **dict2)
```

### Nested Dictionaries

```python
# Create nested dictionary
students = {
    'Alice': {'age': 20, 'grade': 'A', 'courses': ['Math', 'Physics']},
    'Bob': {'age': 22, 'grade': 'B', 'courses': ['Chemistry', 'Biology']}
}

# Access nested values
alice_age = students['Alice']['age']
alice_courses = students['Alice']['courses']

# Modify nested values
students['Alice']['grade'] = 'A+'
students['Bob']['courses'].append('Math')
```

## Dictionary Methods

### Default Values

```python
from collections import defaultdict

# Default dictionary with list
dd = defaultdict(list)
dd['fruits'].append('apple')
dd['fruits'].append('banana')
dd['vegetables'].append('carrot')

# Default dictionary with int (for counting)
word_count = defaultdict(int)
for word in ['apple', 'banana', 'apple', 'cherry']:
    word_count[word] += 1
```

### Counter

```python
from collections import Counter

# Count occurrences
words = ['apple', 'banana', 'apple', 'cherry', 'banana', 'apple']
word_counts = Counter(words)
print(word_counts)  # Counter({'apple': 3, 'banana': 2, 'cherry': 1})

# Most common elements
most_common = word_counts.most_common(2)  # [('apple', 3), ('banana', 2)]

# Arithmetic operations
counter1 = Counter(['a', 'b', 'c', 'a'])
counter2 = Counter(['a', 'b', 'd'])
combined = counter1 + counter2  # Counter({'a': 3, 'b': 2, 'c': 1, 'd': 1})
```

## Performance Considerations

### Time Complexity

| Operation | Average Case | Worst Case |
|-----------|-------------|------------|
| Access | O(1) | O(n) |
| Insert/Update | O(1) | O(n) |
| Delete | O(1) | O(n) |
| Search | O(1) | O(n) |
| Iteration | O(n) | O(n) |

### Memory Usage

- Dictionaries use more memory than lists due to hash table overhead
- Trade-off: memory for speed
- Use dictionaries when you need fast key-based lookups

## Common Patterns

### Grouping Data

```python
# Group students by grade
students = [
    {'name': 'Alice', 'grade': 'A'},
    {'name': 'Bob', 'grade': 'B'},
    {'name': 'Charlie', 'grade': 'A'},
    {'name': 'David', 'grade': 'C'}
]

grouped = {}
for student in students:
    grade = student['grade']
    if grade not in grouped:
        grouped[grade] = []
    grouped[grade].append(student['name'])

# Result: {'A': ['Alice', 'Charlie'], 'B': ['Bob'], 'C': ['David']}
```

### Caching/Memoization

```python
# Simple memoization decorator
def memoize(func):
    cache = {}
    
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    
    return wrapper

@memoize
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

### Configuration Management

```python
# Default configuration with overrides
default_config = {
    'host': 'localhost',
    'port': 8080,
    'debug': False,
    'timeout': 30
}

user_config = {
    'host': '192.168.1.100',
    'debug': True
}

# Merge configurations
config = default_config.copy()
config.update(user_config)
```

## Best Practices

1. **Use `.get()` for safe access**: Avoid KeyError exceptions
2. **Use dictionary comprehension**: More readable than loops
3. **Use `defaultdict` for counting/grouping**: Avoid checking if key exists
4. **Use `Counter` for frequency counting**: Built-in functionality
5. **Consider memory usage**: Dictionaries use more memory than lists
6. **Use appropriate data structures**: Lists for order, sets for uniqueness, dicts for key-value pairs

## Related Topics

- **[Dictionaries Overview](Python_Dictionaries.md)** - Basic dictionary concepts and characteristics
- **[Array Intersection](../../Problem_Solving/Set_Dictionary_Problems/Array_Intersection.md)** - Using dictionaries for counting
- **[Anagram Pairs](../../Problem_Solving/Set_Dictionary_Problems/Anagram_Pairs.md)** - Dictionary-based anagram detection
- **[Unique Strings](../../Problem_Solving/String_Problems/Unique_Strings.md)** - Dictionary-based string problems
