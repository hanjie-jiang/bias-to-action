---
title: String Operations
---

# String Operations

Common string manipulation techniques and operations in Python.

## Basic String Operations

### String Creation and Access

```python
# String literals
s1 = "Hello, World!"
s2 = 'Python Programming'
s3 = """Multi-line
string"""

# String indexing
first_char = s1[0]      # 'H'
last_char = s1[-1]      # '!'
substring = s1[0:5]     # 'Hello'

# String length
length = len(s1)        # 13
```

### String Concatenation

```python
# Using + operator
result = "Hello" + " " + "World"  # "Hello World"

# Using join() method
words = ["Hello", "World", "Python"]
result = " ".join(words)  # "Hello World Python"

# Using f-strings (Python 3.6+)
name = "Alice"
age = 30
result = f"My name is {name} and I am {age} years old"
```

## String Methods

### Case Manipulation

```python
text = "Hello World"

# Convert case
upper_text = text.upper()      # "HELLO WORLD"
lower_text = text.lower()      # "hello world"
title_text = text.title()      # "Hello World"
capitalize_text = text.capitalize()  # "Hello world"
swapcase_text = text.swapcase()      # "hELLO wORLD"
```

### String Searching and Replacing

```python
text = "Hello World Hello"

# Find substring
index = text.find("World")     # 6
index = text.find("Python")    # -1 (not found)

# Count occurrences
count = text.count("Hello")    # 2

# Replace substring
new_text = text.replace("Hello", "Hi")  # "Hi World Hi"

# Check if string starts/ends with
starts_with = text.startswith("Hello")  # True
ends_with = text.endswith("World")      # False
```

### String Splitting and Joining

```python
text = "apple,banana,cherry,date"

# Split by delimiter
fruits = text.split(",")  # ['apple', 'banana', 'cherry', 'date']

# Split with max splits
parts = text.split(",", 2)  # ['apple', 'banana', 'cherry,date']

# Join strings
joined = "-".join(fruits)  # "apple-banana-cherry-date"

# Split by whitespace
sentence = "Hello   World   Python"
words = sentence.split()  # ['Hello', 'World', 'Python']
```

### String Stripping and Padding

```python
text = "   Hello World   "

# Remove whitespace
stripped = text.strip()        # "Hello World"
left_stripped = text.lstrip()  # "Hello World   "
right_stripped = text.rstrip() # "   Hello World"

# Padding
padded = text.center(20, "*")  # "***Hello World****"
left_padded = text.ljust(20, "*")  # "Hello World********"
right_padded = text.rjust(20, "*") # "********Hello World"
```

## Advanced String Operations

### String Formatting

```python
name = "Alice"
age = 30
height = 1.75

# Old-style formatting
result = "Name: %s, Age: %d, Height: %.2f" % (name, age, height)

# str.format() method
result = "Name: {}, Age: {}, Height: {:.2f}".format(name, age, height)
result = "Name: {n}, Age: {a}, Height: {h:.2f}".format(n=name, a=age, h=height)

# f-strings (recommended)
result = f"Name: {name}, Age: {age}, Height: {height:.2f}"
```

### String Validation

```python
text = "Hello123"

# Check character types
is_alpha = text.isalpha()      # False (contains digits)
is_digit = text.isdigit()      # False (contains letters)
is_alnum = text.isalnum()      # True (alphanumeric)
is_lower = text.islower()      # False (contains uppercase)
is_upper = text.isupper()      # False (contains lowercase)
is_space = text.isspace()      # False (no spaces)
```

### String Encoding and Decoding

```python
text = "Hello, 世界"

# Encode to bytes
utf8_bytes = text.encode('utf-8')
ascii_bytes = text.encode('ascii', errors='ignore')

# Decode from bytes
decoded_text = utf8_bytes.decode('utf-8')
```

## Common String Patterns

### Palindrome Check

```python
def is_palindrome(s):
    # Remove non-alphanumeric characters and convert to lowercase
    cleaned = ''.join(char.lower() for char in s if char.isalnum())
    return cleaned == cleaned[::-1]

# Example
print(is_palindrome("A man, a plan, a canal: Panama"))  # True
print(is_palindrome("race a car"))  # False
```

### Anagram Check

```python
def is_anagram(s1, s2):
    # Remove spaces and convert to lowercase
    s1_clean = ''.join(s1.lower().split())
    s2_clean = ''.join(s2.lower().split())
    
    # Sort and compare
    return sorted(s1_clean) == sorted(s2_clean)

# Example
print(is_anagram("listen", "silent"))  # True
print(is_anagram("hello", "world"))    # False
```

### String Reversal

```python
text = "Hello World"

# Using slice
reversed_text = text[::-1]  # "dlroW olleH"

# Using reversed() function
reversed_text = ''.join(reversed(text))  # "dlroW olleH"

# Word by word reversal
words = text.split()
reversed_words = ' '.join(word[::-1] for word in words)  # "olleH dlroW"
```

### Character Frequency

```python
from collections import Counter

def char_frequency(text):
    return Counter(text.lower())

# Example
freq = char_frequency("hello world")
print(freq)  # Counter({'l': 3, 'o': 2, 'h': 1, 'e': 1, ' ': 1, 'w': 1, 'r': 1, 'd': 1})
```

## Performance Considerations

### String Concatenation

```python
# Inefficient: creates new string each time
result = ""
for i in range(1000):
    result += str(i)

# Efficient: use list and join
parts = []
for i in range(1000):
    parts.append(str(i))
result = ''.join(parts)
```

### String vs List Operations

```python
# String operations create new objects
text = "Hello"
text += " World"  # Creates new string

# List operations modify in place
chars = list("Hello")
chars.append("!")  # Modifies existing list
```

## Best Practices

1. **Use f-strings for formatting**: More readable and efficient
2. **Use join() for concatenation**: More efficient than + operator in loops
3. **Use appropriate string methods**: Built-in methods are optimized
4. **Consider immutability**: Strings are immutable, operations create new objects
5. **Use raw strings for regex**: `r"pattern"` to avoid escaping backslashes
6. **Handle encoding properly**: Be explicit about encoding when working with files

## Related Topics

- **[Unique Strings](Unique_Strings.md)** - Finding unique strings in a list
- **[Array Intersection](../Set_Dictionary_Problems/Array_Intersection.md)** - String-based intersection problems
- **[Anagram Pairs](../Set_Dictionary_Problems/Anagram_Pairs.md)** - String anagram detection
- **[Common Patterns](../../Resources/Common_Patterns.md)** - General problem-solving patterns
