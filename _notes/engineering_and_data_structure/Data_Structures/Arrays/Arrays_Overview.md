# Arrays Overview

Arrays are one of the most fundamental data structures in computer science, providing a way to store multiple elements of the same type in a contiguous block of memory.

## What is an Array?

An array is a collection of elements stored at contiguous memory locations. Each element can be accessed directly using its index.

## Key Characteristics

### 1. **Contiguous Memory**
- Elements are stored next to each other in memory
- Enables efficient cache usage
- Allows for pointer arithmetic

### 2. **Fixed Size (Static Arrays)**
- Size is determined at creation time
- Cannot be changed during runtime
- Memory is allocated on the stack or heap

### 3. **Random Access**
- O(1) time complexity for accessing any element
- Direct access using index: `array[index]`

## Time Complexities

| Operation | Time Complexity |
|-----------|----------------|
| Access    | O(1)          |
| Search    | O(n)          |
| Insertion | O(n)          |
| Deletion  | O(n)          |

## Python Implementation

```python
# Static-like array using list (Python lists are dynamic)
numbers = [1, 2, 3, 4, 5]

# Access element
first_element = numbers[0]  # O(1)

# Search for element
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

# Insert element (at end)
numbers.append(6)  # O(1) amortized

# Insert element (at specific position)
numbers.insert(2, 10)  # O(n)

# Delete element
numbers.remove(10)  # O(n)
```

## Advantages

1. **Fast Access**: O(1) random access to elements
2. **Memory Efficient**: No extra memory overhead per element
3. **Cache Friendly**: Contiguous memory improves cache performance
4. **Simple**: Easy to understand and implement

## Disadvantages

1. **Fixed Size**: Cannot resize (for static arrays)
2. **Expensive Insertion/Deletion**: O(n) for arbitrary positions
3. **Memory Waste**: May allocate more memory than needed

## When to Use Arrays

- **Known size**: When you know the number of elements in advance
- **Frequent access**: When you need to frequently access elements by index
- **Memory constraints**: When memory efficiency is important
- **Mathematical operations**: For matrix operations, numerical computations

## Array vs Other Data Structures

| Structure | Access | Search | Insert | Delete | Memory |
|-----------|--------|--------|--------|--------|---------|
| Array     | O(1)   | O(n)   | O(n)   | O(n)   | Low     |
| Linked List| O(n)   | O(n)   | O(1)   | O(1)   | High    |
| Hash Table| O(1)   | O(1)   | O(1)   | O(1)   | Medium  |

## Next Topics

- [[Binary_Search_Fundamentals]] - Efficient searching in sorted arrays
- [[Array_Problems]] - Common array-based coding problems
- [[Dynamic_Arrays]] - Resizable arrays (Python lists, Java ArrayList)
