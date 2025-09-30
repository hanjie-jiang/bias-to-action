# Arrays Overview

Arrays are one of the most fundamental data structures in computer science, providing a way to store multiple elements of the same type in a contiguous block of memory. This guide covers static arrays, their operations, and how they compare to linked lists.

## What is an Array?

An array is a collection of elements stored at contiguous memory locations. Each element can be accessed directly using its index.

## Static vs Dynamic Arrays

### Static Arrays
- **Fixed Size**: Size determined at creation time
- **Memory**: Allocated on stack or heap at compile/initialization time
- **Cannot Resize**: Size cannot be changed during runtime

### Dynamic Arrays (covered in detail in [Dynamic Arrays](Dynamic_Arrays.md))
- **Flexible Size**: Can grow and shrink during runtime
- **Automatic Management**: Handle memory allocation automatically
- **Examples**: Python lists, Java ArrayList, C++ vector

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

## Static Array Operations and Complexities

### Time Complexities for Static Arrays

| Operation | Time Complexity | Notes |
|-----------|----------------|-------|
| **Access by Index** | O(1) | Direct memory access |
| **Search** | O(n) | Must check each element |
| **Insert at Beginning** | O(n) | Shift all elements right |
| **Insert at End** | O(1) | If space available |
| **Insert at Middle** | O(n) | Shift elements right |
| **Delete from Beginning** | O(n) | Shift all elements left |
| **Delete from End** | O(1) | Just remove last element |
| **Delete from Middle** | O(n) | Shift elements left |

## Static Array Implementation Examples

### Python Simulation of Static Array

```python
class StaticArray:
    def __init__(self, capacity):
        self.capacity = capacity
        self.size = 0
        self.data = [None] * capacity
    
    def get(self, index):
        """Access element by index - O(1)"""
        if 0 <= index < self.size:
            return self.data[index]
        raise IndexError("Index out of bounds")
    
    def insert_at_beginning(self, value):
        """Insert at beginning - O(n)"""
        if self.size >= self.capacity:
            raise OverflowError("Array is full")
        
        # Shift all elements to the right
        for i in range(self.size, 0, -1):
            self.data[i] = self.data[i-1]
        
        self.data[0] = value
        self.size += 1
    
    def insert_at_end(self, value):
        """Insert at end - O(1)"""
        if self.size >= self.capacity:
            raise OverflowError("Array is full")
        
        self.data[self.size] = value
        self.size += 1
    
    def delete_from_beginning(self):
        """Delete from beginning - O(n)"""
        if self.size == 0:
            raise IndexError("Array is empty")
        
        deleted_value = self.data[0]
        # Shift all elements to the left
        for i in range(self.size - 1):
            self.data[i] = self.data[i + 1]
        
        self.size -= 1
        return deleted_value
    
    def delete_from_end(self):
        """Delete from end - O(1)"""
        if self.size == 0:
            raise IndexError("Array is empty")
        
        self.size -= 1
        deleted_value = self.data[self.size]
        self.data[self.size] = None
        return deleted_value

# Usage example
arr = StaticArray(5)
arr.insert_at_end(1)        # [1]
arr.insert_at_end(2)        # [1, 2]
arr.insert_at_beginning(0)  # [0, 1, 2]
```

### Basic Python List Operations (Dynamic Array)
```python
# Python lists are dynamic arrays
numbers = [1, 2, 3, 4, 5]

# Access element - O(1)
first_element = numbers[0]

# Search for element - O(n)
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

# Insert operations
numbers.append(6)      # O(1) amortized - at end
numbers.insert(0, 0)   # O(n) - at beginning
numbers.insert(2, 10)  # O(n) - at specific position

# Delete operations  
numbers.pop()          # O(1) - remove last
numbers.pop(0)         # O(n) - remove first
numbers.remove(10)     # O(n) - remove by value
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

## Arrays vs Linked Lists Comparison

### Linked Lists Overview
A **linked list** is a linear data structure where elements (nodes) are stored in sequence, but not in contiguous memory. Each node contains data and a reference (pointer) to the next node.

### Key Differences

| Aspect | Static Arrays | Linked Lists |
|--------|---------------|--------------|
| **Memory Layout** | Contiguous | Scattered |
| **Size** | Fixed | Dynamic |
| **Access by Index** | O(1) | O(n) |
| **Insert at Beginning** | O(n) | O(1) |
| **Insert at End** | O(1) | O(n)* |
| **Delete from Beginning** | O(n) | O(1) |
| **Delete from End** | O(1) | O(n) |
| **Memory Overhead** | None | Pointer per node |
| **Cache Performance** | Excellent | Poor |

*O(1) if tail pointer is maintained

### When to Use Each

**Choose Static Arrays when:**
- Fixed size is acceptable
- Frequent random access by index needed
- Memory efficiency is important
- Cache performance matters

**Choose Linked Lists when:**
- Dynamic size requirements
- Frequent insertions/deletions at beginning
- Unknown final size
- Memory allocation flexibility needed

### Visual Comparison

```
Static Array:
┌─────┬─────┬─────┬─────┬─────┐
│  1  │  2  │  3  │  4  │  5  │  ← Contiguous memory
└─────┴─────┴─────┴─────┴─────┘
  [0]   [1]   [2]   [3]   [4]   ← Direct index access

Linked List:
┌─────┬──┐    ┌─────┬──┐    ┌─────┬──┐
│  1  │ ●┼───►│  2  │ ●┼───►│  3  │ ●┼───► NULL
└─────┴──┘    └─────┴──┘    └─────┴──┘
   Node 1        Node 2        Node 3     ← Scattered memory
```

## Comparison with Other Data Structures

| Structure | Access | Search | Insert | Delete | Memory | Use Case |
|-----------|--------|--------|--------|--------|---------|----------|
| Static Array | O(1) | O(n) | O(n) | O(n) | Low | Fixed size, frequent access |
| Linked List | O(n) | O(n) | O(1)* | O(1)* | High | Dynamic size, frequent insert/delete |
| Dynamic Array | O(1) | O(n) | O(1)** | O(n) | Medium | Flexible size, frequent access |
| Hash Table | O(1) | O(1) | O(1) | O(1) | Medium | Key-based access |

*At beginning; **Amortized at end

## MIT 6.006 Connections

This comparison aligns with MIT 6.006's emphasis on:
- **Time complexity analysis** for different operations
- **Trade-offs** between data structures  
- **Memory vs time** considerations
- **Choosing appropriate data structures** for specific use cases
- **Understanding fundamental building blocks** for more complex algorithms

## Key Takeaways

1. **Static Arrays**: Excel at **random access** and **memory efficiency**
2. **Linked Lists**: Superior for **dynamic operations** and **flexible sizing**
3. **Operation Analysis**: 
   - Insertion/Deletion at beginning: Linked List wins (O(1) vs O(n))
   - Access by index: Static Array wins (O(1) vs O(n))
4. **Memory Considerations**: Static arrays have better cache locality and lower overhead
5. **Use Case Driven**: Choice depends on your application's access patterns

## Next Topics

- **[Dynamic Arrays](Dynamic_Arrays.md)** - Resizable arrays combining benefits of both
- **[Binary Search Fundamentals](../../Algorithms/Search_Algorithms/Binary_Search_Fundamentals.md)** - Efficient searching in sorted arrays
- **[Array Problems](Array_Problems.md)** - Common array-based coding problems
- **[Hash Tables](../Hash_Tables/Hash_Tables_Overview.md)** - Array-based key-value storage
