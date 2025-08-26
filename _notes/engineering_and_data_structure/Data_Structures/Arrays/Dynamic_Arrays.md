# Dynamic Arrays

Dynamic arrays are resizable arrays that can grow or shrink during runtime. Unlike static arrays, they automatically manage memory allocation and provide flexibility in size.

## What are Dynamic Arrays?

Dynamic arrays automatically resize themselves when elements are added or removed. They maintain the benefits of arrays (random access) while providing flexibility in size.

## How Dynamic Arrays Work

### 1. **Capacity vs Size**
- **Size**: Number of elements currently stored
- **Capacity**: Total memory allocated (usually larger than size)
- When size exceeds capacity, array is resized

### 2. **Resizing Strategy**
- Typically doubles the capacity when full
- May shrink when size becomes much smaller than capacity
- Amortizes the cost of resizing over many operations

## Python Lists (Dynamic Arrays)

```python
# Python lists are dynamic arrays
numbers = []  # Empty dynamic array

# Adding elements
numbers.append(1)      # O(1) amortized
numbers.append(2)      # O(1) amortized
numbers.extend([3, 4]) # O(k) where k is number of elements

# Access
print(numbers[0])      # O(1)

# Insertion at specific position
numbers.insert(1, 10)  # O(n)

# Deletion
numbers.pop()          # O(1) - remove last
numbers.pop(0)         # O(n) - remove first
numbers.remove(10)     # O(n) - remove by value

# Size information
print(len(numbers))    # Current size
```

## Time Complexities

| Operation | Average | Worst Case |
|-----------|---------|------------|
| Access    | O(1)    | O(1)       |
| Append    | O(1)    | O(n)*      |
| Insert    | O(n)    | O(n)       |
| Delete    | O(n)    | O(n)       |
| Search    | O(n)    | O(n)       |

*Worst case append is O(n) when resizing is needed

## Memory Management

### Resizing Process
```python
# Conceptual implementation of dynamic array resizing
class DynamicArray:
    def __init__(self):
        self.capacity = 1
        self.size = 0
        self.data = [None] * self.capacity
    
    def append(self, element):
        # Check if resize is needed
        if self.size >= self.capacity:
            self._resize()
        
        self.data[self.size] = element
        self.size += 1
    
    def _resize(self):
        # Double the capacity
        old_capacity = self.capacity
        self.capacity *= 2
        
        # Create new array and copy elements
        new_data = [None] * self.capacity
        for i in range(self.size):
            new_data[i] = self.data[i]
        
        self.data = new_data
```

## Advantages

1. **Flexible Size**: Can grow and shrink as needed
2. **Automatic Management**: Handles memory allocation automatically
3. **Amortized Performance**: Most operations are still fast on average
4. **Easy to Use**: Simple interface for adding/removing elements

## Disadvantages

1. **Memory Overhead**: May waste memory due to over-allocation
2. **Occasional Slow Operations**: Resizing can cause O(n) operations
3. **Memory Fragmentation**: Frequent resizing can fragment memory

## Common Dynamic Array Types

### Python
```python
# List (built-in dynamic array)
my_list = [1, 2, 3]
my_list.append(4)
```

### Java
```java
// ArrayList
ArrayList<Integer> list = new ArrayList<>();
list.add(1);
list.add(2);
```

### C++
```cpp
// std::vector
std::vector<int> vec;
vec.push_back(1);
vec.push_back(2);
```

## Best Practices

1. **Pre-allocate when size is known**:
   ```python
   # If you know approximate size
   numbers = [None] * 1000  # Pre-allocate
   ```

2. **Use appropriate initial capacity**:
   ```python
   # For large datasets, start with reasonable size
   large_list = []
   large_list.extend([0] * 10000)  # Better than 10000 appends
   ```

3. **Consider memory usage**:
   ```python
   # Shrink if memory is a concern
   if len(my_list) < len(my_list) // 4:
       # Consider shrinking or using different structure
   ```

## When to Use Dynamic Arrays

- **Unknown size**: When you don't know how many elements you'll need
- **Frequent additions**: When you frequently add elements to the end
- **General purpose**: Good default choice for most applications
- **Random access needed**: When you need O(1) access to elements

## Related Topics

- [[Arrays_Overview]] - Static arrays fundamentals
- [[Array_Problems]] - Common problems using arrays
- [[Sliding_Window_Overview]] - Pattern that works well with arrays
