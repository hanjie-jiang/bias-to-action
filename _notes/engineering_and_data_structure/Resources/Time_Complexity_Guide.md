---
title: Time Complexity Guide
---

# Time Complexity Guide

Understanding algorithm efficiency and time complexity analysis.

## Big O Notation

Big O notation describes the performance or complexity of an algorithm by showing how the runtime grows as the input size increases.

### Common Time Complexities

| Complexity | Name | Description | Example |
|------------|------|-------------|---------|
| O(1) | Constant | Runtime doesn't change with input size | Array access, Hash table operations |
| O(log n) | Logarithmic | Runtime grows logarithmically | Binary search, Balanced tree operations |
| O(n) | Linear | Runtime grows linearly with input size | Linear search, Array traversal |
| O(n log n) | Linearithmic | Runtime grows as n times log n | Merge sort, Quick sort |
| O(n²) | Quadratic | Runtime grows as square of input size | Bubble sort, Nested loops |
| O(2ⁿ) | Exponential | Runtime grows exponentially | Recursive Fibonacci, Subset generation |
| O(n!) | Factorial | Runtime grows as factorial of input size | Traveling salesman (brute force) |

## Data Structure Complexities

### Array/List Operations

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Access | O(1) | O(1) |
| Search | O(n) | O(1) |
| Insert (end) | O(1) | O(1) |
| Insert (beginning) | O(n) | O(1) |
| Delete (end) | O(1) | O(1) |
| Delete (beginning) | O(n) | O(1) |

### Set Operations

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Access | N/A | N/A |
| Search | O(1) | O(1) |
| Insert | O(1) | O(1) |
| Delete | O(1) | O(1) |
| Union | O(n) | O(n) |
| Intersection | O(n) | O(n) |

### Dictionary Operations

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Access | O(1) | O(1) |
| Search | O(1) | O(1) |
| Insert | O(1) | O(1) |
| Delete | O(1) | O(1) |

## Algorithm Complexities

### Sorting Algorithms

| Algorithm | Time Complexity | Space Complexity | Stable |
|-----------|----------------|------------------|--------|
| Bubble Sort | O(n²) | O(1) | Yes |
| Selection Sort | O(n²) | O(1) | No |
| Insertion Sort | O(n²) | O(1) | Yes |
| Merge Sort | O(n log n) | O(n) | Yes |
| Quick Sort | O(n log n) | O(log n) | No |
| Heap Sort | O(n log n) | O(1) | No |

### Search Algorithms

| Algorithm | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Linear Search | O(n) | O(1) |
| Binary Search | O(log n) | O(1) |
| Depth-First Search | O(V + E) | O(V) |
| Breadth-First Search | O(V + E) | O(V) |

## Space Complexity Guidelines

- **O(1)**: Constant space - no extra space needed
- **O(n)**: Linear space - space grows with input size
- **O(n²)**: Quadratic space - space grows with square of input size
- **O(log n)**: Logarithmic space - space grows with log of input size

## Best Practices

1. **Choose appropriate data structures**: Use sets for membership testing, dictionaries for key-value pairs
2. **Consider trade-offs**: Time vs space complexity
3. **Profile your code**: Measure actual performance, not just theoretical complexity
4. **Optimize bottlenecks**: Focus on the most time-consuming parts of your algorithm
5. **Use built-in functions**: They're often optimized

## Related Topics

- **[Common Patterns](Common_Patterns.md)** - Frequently used patterns and their complexities
- **[Interview Strategies](Interview_Strategies.md)** - Tips for analyzing complexity in interviews
- **[Set Operations](../Data_Structures/Hash_Tables/Python_Set_Operations.md)** - Set operation complexities
- **[Dictionary Operations](../Data_Structures/Hash_Tables/Python_Dictionary_Operations.md)** - Dictionary operation complexities
