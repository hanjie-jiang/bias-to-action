# Queues Overview

A queue is a **First In, First Out (FIFO)** data structure where elements are added at one end (rear/back) and removed from the other end (front).

## Queue Operations

### Core Operations
- **Enqueue**: Add element to rear - O(1)
- **Dequeue**: Remove element from front - O(1)
- **Front/Peek**: View front element - O(1)
- **Empty**: Check if queue is empty - O(1)

### Visual Representation
```
Enqueue →  | 4 | 3 | 2 | 1 |  → Dequeue
          rear              front
          (back)            (head)
```

## Types of Queues

### 1. **Simple Queue (Linear Queue)**
- Basic FIFO structure
- Elements added at rear, removed from front

### 2. **Circular Queue**
- Rear connects back to front when space available
- Efficient space utilization
- Fixed size

### 3. **Priority Queue**
- Elements have priorities
- Higher priority elements dequeue first
- Implemented using heaps

### 4. **Deque (Double-ended Queue)**
- Add/remove from both ends
- Combines stack and queue functionality

## Queue Implementations

### 1. Array-Based Queue
- **Pros**: Simple, good cache performance
- **Cons**: Fixed size, wasted space in linear queues

### 2. Linked List-Based Queue
- **Pros**: Dynamic size, no wasted space
- **Cons**: Extra memory for pointers

### 3. Circular Array
- **Pros**: Efficient space usage, fixed overhead
- **Cons**: Complex pointer management

## Applications of Queues

### 1. **Breadth-First Search (BFS)**
- Level-order tree traversal
- Shortest path algorithms
- Graph exploration

### 2. **System Applications**
- Process scheduling in operating systems
- Print job queues
- IO buffer management

### 3. **Real-world Systems**
- Customer service lines
- Traffic management
- Task scheduling

### 4. **Algorithm Patterns**
- Sliding window problems
- Level-order processing
- Buffer management

## Common Queue Patterns

### 1. **BFS Pattern**
```python
def bfs(start):
    queue = [start]
    visited = set([start])
    
    while queue:
        node = queue.pop(0)  # dequeue
        process(node)
        
        for neighbor in get_neighbors(node):
            if neighbor not in visited:
                queue.append(neighbor)  # enqueue
                visited.add(neighbor)
```

### 2. **Sliding Window**
- Use deque for efficient window management
- Add/remove elements as window slides

### 3. **Level Processing**
- Process all elements at current level
- Add next level elements to queue

## Queue vs Other Data Structures

| Feature | Queue | Stack | Priority Queue |
|---------|-------|-------|----------------|
| Access pattern | FIFO | LIFO | Priority-based |
| Add | O(1) rear | O(1) top | O(log n) |
| Remove | O(1) front | O(1) top | O(log n) |
| Use case | BFS, scheduling | DFS, undo | Task priority |

## When to Use Queues

### ✅ **Perfect for:**
- BFS algorithms
- Process scheduling
- Buffer management
- Level-order processing
- Producer-consumer problems

### ❌ **Not suitable for:**
- Random access needed
- LIFO behavior required
- Need to access middle elements
- Priority-based processing (use priority queue)

## Queue Variants Summary

| Type | Add | Remove | Use Case |
|------|-----|--------|----------|
| Simple Queue | Rear O(1) | Front O(1) | Basic FIFO |
| Circular Queue | Rear O(1) | Front O(1) | Fixed-size FIFO |
| Priority Queue | O(log n) | O(log n) | Priority-based |
| Deque | Both ends O(1) | Both ends O(1) | Flexible access |

## Next Topics

- **[Queue Implementation](Queue_Implementation.md)** - Python implementation details
- **[Queue Problems](Queue_Problems.md)** - LeetCode problems and BFS patterns
- **[Stacks Overview](../Stacks/Stacks_Overview.md)** - LIFO counterpart to queues
- **[Binary Search Fundamentals](../../Algorithms/Search_Algorithms/Binary_Search_Fundamentals.md)** - BFS applications in trees