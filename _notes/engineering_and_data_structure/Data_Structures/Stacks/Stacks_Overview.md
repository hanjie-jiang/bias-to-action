# Stacks Overview

A stack is a **Last In, First Out (LIFO)** data structure where elements are added and removed from the same end, called the "top" of the stack.

## Stack Operations

### Core Operations
Overall, the space complexity is $O(n)$, where $n$ is the number of elements in the stack.
- **Push**: Add element to top - O(1)
- **Pop**: Remove element from top - O(1) 
- **Peek/Top**: View top element without removing - O(1)
- **Empty**: Check if stack is empty - O(1)

### Visual Representation
```
    Push →  |   |  ← Pop
            | 3 |  (top)
            | 2 |
            | 1 |
            |___|  (bottom)
```

## Stack Implementations

### 1. Array-Based Stack
- **Pros**: Simple, good cache performance
- **Cons**: Fixed size (unless using dynamic array)

### 2. Linked List-Based Stack  
- **Pros**: Dynamic size, memory efficient
- **Cons**: Extra memory for pointers, worse cache performance

## Applications of Stacks

### 1. **Function Call Management**
- Function call stack
- Local variables and return addresses
- Recursion implementation

### 2. **Expression Evaluation**
- Infix to postfix conversion
- Balanced parentheses checking
- Calculator implementation

### 3. **Algorithms**
- Depth-First Search (DFS)
- Backtracking algorithms
- Undo operations

### 4. **Parsing and Compilers**
- Syntax analysis
- Bracket matching
- Compiler implementation

## Common Stack Patterns

### 1. **Monotonic Stack**
- Maintain elements in monotonic order
- Used for: Next Greater Element, Largest Rectangle

### 2. **Expression Parsing**
- Validate parentheses, brackets, braces
- Evaluate mathematical expressions

### 3. **Backtracking**
- Try solutions and undo when needed
- Path finding, puzzle solving

## Stack vs Other Data Structures

| Feature | Stack | Queue | Array |
|---------|-------|-------|-------|
| Access pattern | LIFO | FIFO | Random |
| Add/Remove | O(1) top only | O(1) both ends | O(1) end, O(n) middle |
| Use case | Undo, DFS | BFS, scheduling | General purpose |

## When to Use Stacks

### **Perfect for:**
- Undo/Redo functionality
- Parsing nested structures
- DFS traversals
- Managing function calls
- Backtracking algorithms

### **Not suitable for:**
- Random access to elements
- FIFO behavior needed
- Need to access middle elements

## Next Topics

- **[Stack Implementation](Stack_Implementation.md)** - Python implementation details
- **[Stack Problems](Stack_Problems.md)** - LeetCode problems and patterns
- **[Queues Overview](../Queues/Queues_Overview.md)** - FIFO counterpart to stacks
- **[Recursion Overview](../Recursion/recursion_overview.md)** - Natural stack-based process