# Linked Lists Overview

Linked lists are dynamic data structures where elements (nodes) are stored in sequence, with each node containing data and a reference (pointer) to the next node.

## Types of Linked Lists

### 1. Singly Linked List
- Each node points to the next node
- Last node points to `null`
- Can only traverse forward

### 2. Doubly Linked List  
- Each node has pointers to both next and previous nodes
- Can traverse in both directions
- Requires more memory per node

### 3. Circular Linked List
- Last node points back to the first node
- Forms a circle, no null pointers
- Can be singly or doubly circular

## Linked Lists vs Arrays

| Operation | Array | Linked List |
|-----------|-------|-------------|
| Access by index | O(1) | O(n) |
| Search | O(n) | O(n) |
| Insertion at beginning | O(n) | O(1) |
| Insertion at end | O(1)* | O(n) or O(1)** |
| Deletion at beginning | O(n) | O(1) |
| Memory usage | Contiguous | Scattered |
| Cache performance | Better | Worse |

*Amortized for dynamic arrays  
**O(1) if tail pointer is maintained

## When to Use Linked Lists

### ✅ **Good for:**
- Frequent insertions/deletions at the beginning
- Unknown or highly variable size
- Implementing other data structures (stacks, queues)
- When memory is allocated dynamically

### ❌ **Not ideal for:**
- Random access to elements
- Cache-sensitive applications
- Memory-constrained environments
- Small, fixed-size collections

## Key Concepts

### Node Structure
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
```

### Common Operations
- **Traversal**: Follow next pointers
- **Insertion**: Update pointer references
- **Deletion**: Skip nodes by updating pointers
- **Reversal**: Change direction of all pointers

## Next Topics

- **[Linked List Implementation](Linked_List_Implementation.md)** - Detailed implementation in Python
- **[Linked List Problems](Linked_List_Problems.md)** - LeetCode problems and patterns
- **[Arrays Overview](../Arrays/Arrays_Overview.md)** - Comparison with array data structures
- **[Stacks Overview](../Stacks/Stacks_Overview.md)** - Stack implementation using linked lists