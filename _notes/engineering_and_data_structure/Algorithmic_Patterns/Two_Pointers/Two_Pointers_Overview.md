# Two Pointers Overview

The Two Pointers technique is a powerful algorithmic pattern that uses two pointers to traverse data structures efficiently. It's particularly useful for solving problems on arrays, strings, and linked lists.

## What is the Two Pointers Technique?

Two pointers is an algorithmic pattern where you use two pointers (indices) to traverse a data structure. The pointers can move:
- **In the same direction** (both left to right)
- **In opposite directions** (one from start, one from end)
- **At different speeds** (fast and slow pointers)

## Types of Two Pointers

### 1. Opposite Direction Pointers
```python
def two_sum_sorted(arr, target):
    """Find two numbers that add up to target in sorted array."""
    left, right = 0, len(arr) - 1
    
    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    
    return []
```

### 2. Same Direction Pointers
```python
def remove_duplicates(arr):
    """Remove duplicates from sorted array in-place."""
    if not arr:
        return 0
    
    slow = 0  # Position for next unique element
    
    for fast in range(1, len(arr)):
        if arr[fast] != arr[slow]:
            slow += 1
            arr[slow] = arr[fast]
    
    return slow + 1
```

### 3. Fast and Slow Pointers
```python
def find_middle(head):
    """Find middle node of linked list."""
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    return slow
```

## When to Use Two Pointers

### ✅ Good for:
- **Sorted arrays**: Finding pairs, triplets, or specific sums
- **Palindromes**: Checking if string/array is palindrome
- **Linked lists**: Finding cycles, middle elements
- **Sliding window problems**: When window size varies
- **In-place operations**: Removing duplicates, rearranging elements

### ❌ Not suitable for:
- **Unsorted data** (unless you sort first)
- **Hash table lookups** are more efficient
- **Complex data structures** without clear traversal order

## Common Problem Patterns

### 1. Target Sum Problems
```python
def two_sum_sorted(arr, target):
    """Classic two sum on sorted array."""
    left, right = 0, len(arr) - 1
    
    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return [arr[left], arr[right]]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    
    return []

def three_sum(arr, target=0):
    """Find all unique triplets that sum to target."""
    arr.sort()
    result = []
    
    for i in range(len(arr) - 2):
        # Skip duplicates for first element
        if i > 0 and arr[i] == arr[i-1]:
            continue
            
        left, right = i + 1, len(arr) - 1
        
        while left < right:
            current_sum = arr[i] + arr[left] + arr[right]
            
            if current_sum == target:
                result.append([arr[i], arr[left], arr[right]])
                
                # Skip duplicates
                while left < right and arr[left] == arr[left + 1]:
                    left += 1
                while left < right and arr[right] == arr[right - 1]:
                    right -= 1
                
                left += 1
                right -= 1
            elif current_sum < target:
                left += 1
            else:
                right -= 1
    
    return result
```

### 2. Palindrome Checking
```python
def is_palindrome(s):
    """Check if string is palindrome (ignoring case and non-alphanumeric)."""
    left, right = 0, len(s) - 1
    
    while left < right:
        # Skip non-alphanumeric characters
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        
        # Compare characters
        if s[left].lower() != s[right].lower():
            return False
        
        left += 1
        right -= 1
    
    return True

def valid_palindrome_with_deletion(s):
    """Check if string can be palindrome by deleting at most one character."""
    def is_palindrome_range(left, right):
        while left < right:
            if s[left] != s[right]:
                return False
            left += 1
            right -= 1
        return True
    
    left, right = 0, len(s) - 1
    
    while left < right:
        if s[left] != s[right]:
            # Try deleting either left or right character
            return (is_palindrome_range(left + 1, right) or 
                    is_palindrome_range(left, right - 1))
        left += 1
        right -= 1
    
    return True
```

### 3. Array Manipulation
```python
def remove_element(arr, val):
    """Remove all instances of val in-place."""
    slow = 0
    
    for fast in range(len(arr)):
        if arr[fast] != val:
            arr[slow] = arr[fast]
            slow += 1
    
    return slow

def move_zeros(arr):
    """Move all zeros to end while maintaining order of non-zeros."""
    slow = 0  # Position for next non-zero element
    
    # Move all non-zeros to front
    for fast in range(len(arr)):
        if arr[fast] != 0:
            arr[slow] = arr[fast]
            slow += 1
    
    # Fill remaining positions with zeros
    while slow < len(arr):
        arr[slow] = 0
        slow += 1

def reverse_array(arr):
    """Reverse array in-place."""
    left, right = 0, len(arr) - 1
    
    while left < right:
        arr[left], arr[right] = arr[right], arr[left]
        left += 1
        right -= 1
```

### 4. Linked List Problems
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def has_cycle(head):
    """Detect if linked list has cycle using Floyd's algorithm."""
    if not head or not head.next:
        return False
    
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
        if slow == fast:
            return True
    
    return False

def find_cycle_start(head):
    """Find the start of cycle in linked list."""
    if not head or not head.next:
        return None
    
    # Phase 1: Detect cycle
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    else:
        return None  # No cycle
    
    # Phase 2: Find cycle start
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    
    return slow

def remove_nth_from_end(head, n):
    """Remove nth node from end of linked list."""
    dummy = ListNode(0, head)
    slow = fast = dummy
    
    # Move fast pointer n+1 steps ahead
    for _ in range(n + 1):
        fast = fast.next
    
    # Move both pointers until fast reaches end
    while fast:
        slow = slow.next
        fast = fast.next
    
    # Remove the nth node from end
    slow.next = slow.next.next
    
    return dummy.next
```

## Time and Space Complexity

### Time Complexity
- **Most cases**: O(n) - single pass through data
- **Sorted array problems**: O(n) after O(n log n) sorting
- **Linked list problems**: O(n) - traverse list once

### Space Complexity
- **Usually**: O(1) - only use two pointer variables
- **In-place operations**: O(1) extra space
- **Result storage**: O(k) where k is size of result

## Advantages

1. **Efficient**: Often reduces time complexity from O(n²) to O(n)
2. **Space-efficient**: Usually O(1) extra space
3. **Intuitive**: Easy to understand and implement
4. **Versatile**: Works on arrays, strings, linked lists

## Common Mistakes

### 1. **Incorrect Boundary Conditions**
```python
# Wrong: May cause index out of bounds
while left <= right:  # Should be left < right for most cases

# Correct: Proper boundary check
while left < right:
```

### 2. **Not Handling Edge Cases**
```python
def two_sum_sorted(arr, target):
    # Always check for empty or single-element arrays
    if len(arr) < 2:
        return []
    
    left, right = 0, len(arr) - 1
    # ... rest of implementation
```

### 3. **Forgetting to Skip Duplicates**
```python
def three_sum_unique(arr):
    arr.sort()
    result = []
    
    for i in range(len(arr) - 2):
        # Important: Skip duplicate values for first element
        if i > 0 and arr[i] == arr[i-1]:
            continue
        # ... rest of implementation
```

## Practice Problems

### Easy
1. Two Sum (sorted array)
2. Valid Palindrome
3. Remove Duplicates from Sorted Array
4. Move Zeros

### Medium
1. 3Sum
2. Container With Most Water
3. Linked List Cycle II
4. Remove Nth Node From End

### Hard
1. Trapping Rain Water
2. 4Sum
3. Minimum Window Substring (with sliding window)

## Next Topics

- [[Opposite_Direction_Pointers]] - Deep dive into convergent pointers
- [[Same_Direction_Pointers]] - Fast and slow pointer patterns
- [[Sliding_Window_Overview]] - Related technique for subarray problems
- [[Two_Pointers_Problems]] - Practice problems and solutions
