# Same Direction Pointers

Same direction pointers (also called fast and slow pointers, or tortoise and hare) move in the same direction but at different speeds or with different purposes. This technique is powerful for cycle detection, finding middle elements, and in-place array modifications.

## Core Concept

In same direction pointer problems:
1. **Both pointers start from the same end** (usually beginning)
2. **Pointers move at different speeds** or serve different purposes
3. **Fast pointer explores ahead** while slow pointer processes elements
4. **Gap between pointers** is maintained or adjusted based on problem requirements

## Basic Templates

### Template 1: Fast and Slow Pointers
```python
def fast_slow_template(arr):
    """Template for fast and slow pointer problems."""
    slow = fast = 0
    
    while fast < len(arr) and condition:
        # Move fast pointer
        fast += 1  # or more steps
        
        # Move slow pointer conditionally
        if some_condition:
            slow += 1
    
    return slow  # or process result
```

### Template 2: Read and Write Pointers
```python
def read_write_template(arr):
    """Template for in-place array modification."""
    write_index = 0
    
    for read_index in range(len(arr)):
        if should_keep(arr[read_index]):
            arr[write_index] = arr[read_index]
            write_index += 1
    
    return write_index  # New length
```

## Linked List Problems

### 1. **Cycle Detection (Floyd's Algorithm)**
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def has_cycle(head):
    """Detect if linked list has cycle."""
    if not head or not head.next:
        return False
    
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
        if slow == fast:
            return True
    
    return False

# Time: O(n), Space: O(1)
```

### 2. **Find Cycle Start**
```python
def detect_cycle_start(head):
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

# Time: O(n), Space: O(1)
```

### 3. **Find Middle Node**
```python
def find_middle(head):
    """Find middle node of linked list."""
    if not head:
        return None
    
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    return slow  # Middle node

# For even length, returns second middle node
# For odd length, returns exact middle
```

### 4. **Remove Nth Node from End**
```python
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

# Time: O(n), Space: O(1)
```

### 5. **Palindrome Linked List**
```python
def is_palindrome_list(head):
    """Check if linked list is palindrome."""
    if not head or not head.next:
        return True
    
    # Find middle using fast/slow pointers
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    # Reverse second half
    def reverse_list(node):
        prev = None
        while node:
            next_node = node.next
            node.next = prev
            prev = node
            node = next_node
        return prev
    
    second_half = reverse_list(slow)
    
    # Compare first and second half
    first_half = head
    while second_half:
        if first_half.val != second_half.val:
            return False
        first_half = first_half.next
        second_half = second_half.next
    
    return True

# Time: O(n), Space: O(1)
```

## Array Modification Problems

### 1. **Remove Duplicates**
```python
def remove_duplicates(nums):
    """Remove duplicates from sorted array in-place."""
    if not nums:
        return 0
    
    write_index = 1  # Position for next unique element
    
    for read_index in range(1, len(nums)):
        if nums[read_index] != nums[read_index - 1]:
            nums[write_index] = nums[read_index]
            write_index += 1
    
    return write_index

# Example: [1,1,2] → returns 2, array becomes [1,2,...]
```

### 2. **Remove Element**
```python
def remove_element(nums, val):
    """Remove all instances of val in-place."""
    write_index = 0
    
    for read_index in range(len(nums)):
        if nums[read_index] != val:
            nums[write_index] = nums[read_index]
            write_index += 1
    
    return write_index

# Example: nums = [3,2,2,3], val = 3 → returns 2
```

### 3. **Move Zeros**
```python
def move_zeros(nums):
    """Move all zeros to end while maintaining order of non-zeros."""
    write_index = 0  # Position for next non-zero element
    
    # Move all non-zeros to front
    for read_index in range(len(nums)):
        if nums[read_index] != 0:
            nums[write_index] = nums[read_index]
            write_index += 1
    
    # Fill remaining positions with zeros
    while write_index < len(nums):
        nums[write_index] = 0
        write_index += 1

# Example: [0,1,0,3,12] → [1,3,12,0,0]
```

### 4. **Remove Duplicates II (At Most 2)**
```python
def remove_duplicates_ii(nums):
    """Remove duplicates so each element appears at most twice."""
    if len(nums) <= 2:
        return len(nums)
    
    write_index = 2  # Position for next valid element
    
    for read_index in range(2, len(nums)):
        # Keep element if it's different from element 2 positions back
        if nums[read_index] != nums[write_index - 2]:
            nums[write_index] = nums[read_index]
            write_index += 1
    
    return write_index

# Example: [1,1,1,2,2,3] → returns 5, array becomes [1,1,2,2,3,...]
```

## Subarray Problems

### 1. **Longest Subarray with At Most K Zeros**
```python
def longest_subarray_k_zeros(nums, k):
    """Find longest subarray with at most k zeros."""
    left = 0
    zero_count = 0
    max_length = 0
    
    for right in range(len(nums)):
        if nums[right] == 0:
            zero_count += 1
        
        # Shrink window if too many zeros
        while zero_count > k:
            if nums[left] == 0:
                zero_count -= 1
            left += 1
        
        max_length = max(max_length, right - left + 1)
    
    return max_length

# Time: O(n), Space: O(1)
```

### 2. **Max Consecutive Ones**
```python
def find_max_consecutive_ones(nums):
    """Find maximum number of consecutive 1s."""
    max_count = 0
    current_count = 0
    
    for num in nums:
        if num == 1:
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 0
    
    return max_count

# Alternative using two pointers
def find_max_consecutive_ones_v2(nums):
    left = 0
    max_length = 0
    
    for right in range(len(nums)):
        if nums[right] == 0:
            left = right + 1  # Reset left pointer
        else:
            max_length = max(max_length, right - left + 1)
    
    return max_length
```

## String Problems

### 1. **Valid Subsequence**
```python
def is_subsequence(s, t):
    """Check if s is subsequence of t."""
    s_index = 0
    
    for t_index in range(len(t)):
        if s_index < len(s) and s[s_index] == t[t_index]:
            s_index += 1
    
    return s_index == len(s)

# Example: s = "ace", t = "abcde" → True
```

### 2. **Merge Strings Alternately**
```python
def merge_alternately(word1, word2):
    """Merge strings alternately."""
    result = []
    i = j = 0
    
    # Merge characters alternately
    while i < len(word1) and j < len(word2):
        result.append(word1[i])
        result.append(word2[j])
        i += 1
        j += 1
    
    # Append remaining characters
    while i < len(word1):
        result.append(word1[i])
        i += 1
    
    while j < len(word2):
        result.append(word2[j])
        j += 1
    
    return ''.join(result)

# Example: word1 = "abc", word2 = "pqr" → "apbqcr"
```

### 3. **Backspace String Compare**
```python
def backspace_compare(s, t):
    """Compare strings with backspace operations."""
    def get_next_valid_char(string, index):
        backspace_count = 0
        while index >= 0:
            if string[index] == '#':
                backspace_count += 1
            elif backspace_count > 0:
                backspace_count -= 1
            else:
                return index
            index -= 1
        return index
    
    s_index, t_index = len(s) - 1, len(t) - 1
    
    while s_index >= 0 or t_index >= 0:
        s_index = get_next_valid_char(s, s_index)
        t_index = get_next_valid_char(t, t_index)
        
        # Compare characters
        if s_index < 0 and t_index < 0:
            return True
        if s_index < 0 or t_index < 0:
            return False
        if s[s_index] != t[t_index]:
            return False
        
        s_index -= 1
        t_index -= 1
    
    return True

# Example: s = "ab#c", t = "ad#c" → True (both become "ac")
```

## Advanced Same Direction Techniques

### 1. **Sliding Window with Condition**
```python
def longest_substring_condition(s, condition_func):
    """Find longest substring satisfying condition."""
    left = 0
    max_length = 0
    current_state = {}
    
    for right in range(len(s)):
        # Update state with new character
        char = s[right]
        current_state[char] = current_state.get(char, 0) + 1
        
        # Shrink window while condition is violated
        while not condition_func(current_state):
            left_char = s[left]
            current_state[left_char] -= 1
            if current_state[left_char] == 0:
                del current_state[left_char]
            left += 1
        
        max_length = max(max_length, right - left + 1)
    
    return max_length
```

### 2. **K-Way Merge Using Pointers**
```python
def merge_k_sorted_arrays(arrays):
    """Merge k sorted arrays using pointers."""
    import heapq
    
    result = []
    heap = []
    
    # Initialize heap with first element from each array
    for i, array in enumerate(arrays):
        if array:
            heapq.heappush(heap, (array[0], i, 0))
    
    while heap:
        val, array_idx, element_idx = heapq.heappop(heap)
        result.append(val)
        
        # Add next element from same array
        if element_idx + 1 < len(arrays[array_idx]):
            next_val = arrays[array_idx][element_idx + 1]
            heapq.heappush(heap, (next_val, array_idx, element_idx + 1))
    
    return result

# Time: O(n log k), Space: O(k)
```

## Common Patterns

### 1. **Distance Maintenance**
```python
# Maintain fixed distance between pointers
def maintain_distance(arr, k):
    slow = 0
    for fast in range(k, len(arr)):
        # Process pair (slow, fast) with distance k
        process_pair(arr[slow], arr[fast])
        slow += 1
```

### 2. **Conditional Advancement**
```python
# Advance pointers based on conditions
def conditional_advance(arr):
    slow = 0
    for fast in range(len(arr)):
        if condition(arr[fast]):
            arr[slow] = arr[fast]
            slow += 1
    return slow
```

### 3. **State Tracking**
```python
# Track state between pointers
def track_state(arr):
    slow = 0
    state = initialize_state()
    
    for fast in range(len(arr)):
        update_state(state, arr[fast])
        
        while violates_condition(state):
            remove_from_state(state, arr[slow])
            slow += 1
        
        process_window(slow, fast)
```

## Common Mistakes

### 1. **Incorrect Pointer Initialization**
```python
# Wrong: Both pointers start at same position when distance needed
slow = fast = 0

# Correct: Initialize with proper distance
slow = 0
fast = k
```

### 2. **Not Handling Edge Cases**
```python
# Always check for empty input
if not arr:
    return appropriate_default

# Check bounds before accessing
if fast < len(arr):
    process(arr[fast])
```

### 3. **Infinite Loops in Linked Lists**
```python
# Ensure proper termination conditions
while fast and fast.next:  # Not just while fast
    slow = slow.next
    fast = fast.next.next
```

## Practice Problems

### Easy
1. Remove Duplicates from Sorted Array (LeetCode 26)
2. Remove Element (LeetCode 27)
3. Move Zeroes (LeetCode 283)
4. Linked List Cycle (LeetCode 141)
5. Is Subsequence (LeetCode 392)

### Medium
1. Remove Nth Node From End of List (LeetCode 19)
2. Linked List Cycle II (LeetCode 142)
3. Remove Duplicates from Sorted Array II (LeetCode 80)
4. Palindrome Linked List (LeetCode 234)
5. Max Consecutive Ones III (LeetCode 1004)

### Hard
1. Merge k Sorted Lists (LeetCode 23)
2. Trapping Rain Water (LeetCode 42)
3. Minimum Window Substring (LeetCode 76)

## Next Topics

- [[Opposite_Direction_Pointers]] - Learn about convergent pointer techniques
- [[Two_Pointers_Problems]] - Practice problems using two pointers
- [[Sliding_Window_Overview]] - Related technique for subarray problems
