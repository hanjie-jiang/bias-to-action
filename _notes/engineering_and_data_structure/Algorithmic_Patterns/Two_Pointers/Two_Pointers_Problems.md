# Two Pointers Problems

Practice problems that utilize the two pointers technique. Problems are organized by pattern type and difficulty level.

## Opposite Direction Pointers Problems

### Easy Problems

#### 1. Two Sum II - Input Array Is Sorted (LeetCode 167)
```python
def two_sum(numbers, target):
    """Find two numbers that add up to target in sorted array."""
    left, right = 0, len(numbers) - 1
    
    while left < right:
        current_sum = numbers[left] + numbers[right]
        
        if current_sum == target:
            return [left + 1, right + 1]  # 1-indexed
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    
    return []

# Time: O(n), Space: O(1)
```

#### 2. Valid Palindrome (LeetCode 125)
```python
def is_palindrome(s):
    """Check if string is palindrome ignoring case and non-alphanumeric."""
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

# Time: O(n), Space: O(1)
```

#### 3. Reverse String (LeetCode 344)
```python
def reverse_string(s):
    """Reverse string in-place."""
    left, right = 0, len(s) - 1
    
    while left < right:
        s[left], s[right] = s[right], s[left]
        left += 1
        right -= 1

# Time: O(n), Space: O(1)
```

### Medium Problems

#### 1. 3Sum (LeetCode 15)
```python
def three_sum(nums):
    """Find all unique triplets that sum to zero."""
    nums.sort()
    result = []
    
    for i in range(len(nums) - 2):
        # Skip duplicates for first element
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        
        left, right = i + 1, len(nums) - 1
        
        while left < right:
            current_sum = nums[i] + nums[left] + nums[right]
            
            if current_sum == 0:
                result.append([nums[i], nums[left], nums[right]])
                
                # Skip duplicates
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                
                left += 1
                right -= 1
            elif current_sum < 0:
                left += 1
            else:
                right -= 1
    
    return result

# Time: O(n²), Space: O(1) excluding result
```

#### 2. Container With Most Water (LeetCode 11)
```python
def max_area(height):
    """Find two lines that form container with most water."""
    left, right = 0, len(height) - 1
    max_water = 0
    
    while left < right:
        # Calculate current area
        width = right - left
        current_area = min(height[left], height[right]) * width
        max_water = max(max_water, current_area)
        
        # Move pointer with smaller height
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    
    return max_water

# Time: O(n), Space: O(1)
```

#### 3. 3Sum Closest (LeetCode 16)
```python
def three_sum_closest(nums, target):
    """Find three numbers whose sum is closest to target."""
    nums.sort()
    n = len(nums)
    closest_sum = float('inf')
    
    for i in range(n - 2):
        left, right = i + 1, n - 1
        
        while left < right:
            current_sum = nums[i] + nums[left] + nums[right]
            
            if abs(current_sum - target) < abs(closest_sum - target):
                closest_sum = current_sum
            
            if current_sum == target:
                return target
            elif current_sum < target:
                left += 1
            else:
                right -= 1
    
    return closest_sum

# Time: O(n²), Space: O(1)
```

### Hard Problems

#### 1. Trapping Rain Water (LeetCode 42)
```python
def trap(height):
    """Calculate trapped rainwater using two pointers."""
    if not height:
        return 0
    
    left, right = 0, len(height) - 1
    left_max, right_max = 0, 0
    water_trapped = 0
    
    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                water_trapped += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water_trapped += right_max - height[right]
            right -= 1
    
    return water_trapped

# Time: O(n), Space: O(1)
```

## Same Direction Pointers Problems

### Easy Problems

#### 1. Remove Duplicates from Sorted Array (LeetCode 26)
```python
def remove_duplicates(nums):
    """Remove duplicates from sorted array in-place."""
    if not nums:
        return 0
    
    write_index = 1
    
    for read_index in range(1, len(nums)):
        if nums[read_index] != nums[read_index - 1]:
            nums[write_index] = nums[read_index]
            write_index += 1
    
    return write_index

# Time: O(n), Space: O(1)
```

#### 2. Remove Element (LeetCode 27)
```python
def remove_element(nums, val):
    """Remove all instances of val in-place."""
    write_index = 0
    
    for read_index in range(len(nums)):
        if nums[read_index] != val:
            nums[write_index] = nums[read_index]
            write_index += 1
    
    return write_index

# Time: O(n), Space: O(1)
```

#### 3. Move Zeroes (LeetCode 283)
```python
def move_zeroes(nums):
    """Move all zeros to end maintaining order of non-zeros."""
    write_index = 0
    
    # Move all non-zeros to front
    for read_index in range(len(nums)):
        if nums[read_index] != 0:
            nums[write_index] = nums[read_index]
            write_index += 1
    
    # Fill remaining with zeros
    while write_index < len(nums):
        nums[write_index] = 0
        write_index += 1

# Time: O(n), Space: O(1)
```

#### 4. Linked List Cycle (LeetCode 141)
```python
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

# Time: O(n), Space: O(1)
```

### Medium Problems

#### 1. Remove Nth Node From End of List (LeetCode 19)
```python
def remove_nth_from_end(head, n):
    """Remove nth node from end of linked list."""
    dummy = ListNode(0, head)
    slow = fast = dummy
    
    # Move fast pointer n+1 steps ahead
    for _ in range(n + 1):
        fast = fast.next
    
    # Move both until fast reaches end
    while fast:
        slow = slow.next
        fast = fast.next
    
    # Remove nth node from end
    slow.next = slow.next.next
    
    return dummy.next

# Time: O(n), Space: O(1)
```

#### 2. Linked List Cycle II (LeetCode 142)
```python
def detect_cycle(head):
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
        return None
    
    # Phase 2: Find cycle start
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    
    return slow

# Time: O(n), Space: O(1)
```

#### 3. Remove Duplicates from Sorted Array II (LeetCode 80)
```python
def remove_duplicates(nums):
    """Remove duplicates so each element appears at most twice."""
    if len(nums) <= 2:
        return len(nums)
    
    write_index = 2
    
    for read_index in range(2, len(nums)):
        if nums[read_index] != nums[write_index - 2]:
            nums[write_index] = nums[read_index]
            write_index += 1
    
    return write_index

# Time: O(n), Space: O(1)
```

## Sliding Window with Two Pointers

### Medium Problems

#### 1. Longest Substring Without Repeating Characters (LeetCode 3)
```python
def length_of_longest_substring(s):
    """Find length of longest substring without repeating characters."""
    char_set = set()
    left = 0
    max_length = 0
    
    for right in range(len(s)):
        # Shrink window until no duplicates
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        
        char_set.add(s[right])
        max_length = max(max_length, right - left + 1)
    
    return max_length

# Time: O(n), Space: O(min(m,n)) where m is charset size
```

#### 2. Max Consecutive Ones III (LeetCode 1004)
```python
def longest_ones(nums, k):
    """Longest subarray of 1s after flipping at most k zeros."""
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

#### 3. Minimum Size Subarray Sum (LeetCode 209)
```python
def min_subarray_len(target, nums):
    """Minimum length subarray with sum >= target."""
    left = 0
    min_length = float('inf')
    current_sum = 0
    
    for right in range(len(nums)):
        current_sum += nums[right]
        
        # Shrink window while sum >= target
        while current_sum >= target:
            min_length = min(min_length, right - left + 1)
            current_sum -= nums[left]
            left += 1
    
    return min_length if min_length != float('inf') else 0

# Time: O(n), Space: O(1)
```

## String Problems with Two Pointers

#### 1. Valid Palindrome II (LeetCode 680)
```python
def valid_palindrome(s):
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

# Time: O(n), Space: O(1)
```

#### 2. Is Subsequence (LeetCode 392)
```python
def is_subsequence(s, t):
    """Check if s is subsequence of t."""
    s_index = 0
    
    for t_index in range(len(t)):
        if s_index < len(s) and s[s_index] == t[t_index]:
            s_index += 1
    
    return s_index == len(s)

# Time: O(n), Space: O(1)
```

## Advanced Two Pointer Problems

#### 1. Sort Colors (LeetCode 75)
```python
def sort_colors(nums):
    """Sort array of 0s, 1s, and 2s in-place (Dutch Flag Algorithm)."""
    left, right = 0, len(nums) - 1
    current = 0
    
    while current <= right:
        if nums[current] == 0:
            nums[left], nums[current] = nums[current], nums[left]
            left += 1
            current += 1
        elif nums[current] == 1:
            current += 1
        else:  # nums[current] == 2
            nums[current], nums[right] = nums[right], nums[current]
            right -= 1
            # Don't increment current, need to check swapped element

# Time: O(n), Space: O(1)
```

#### 2. 4Sum (LeetCode 18)
```python
def four_sum(nums, target):
    """Find all unique quadruplets that sum to target."""
    nums.sort()
    n = len(nums)
    result = []
    
    for i in range(n - 3):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        
        for j in range(i + 1, n - 2):
            if j > i + 1 and nums[j] == nums[j - 1]:
                continue
            
            left, right = j + 1, n - 1
            
            while left < right:
                current_sum = nums[i] + nums[j] + nums[left] + nums[right]
                
                if current_sum == target:
                    result.append([nums[i], nums[j], nums[left], nums[right]])
                    
                    # Skip duplicates
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    
                    left += 1
                    right -= 1
                elif current_sum < target:
                    left += 1
                else:
                    right -= 1
    
    return result

# Time: O(n³), Space: O(1) excluding result
```

## Problem-Solving Strategies

### 1. **Choosing the Right Pattern**

```python
# Decision Tree for Two Pointers
def choose_pattern(problem_type):
    if problem_type == "target_sum_sorted":
        return "opposite_direction"
    elif problem_type == "palindrome":
        return "opposite_direction" 
    elif problem_type == "remove_elements":
        return "same_direction_read_write"
    elif problem_type == "cycle_detection":
        return "same_direction_fast_slow"
    elif problem_type == "sliding_window":
        return "same_direction_variable_gap"
    else:
        return "analyze_further"
```

### 2. **Common Templates by Problem Type**

#### Target Sum Template
```python
def target_sum_template(arr, target):
    arr.sort()  # Often needed
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

#### Array Modification Template
```python
def modify_array_template(arr, condition):
    write_index = 0
    
    for read_index in range(len(arr)):
        if condition(arr[read_index]):
            arr[write_index] = arr[read_index]
            write_index += 1
    
    return write_index
```

#### Cycle Detection Template
```python
def detect_cycle_template(head):
    if not head or not head.next:
        return False
    
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    
    return False
```

### 3. **Optimization Techniques**

#### Skip Duplicates
```python
# In sorted arrays
while left < right and nums[left] == nums[left + 1]:
    left += 1
while left < right and nums[right] == nums[right - 1]:
    right -= 1
```

#### Early Termination
```python
# For target sum problems
if nums[i] + nums[i + 1] + nums[i + 2] > target:
    break  # Remaining triplets will be too large
if nums[i] + nums[n - 2] + nums[n - 1] < target:
    continue  # Current triplet too small
```

## Practice Schedule

### Week 1: Basic Two Pointers
1. Two Sum II - Input Array Is Sorted
2. Valid Palindrome
3. Remove Duplicates from Sorted Array
4. Move Zeroes

### Week 2: Intermediate Problems
1. 3Sum
2. Container With Most Water
3. Linked List Cycle
4. Remove Nth Node From End

### Week 3: Advanced Applications
1. Trapping Rain Water
2. 4Sum
3. Minimum Window Substring
4. Sort Colors

## Common Pitfalls

### 1. **Boundary Conditions**
```python
# Always check for empty input
if not arr:
    return default_value

# Proper loop conditions
while left < right:  # Not left <= right for most cases
```

### 2. **Pointer Updates**
```python
# Ensure pointers always make progress
if condition:
    left += 1
else:
    right -= 1  # Both branches should move pointers
```

### 3. **Duplicate Handling**
```python
# Skip duplicates properly in sorted arrays
while left < right and arr[left] == arr[left + 1]:
    left += 1
# Similar for right pointer
```

## Next Topics

- [[Opposite_Direction_Pointers]] - Deep dive into convergent pointers
- [[Same_Direction_Pointers]] - Fast and slow pointer patterns
- [[Sliding_Window_Overview]] - Related technique for subarray problems
