# Variable Size Window

Variable size sliding window dynamically adjusts the window size based on certain conditions. This technique is powerful for optimization problems where you need to find the optimal subarray or substring.

## Core Concept

In variable size window problems:
1. **Window size changes** based on conditions
2. **Expand window** by moving right pointer
3. **Shrink window** by moving left pointer
4. **Track optimal solution** during the process

## Basic Template

```python
def variable_window_template(arr, condition):
    """Template for variable size window problems."""
    left = 0
    best_result = 0  # or float('inf') for minimum problems
    current_state = initialize_state()
    
    for right in range(len(arr)):
        # Expand window: add arr[right]
        update_state_add(current_state, arr[right])
        
        # Shrink window while condition is violated
        while violates_condition(current_state, condition):
            update_state_remove(current_state, arr[left])
            left += 1
        
        # Update best result
        best_result = update_best(best_result, right - left + 1)
    
    return best_result
```

## Maximum Window Problems

### 1. **Longest Substring Without Repeating Characters**
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
        
        # Add current character and update max
        char_set.add(s[right])
        max_length = max(max_length, right - left + 1)
    
    return max_length

# Example: s = "abcabcbb"
# Result: 3 (substring "abc")
```

### 2. **Longest Substring with At Most K Distinct Characters**
```python
def length_of_longest_substring_k_distinct(s, k):
    """Longest substring with at most k distinct characters."""
    if k == 0:
        return 0
    
    char_count = {}
    left = 0
    max_length = 0
    
    for right in range(len(s)):
        # Add current character
        char_count[s[right]] = char_count.get(s[right], 0) + 1
        
        # Shrink window if too many distinct characters
        while len(char_count) > k:
            char_count[s[left]] -= 1
            if char_count[s[left]] == 0:
                del char_count[s[left]]
            left += 1
        
        max_length = max(max_length, right - left + 1)
    
    return max_length

# Example: s = "eceba", k = 2
# Result: 3 (substring "ece")
```

### 3. **Max Consecutive Ones III**
```python
def longest_ones(nums, k):
    """Longest subarray of 1s after flipping at most k zeros."""
    left = 0
    zero_count = 0
    max_length = 0
    
    for right in range(len(nums)):
        # Count zeros in current window
        if nums[right] == 0:
            zero_count += 1
        
        # Shrink window if too many zeros
        while zero_count > k:
            if nums[left] == 0:
                zero_count -= 1
            left += 1
        
        max_length = max(max_length, right - left + 1)
    
    return max_length

# Example: nums = [1,1,1,0,0,0,1,1,1,1,0], k = 2
# Result: 6
```

### 4. **Fruit Into Baskets**
```python
def total_fruit(fruits):
    """Maximum fruits you can collect (at most 2 types)."""
    fruit_count = {}
    left = 0
    max_fruits = 0
    
    for right in range(len(fruits)):
        # Add current fruit
        fruit_count[fruits[right]] = fruit_count.get(fruits[right], 0) + 1
        
        # Shrink window if more than 2 fruit types
        while len(fruit_count) > 2:
            fruit_count[fruits[left]] -= 1
            if fruit_count[fruits[left]] == 0:
                del fruit_count[fruits[left]]
            left += 1
        
        max_fruits = max(max_fruits, right - left + 1)
    
    return max_fruits

# Example: fruits = [1,2,1]
# Result: 3 (all fruits)
```

## Minimum Window Problems

### 1. **Minimum Window Substring**
```python
def min_window(s, t):
    """Minimum window substring containing all characters of t."""
    if not s or not t:
        return ""
    
    from collections import Counter
    
    t_count = Counter(t)
    required = len(t_count)
    formed = 0
    window_count = {}
    
    left = 0
    min_len = float('inf')
    min_left = 0
    
    for right in range(len(s)):
        char = s[right]
        window_count[char] = window_count.get(char, 0) + 1
        
        # Check if current character contributes to desired frequency
        if char in t_count and window_count[char] == t_count[char]:
            formed += 1
        
        # Try to shrink window
        while formed == required and left <= right:
            char = s[left]
            
            # Update minimum window if current is smaller
            if right - left + 1 < min_len:
                min_len = right - left + 1
                min_left = left
            
            window_count[char] -= 1
            if char in t_count and window_count[char] < t_count[char]:
                formed -= 1
            
            left += 1
    
    return "" if min_len == float('inf') else s[min_left:min_left + min_len]

# Example: s = "ADOBECODEBANC", t = "ABC"
# Result: "BANC"
```

### 2. **Minimum Size Subarray Sum**
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

# Example: target = 7, nums = [2,3,1,2,4,3]
# Result: 2 (subarray [4,3])
```

### 3. **Smallest Window with All Characters**
```python
def min_window_all_chars(s):
    """Smallest window containing all unique characters of string."""
    from collections import Counter
    
    char_count = Counter(s)
    required = len(char_count)
    window_count = {}
    formed = 0
    
    left = 0
    min_len = float('inf')
    min_left = 0
    
    for right in range(len(s)):
        char = s[right]
        window_count[char] = window_count.get(char, 0) + 1
        
        if char in char_count and window_count[char] == char_count[char]:
            formed += 1
        
        # Try to shrink window
        while formed == required and left <= right:
            if right - left + 1 < min_len:
                min_len = right - left + 1
                min_left = left
            
            char = s[left]
            window_count[char] -= 1
            if char in char_count and window_count[char] < char_count[char]:
                formed -= 1
            
            left += 1
    
    return s[min_left:min_left + min_len] if min_len != float('inf') else ""

# Time: O(n), Space: O(k) where k is unique characters
```

## Exact Match Problems

### 1. **Subarray Sum Equals K**
```python
def subarray_sum(nums, k):
    """Count subarrays with sum equal to k."""
    from collections import defaultdict
    
    count = 0
    prefix_sum = 0
    sum_count = defaultdict(int)
    sum_count[0] = 1  # Empty prefix
    
    for num in nums:
        prefix_sum += num
        
        # Check if (prefix_sum - k) exists
        if prefix_sum - k in sum_count:
            count += sum_count[prefix_sum - k]
        
        sum_count[prefix_sum] += 1
    
    return count

# Note: This uses prefix sum, not sliding window
# For positive numbers only, can use sliding window
def subarray_sum_positive(nums, k):
    """Count subarrays with sum = k (positive numbers only)."""
    left = 0
    current_sum = 0
    count = 0
    
    for right in range(len(nums)):
        current_sum += nums[right]
        
        # Shrink window if sum > k
        while current_sum > k and left <= right:
            current_sum -= nums[left]
            left += 1
        
        # Check if current sum equals k
        if current_sum == k:
            count += 1
    
    return count
```

### 2. **Binary Subarrays with Sum**
```python
def num_subarrays_with_sum(nums, goal):
    """Number of binary subarrays with sum equal to goal."""
    def at_most(k):
        if k < 0:
            return 0
        
        left = 0
        current_sum = 0
        count = 0
        
        for right in range(len(nums)):
            current_sum += nums[right]
            
            while current_sum > k:
                current_sum -= nums[left]
                left += 1
            
            count += right - left + 1
        
        return count
    
    return at_most(goal) - at_most(goal - 1)

# Time: O(n), Space: O(1)
```

## Advanced Variable Window Techniques

### 1. **Sliding Window with Multiple Conditions**
```python
def complex_window_condition(s, k1, k2):
    """Example with multiple sliding conditions."""
    from collections import defaultdict
    
    char_count = defaultdict(int)
    left = 0
    result = 0
    
    for right in range(len(s)):
        char_count[s[right]] += 1
        
        # Multiple shrinking conditions
        while (len(char_count) > k1 or 
               max(char_count.values()) > k2):
            char_count[s[left]] -= 1
            if char_count[s[left]] == 0:
                del char_count[s[left]]
            left += 1
        
        result = max(result, right - left + 1)
    
    return result
```

### 2. **Sliding Window with State Tracking**
```python
def window_with_state(nums, target):
    """Window that tracks complex state."""
    left = 0
    current_sum = 0
    current_product = 1
    valid_windows = 0
    
    for right in range(len(nums)):
        current_sum += nums[right]
        current_product *= nums[right]
        
        # Shrink based on multiple conditions
        while (left <= right and 
               (current_sum > target or current_product > target)):
            current_sum -= nums[left]
            current_product //= nums[left] if nums[left] != 0 else 1
            left += 1
        
        valid_windows += right - left + 1
    
    return valid_windows
```

## Implementation Strategies

### 1. **Choosing Window Type**
```python
# Maximum/Longest problems
def max_window_template(arr):
    left = 0
    max_result = 0
    
    for right in range(len(arr)):
        # Add arr[right] to window
        
        # Shrink if condition violated
        while condition_violated():
            # Remove arr[left] from window
            left += 1
        
        max_result = max(max_result, right - left + 1)
    
    return max_result

# Minimum problems
def min_window_template(arr):
    left = 0
    min_result = float('inf')
    
    for right in range(len(arr)):
        # Add arr[right] to window
        
        # Shrink while condition satisfied
        while condition_satisfied():
            min_result = min(min_result, right - left + 1)
            # Remove arr[left] from window
            left += 1
    
    return min_result if min_result != float('inf') else 0
```

### 2. **State Management**
```python
# Using hash map for frequency
from collections import defaultdict
char_count = defaultdict(int)

# Using set for uniqueness
char_set = set()

# Using variables for simple state
zero_count = 0
sum_value = 0
```

## Common Patterns

### 1. **At Most K Pattern**
```python
def at_most_k(arr, k):
    """Subarrays with at most k distinct elements."""
    count_map = {}
    left = 0
    result = 0
    
    for right in range(len(arr)):
        count_map[arr[right]] = count_map.get(arr[right], 0) + 1
        
        while len(count_map) > k:
            count_map[arr[left]] -= 1
            if count_map[arr[left]] == 0:
                del count_map[arr[left]]
            left += 1
        
        result += right - left + 1  # All subarrays ending at right
    
    return result

# Exactly k = at_most(k) - at_most(k-1)
```

### 2. **Two Pointer Shrinking**
```python
def two_pointer_shrink(arr, target):
    """Shrink from both ends based on condition."""
    left, right = 0, len(arr) - 1
    result = 0
    
    while left < right:
        current = calculate_value(arr, left, right)
        
        if current == target:
            result += 1
            left += 1
            right -= 1
        elif current < target:
            left += 1
        else:
            right -= 1
    
    return result
```

## Common Mistakes

### 1. **Incorrect Shrinking Condition**
```python
# Wrong: May miss valid windows
while condition_violated():
    left += 1

# Correct: Update state while shrinking
while condition_violated():
    remove_from_state(arr[left])
    left += 1
```

### 2. **Not Handling Empty Windows**
```python
# Add boundary checks
if left > right:
    continue  # or break, depending on problem
```

### 3. **State Inconsistency**
```python
# Ensure state is updated correctly
def add_to_window(char):
    char_count[char] += 1

def remove_from_window(char):
    char_count[char] -= 1
    if char_count[char] == 0:
        del char_count[char]  # Important for distinct count
```

## Practice Problems

### Easy
1. Maximum Average Subarray I (LeetCode 643)
2. Longest Substring Without Repeating Characters (LeetCode 3)

### Medium
1. Minimum Window Substring (LeetCode 76)
2. Longest Substring with At Most K Distinct Characters (LeetCode 340)
3. Max Consecutive Ones III (LeetCode 1004)
4. Minimum Size Subarray Sum (LeetCode 209)
5. Fruit Into Baskets (LeetCode 904)

### Hard
1. Sliding Window Maximum (LeetCode 239)
2. Minimum Window Subsequence (LeetCode 727)
3. Subarrays with K Different Integers (LeetCode 992)

## Next Topics

- [[Fixed_Size_Window]] - Learn about constant window size problems
- [[Sliding_Window_Problems]] - Practice problems using sliding window
- [[Two_Pointers_Overview]] - Related technique for array traversal
