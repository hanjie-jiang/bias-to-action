# Sliding Window Overview

The Sliding Window technique is a powerful algorithmic pattern used to solve problems involving subarrays or substrings. It optimizes brute force solutions from O(n²) or O(n³) to O(n) by maintaining a "window" that slides through the data.

## What is Sliding Window?

Sliding Window maintains a subset (window) of elements and efficiently updates this window as it moves through the data structure. Instead of recalculating everything for each position, it adds new elements and removes old ones.

## Types of Sliding Windows

### 1. Fixed Size Window
The window size remains constant throughout the traversal.

```python
def max_sum_subarray(arr, k):
    """Find maximum sum of subarray of size k."""
    if len(arr) < k:
        return None
    
    # Calculate sum of first window
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    # Slide the window
    for i in range(k, len(arr)):
        # Remove leftmost element, add rightmost element
        window_sum = window_sum - arr[i - k] + arr[i]
        max_sum = max(max_sum, window_sum)
    
    return max_sum
```

### 2. Variable Size Window
The window size changes based on certain conditions.

```python
def longest_substring_without_repeating(s):
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
```

## When to Use Sliding Window

### ✅ Perfect for:
- **Subarray/substring problems**: Finding max, min, or specific properties
- **Contiguous sequences**: Problems requiring consecutive elements
- **Optimization problems**: Converting O(n²) to O(n)
- **String pattern matching**: When pattern has specific constraints

### ❌ Not suitable for:
- **Non-contiguous sequences**: When elements don't need to be adjacent
- **Global optimization**: When you need to consider all possible combinations
- **Complex dependencies**: When current window depends on distant elements

## Fixed Size Window Patterns

### 1. Maximum/Minimum in Window
```python
def max_in_sliding_window(arr, k):
    """Find maximum element in each sliding window of size k."""
    from collections import deque
    
    result = []
    dq = deque()  # Store indices
    
    for i in range(len(arr)):
        # Remove elements outside current window
        while dq and dq[0] <= i - k:
            dq.popleft()
        
        # Remove smaller elements (they can't be maximum)
        while dq and arr[dq[-1]] < arr[i]:
            dq.pop()
        
        dq.append(i)
        
        # Add to result if window is complete
        if i >= k - 1:
            result.append(arr[dq[0]])
    
    return result

def average_of_subarrays(arr, k):
    """Find average of each subarray of size k."""
    result = []
    window_sum = sum(arr[:k])
    result.append(window_sum / k)
    
    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i - k] + arr[i]
        result.append(window_sum / k)
    
    return result
```

### 2. Pattern Matching
```python
def find_anagrams(s, p):
    """Find all anagrams of p in s."""
    if len(p) > len(s):
        return []
    
    from collections import Counter
    
    p_count = Counter(p)
    window_count = Counter()
    result = []
    
    for i in range(len(s)):
        # Add current character to window
        window_count[s[i]] += 1
        
        # Remove character outside window
        if i >= len(p):
            left_char = s[i - len(p)]
            window_count[left_char] -= 1
            if window_count[left_char] == 0:
                del window_count[left_char]
        
        # Check if current window is anagram
        if window_count == p_count:
            result.append(i - len(p) + 1)
    
    return result
```

## Variable Size Window Patterns

### 1. Longest Substring Problems
```python
def longest_substring_k_distinct(s, k):
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

def longest_ones_with_k_flips(arr, k):
    """Longest subarray of 1s after flipping at most k zeros."""
    left = 0
    zero_count = 0
    max_length = 0
    
    for right in range(len(arr)):
        if arr[right] == 0:
            zero_count += 1
        
        # Shrink window if too many zeros
        while zero_count > k:
            if arr[left] == 0:
                zero_count -= 1
            left += 1
        
        max_length = max(max_length, right - left + 1)
    
    return max_length
```

### 2. Minimum Window Problems
```python
def min_window_substring(s, t):
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
        
        if char in t_count and window_count[char] == t_count[char]:
            formed += 1
        
        # Try to shrink window
        while formed == required and left <= right:
            char = s[left]
            
            # Update minimum window
            if right - left + 1 < min_len:
                min_len = right - left + 1
                min_left = left
            
            window_count[char] -= 1
            if char in t_count and window_count[char] < t_count[char]:
                formed -= 1
            
            left += 1
    
    return "" if min_len == float('inf') else s[min_left:min_left + min_len]

def min_subarray_sum(arr, target):
    """Minimum length subarray with sum >= target."""
    left = 0
    min_length = float('inf')
    current_sum = 0
    
    for right in range(len(arr)):
        current_sum += arr[right]
        
        # Shrink window while sum >= target
        while current_sum >= target:
            min_length = min(min_length, right - left + 1)
            current_sum -= arr[left]
            left += 1
    
    return min_length if min_length != float('inf') else 0
```

## Time and Space Complexity

### Time Complexity
- **Fixed window**: O(n) - each element visited once
- **Variable window**: O(n) - each element added and removed at most once
- **With hash maps**: O(n) - hash operations are O(1) average

### Space Complexity
- **Basic sliding window**: O(1) - only store window boundaries
- **With data structures**: O(k) where k is window size or distinct elements
- **Result storage**: O(n) if storing all results

## Common Implementation Patterns

### 1. Two Pointer Template
```python
def sliding_window_template(arr):
    left = 0
    # Initialize window state
    
    for right in range(len(arr)):
        # Add arr[right] to window
        
        # Shrink window if necessary
        while condition_to_shrink:
            # Remove arr[left] from window
            left += 1
        
        # Update result with current window
    
    return result
```

### 2. Fixed Window Template
```python
def fixed_window_template(arr, k):
    # Initialize first window
    for i in range(k):
        # Add arr[i] to window
    
    # Process first window
    result = [process_window()]
    
    # Slide window
    for i in range(k, len(arr)):
        # Remove arr[i-k], add arr[i]
        result.append(process_window())
    
    return result
```

## Optimization Techniques

### 1. Using Deque for Min/Max
```python
from collections import deque

def sliding_window_maximum(arr, k):
    """Efficient maximum in sliding window using deque."""
    dq = deque()
    result = []
    
    for i in range(len(arr)):
        # Remove elements outside window
        while dq and dq[0] <= i - k:
            dq.popleft()
        
        # Maintain decreasing order
        while dq and arr[dq[-1]] < arr[i]:
            dq.pop()
        
        dq.append(i)
        
        if i >= k - 1:
            result.append(arr[dq[0]])
    
    return result
```

### 2. Counter for Character Frequency
```python
from collections import Counter

def sliding_window_with_counter(s, pattern):
    """Use Counter for efficient character frequency tracking."""
    pattern_count = Counter(pattern)
    window_count = Counter()
    matches = 0
    
    for char in s:
        # Add character
        window_count[char] += 1
        if window_count[char] == pattern_count[char]:
            matches += 1
        
        # Remove excess characters
        # ... shrinking logic
```

## Common Mistakes

### 1. **Incorrect Window Bounds**
```python
# Wrong: Off-by-one error
for right in range(len(arr)):
    if right >= k - 1:  # Should be right >= k - 1
        # Process window arr[right-k+1:right+1]

# Correct: Proper indexing
window_start = right - k + 1
```

### 2. **Not Handling Edge Cases**
```python
def robust_sliding_window(arr, k):
    if not arr or k <= 0 or k > len(arr):
        return []
    # ... implementation
```

### 3. **Inefficient Window Updates**
```python
# Wrong: Recalculating entire window
def inefficient_max_sum(arr, k):
    max_sum = 0
    for i in range(len(arr) - k + 1):
        current_sum = sum(arr[i:i+k])  # O(k) for each position
        max_sum = max(max_sum, current_sum)
    return max_sum

# Correct: Update incrementally
def efficient_max_sum(arr, k):
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i-k] + arr[i]  # O(1) update
        max_sum = max(max_sum, window_sum)
    
    return max_sum
```

## Practice Problems

### Easy
1. Maximum Sum Subarray of Size K
2. Average of Subarrays of Size K
3. Find All Anagrams in a String

### Medium
1. Longest Substring Without Repeating Characters
2. Minimum Window Substring
3. Longest Substring with At Most K Distinct Characters
4. Max Consecutive Ones III

### Hard
1. Sliding Window Maximum
2. Minimum Window Subsequence
3. Longest Substring with At Most Two Distinct Characters

## Next Topics

- [[Fixed_Size_Window]] - Deep dive into fixed window problems
- [[Variable_Size_Window]] - Advanced variable window techniques
- [[Two_Pointers_Overview]] - Related technique for array problems
- [[Sliding_Window_Problems]] - Practice problems and solutions
