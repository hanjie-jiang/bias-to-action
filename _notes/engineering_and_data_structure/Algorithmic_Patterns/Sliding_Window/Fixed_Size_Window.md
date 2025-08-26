# Fixed Size Window

Fixed size sliding window maintains a constant window size while traversing through data. This technique is particularly effective for problems involving subarrays or substrings of a specific length.

## Core Concept

In fixed size window problems:
1. **Window size remains constant** throughout the traversal
2. **Add one element** from the right
3. **Remove one element** from the left
4. **Process current window** state

## Basic Template

```python
def fixed_window_template(arr, k):
    """Template for fixed size window problems."""
    if len(arr) < k:
        return []
    
    # Initialize first window
    window_sum = 0
    for i in range(k):
        window_sum += arr[i]
    
    result = [process_window(window_sum)]
    
    # Slide the window
    for i in range(k, len(arr)):
        # Remove leftmost element, add rightmost element
        window_sum = window_sum - arr[i - k] + arr[i]
        result.append(process_window(window_sum))
    
    return result
```

## Common Problem Types

### 1. **Maximum/Minimum in Window**

#### Maximum Sum Subarray of Size K
```python
def max_sum_subarray(arr, k):
    """Find maximum sum of subarray of size k."""
    if len(arr) < k:
        return 0
    
    # Calculate sum of first window
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    # Slide window and update maximum
    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i - k] + arr[i]
        max_sum = max(max_sum, window_sum)
    
    return max_sum

# Example: arr = [2, 1, 5, 1, 3, 2], k = 3
# Windows: [2,1,5]=8, [1,5,1]=7, [5,1,3]=9, [1,3,2]=6
# Result: 9
```

#### Average of Subarrays of Size K
```python
def find_averages(arr, k):
    """Find average of each subarray of size k."""
    if len(arr) < k:
        return []
    
    result = []
    window_sum = sum(arr[:k])
    result.append(window_sum / k)
    
    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i - k] + arr[i]
        result.append(window_sum / k)
    
    return result

# Time: O(n), Space: O(1) excluding result
```

### 2. **Pattern Matching**

#### Find All Anagrams in String
```python
def find_anagrams(s, p):
    """Find all start indices of anagrams of p in s."""
    if len(p) > len(s):
        return []
    
    from collections import Counter
    
    p_count = Counter(p)
    window_count = Counter()
    result = []
    
    for i in range(len(s)):
        # Add current character
        window_count[s[i]] += 1
        
        # Remove character outside window
        if i >= len(p):
            left_char = s[i - len(p)]
            window_count[left_char] -= 1
            if window_count[left_char] == 0:
                del window_count[left_char]
        
        # Check if current window is anagram
        if i >= len(p) - 1 and window_count == p_count:
            result.append(i - len(p) + 1)
    
    return result

# Example: s = "abab", p = "ab"
# Result: [0, 2] (anagrams at indices 0 and 2)
```

#### Permutation in String
```python
def check_inclusion(s1, s2):
    """Check if s2 contains permutation of s1."""
    if len(s1) > len(s2):
        return False
    
    from collections import Counter
    
    s1_count = Counter(s1)
    window_count = Counter()
    
    for i in range(len(s2)):
        # Add current character
        window_count[s2[i]] += 1
        
        # Remove character outside window
        if i >= len(s1):
            left_char = s2[i - len(s1)]
            window_count[left_char] -= 1
            if window_count[left_char] == 0:
                del window_count[left_char]
        
        # Check if current window is permutation
        if i >= len(s1) - 1 and window_count == s1_count:
            return True
    
    return False

# Time: O(n), Space: O(k) where k is size of character set
```

### 3. **Sliding Window Maximum/Minimum**

#### Sliding Window Maximum
```python
def sliding_window_maximum(nums, k):
    """Find maximum in each sliding window of size k."""
    from collections import deque
    
    if not nums or k == 0:
        return []
    
    dq = deque()  # Store indices
    result = []
    
    for i in range(len(nums)):
        # Remove elements outside current window
        while dq and dq[0] <= i - k:
            dq.popleft()
        
        # Remove smaller elements (they can't be maximum)
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        
        dq.append(i)
        
        # Add maximum of current window to result
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result

# Example: nums = [1,3,-1,-3,5,3,6,7], k = 3
# Result: [3,3,5,5,6,7]
```

#### First Negative in Window
```python
def first_negative_in_window(arr, k):
    """Find first negative number in each window of size k."""
    from collections import deque
    
    negatives = deque()  # Store indices of negative numbers
    result = []
    
    for i in range(len(arr)):
        # Add current element if negative
        if arr[i] < 0:
            negatives.append(i)
        
        # Remove elements outside current window
        while negatives and negatives[0] <= i - k:
            negatives.popleft()
        
        # Add result for current window
        if i >= k - 1:
            if negatives:
                result.append(arr[negatives[0]])
            else:
                result.append(0)  # No negative number
    
    return result

# Time: O(n), Space: O(k)
```

## Advanced Fixed Window Problems

### 1. **String Problems**

#### Maximum Number of Vowels in Substring
```python
def max_vowels(s, k):
    """Find maximum number of vowels in any substring of length k."""
    vowels = set('aeiou')
    
    # Count vowels in first window
    vowel_count = sum(1 for char in s[:k] if char in vowels)
    max_vowels = vowel_count
    
    # Slide window
    for i in range(k, len(s)):
        # Remove leftmost character
        if s[i - k] in vowels:
            vowel_count -= 1
        
        # Add rightmost character
        if s[i] in vowels:
            vowel_count += 1
        
        max_vowels = max(max_vowels, vowel_count)
    
    return max_vowels

# Time: O(n), Space: O(1)
```

#### Get Equal Substrings Within Budget
```python
def equal_substring(s, t, max_cost):
    """Find length of longest substring where cost <= max_cost."""
    def get_cost(c1, c2):
        return abs(ord(c1) - ord(c2))
    
    left = 0
    current_cost = 0
    max_length = 0
    
    for right in range(len(s)):
        # Add cost of current character
        current_cost += get_cost(s[right], t[right])
        
        # Shrink window if cost exceeds budget
        while current_cost > max_cost:
            current_cost -= get_cost(s[left], t[left])
            left += 1
        
        max_length = max(max_length, right - left + 1)
    
    return max_length

# Time: O(n), Space: O(1)
```

### 2. **Array Problems**

#### Maximum Points from Cards
```python
def max_score(card_points, k):
    """Maximum score from taking k cards from either end."""
    n = len(card_points)
    
    # Take all k cards from left initially
    current_sum = sum(card_points[:k])
    max_sum = current_sum
    
    # Try taking i cards from right, k-i from left
    for i in range(1, k + 1):
        # Remove one from left, add one from right
        current_sum = current_sum - card_points[k - i] + card_points[n - i]
        max_sum = max(max_sum, current_sum)
    
    return max_sum

# Time: O(k), Space: O(1)
```

#### Defuse the Bomb
```python
def decrypt(code, k):
    """Decrypt circular array based on k."""
    n = len(code)
    result = [0] * n
    
    if k == 0:
        return result
    
    # Determine window direction
    start = 1 if k > 0 else n + k
    end = k if k > 0 else n - 1
    
    # Calculate sum for first window
    window_sum = sum(code[i % n] for i in range(start, start + abs(k)))
    
    for i in range(n):
        result[i] = window_sum
        
        # Slide window
        window_sum -= code[start % n]
        window_sum += code[(start + abs(k)) % n]
        start += 1
    
    return result

# Time: O(n), Space: O(1) excluding result
```

## Implementation Tips

### 1. **Boundary Handling**
```python
def safe_fixed_window(arr, k):
    # Always check if array is large enough
    if not arr or k <= 0 or k > len(arr):
        return []
    
    # Handle edge case of k = 1
    if k == 1:
        return arr[:]
    
    # Regular sliding window logic
    # ...
```

### 2. **Efficient Updates**
```python
# Good: O(1) per window
window_sum = window_sum - arr[i - k] + arr[i]

# Bad: O(k) per window
window_sum = sum(arr[i - k + 1:i + 1])
```

### 3. **Using Data Structures**
```python
from collections import deque, Counter

# For min/max tracking
dq = deque()

# For frequency counting
counter = Counter()

# For set operations
char_set = set()
```

## Common Mistakes

### 1. **Index Errors**
```python
# Wrong: May go out of bounds
for i in range(len(arr)):
    if i >= k:
        # Process window arr[i-k+1:i+1]

# Correct: Proper bounds checking
for i in range(k - 1, len(arr)):
    # Process window arr[i-k+1:i+1]
```

### 2. **Not Handling Edge Cases**
```python
def robust_sliding_window(arr, k):
    if not arr or k <= 0:
        return []
    
    if k > len(arr):
        return []  # or return [sum(arr)] depending on problem
    
    # Main logic here
```

### 3. **Inefficient Window Recalculation**
```python
# Inefficient: O(n*k)
for i in range(len(arr) - k + 1):
    window_sum = sum(arr[i:i+k])  # Recalculates every time

# Efficient: O(n)
window_sum = sum(arr[:k])
for i in range(k, len(arr)):
    window_sum = window_sum - arr[i-k] + arr[i]
```

## Practice Problems

### Easy
1. Maximum Average Subarray I (LeetCode 643)
2. Find All Anagrams in a String (LeetCode 438)
3. Defuse the Bomb (LeetCode 1652)

### Medium
1. Sliding Window Maximum (LeetCode 239)
2. Maximum Number of Vowels (LeetCode 1456)
3. Get Equal Substrings Within Budget (LeetCode 1208)
4. Maximum Points You Can Obtain from Cards (LeetCode 1423)

### Hard
1. Minimum Window Substring (LeetCode 76)
2. Sliding Window Median (LeetCode 480)

## Next Topics

- [[Variable_Size_Window]] - Learn about dynamic window sizing
- [[Sliding_Window_Problems]] - Practice problems using sliding window
- [[Two_Pointers_Overview]] - Related technique for array traversal
