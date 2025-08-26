# Sliding Window Problems

Practice problems using the sliding window technique, organized by window type and difficulty level.

## Fixed Size Window Problems

### Easy Problems

#### 1. Maximum Average Subarray I (LeetCode 643)
```python
def find_max_average(nums, k):
    """Find maximum average of subarray of size k."""
    # Calculate sum of first window
    window_sum = sum(nums[:k])
    max_sum = window_sum
    
    # Slide window and update maximum
    for i in range(k, len(nums)):
        window_sum = window_sum - nums[i - k] + nums[i]
        max_sum = max(max_sum, window_sum)
    
    return max_sum / k

# Time: O(n), Space: O(1)
```

#### 2. Find All Anagrams in a String (LeetCode 438)
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

# Time: O(n), Space: O(k) where k is size of character set
```

### Medium Problems

#### 1. Sliding Window Maximum (LeetCode 239)
```python
def max_sliding_window(nums, k):
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

# Time: O(n), Space: O(k)
```

#### 2. Permutation in String (LeetCode 567)
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

# Time: O(n), Space: O(k)
```

#### 3. Maximum Number of Vowels in Substring (LeetCode 1456)
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

## Variable Size Window Problems

### Easy Problems

#### 1. Best Time to Buy and Sell Stock (LeetCode 121)
```python
def max_profit(prices):
    """Find maximum profit from buying and selling stock once."""
    min_price = float('inf')
    max_profit = 0
    
    for price in prices:
        if price < min_price:
            min_price = price
        else:
            max_profit = max(max_profit, price - min_price)
    
    return max_profit

# Time: O(n), Space: O(1)
```

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

# Time: O(n), Space: O(min(m,n))
```

#### 2. Minimum Size Subarray Sum (LeetCode 209)
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

#### 3. Max Consecutive Ones III (LeetCode 1004)
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

#### 4. Fruit Into Baskets (LeetCode 904)
```python
def total_fruit(fruits):
    """Maximum fruits collected (at most 2 types)."""
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

# Time: O(n), Space: O(1)
```

#### 5. Longest Substring with At Most K Distinct Characters (LeetCode 340)
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

# Time: O(n), Space: O(k)
```

### Hard Problems

#### 1. Minimum Window Substring (LeetCode 76)
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

# Time: O(|s| + |t|), Space: O(|s| + |t|)
```

#### 2. Subarrays with K Different Integers (LeetCode 992)
```python
def subarrays_with_k_distinct(nums, k):
    """Count subarrays with exactly k distinct integers."""
    def at_most_k(k):
        if k == 0:
            return 0
        
        count_map = {}
        left = 0
        result = 0
        
        for right in range(len(nums)):
            count_map[nums[right]] = count_map.get(nums[right], 0) + 1
            
            while len(count_map) > k:
                count_map[nums[left]] -= 1
                if count_map[nums[left]] == 0:
                    del count_map[nums[left]]
                left += 1
            
            result += right - left + 1
        
        return result
    
    return at_most_k(k) - at_most_k(k - 1)

# Time: O(n), Space: O(k)
```

## Specialized Sliding Window Problems

### String Pattern Problems

#### 1. Longest Repeating Character Replacement (LeetCode 424)
```python
def character_replacement(s, k):
    """Longest substring with same character after k replacements."""
    char_count = {}
    left = 0
    max_length = 0
    max_freq = 0
    
    for right in range(len(s)):
        char_count[s[right]] = char_count.get(s[right], 0) + 1
        max_freq = max(max_freq, char_count[s[right]])
        
        # If window size - max frequency > k, shrink window
        if right - left + 1 - max_freq > k:
            char_count[s[left]] -= 1
            left += 1
        
        max_length = max(max_length, right - left + 1)
    
    return max_length

# Time: O(n), Space: O(1) since at most 26 characters
```

#### 2. Minimum Window Subsequence (LeetCode 727)
```python
def min_window(s, t):
    """Minimum window subsequence of s that contains t."""
    m, n = len(s), len(t)
    min_len = float('inf')
    min_start = 0
    
    i = 0
    while i < m:
        j = 0
        # Find subsequence starting from i
        start = i
        while i < m and j < n:
            if s[i] == t[j]:
                j += 1
            i += 1
        
        if j == n:  # Found complete subsequence
            # Shrink from right to find minimum window
            i -= 1
            j -= 1
            while j >= 0:
                if s[i] == t[j]:
                    j -= 1
                i -= 1
            i += 1
            
            if i - start < min_len:
                min_len = i - start
                min_start = start
        else:
            break
    
    return "" if min_len == float('inf') else s[min_start:min_start + min_len]

# Time: O(m * n), Space: O(1)
```

### Array Sum Problems

#### 1. Binary Subarrays With Sum (LeetCode 930)
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

#### 2. Subarray Product Less Than K (LeetCode 713)
```python
def num_subarrays_less_than_k(nums, k):
    """Count subarrays with product less than k."""
    if k <= 1:
        return 0
    
    left = 0
    product = 1
    count = 0
    
    for right in range(len(nums)):
        product *= nums[right]
        
        while product >= k:
            product //= nums[left]
            left += 1
        
        count += right - left + 1
    
    return count

# Time: O(n), Space: O(1)
```

## Advanced Sliding Window Techniques

### Multiple Conditions

#### 1. Number of Substrings Containing All Three Characters (LeetCode 1358)
```python
def number_of_substrings(s):
    """Count substrings containing all three characters a, b, c."""
    last_seen = {'a': -1, 'b': -1, 'c': -1}
    count = 0
    
    for i, char in enumerate(s):
        last_seen[char] = i
        
        # Count substrings ending at position i
        min_index = min(last_seen.values())
        if min_index != -1:
            count += min_index + 1
    
    return count

# Time: O(n), Space: O(1)
```

### Optimization Problems

#### 1. Get Equal Substrings Within Budget (LeetCode 1208)
```python
def equal_substring(s, t, max_cost):
    """Longest substring where cost <= max_cost."""
    def get_cost(c1, c2):
        return abs(ord(c1) - ord(c2))
    
    left = 0
    current_cost = 0
    max_length = 0
    
    for right in range(len(s)):
        current_cost += get_cost(s[right], t[right])
        
        # Shrink window if cost exceeds budget
        while current_cost > max_cost:
            current_cost -= get_cost(s[left], t[left])
            left += 1
        
        max_length = max(max_length, right - left + 1)
    
    return max_length

# Time: O(n), Space: O(1)
```

## Problem-Solving Strategies

### 1. **Identifying Window Type**
```python
def identify_window_type(problem_description):
    if "size k" in problem_description or "length k" in problem_description:
        return "fixed_size"
    elif "at most" in problem_description or "maximum" in problem_description:
        return "variable_size_maximum"
    elif "minimum" in problem_description:
        return "variable_size_minimum"
    elif "exactly" in problem_description:
        return "at_most_trick"  # exactly k = at_most(k) - at_most(k-1)
    else:
        return "analyze_further"
```

### 2. **Template Selection**
```python
# Fixed size template
def fixed_size_template(arr, k):
    window_state = initialize_window(arr[:k])
    result = [process_window(window_state)]
    
    for i in range(k, len(arr)):
        update_window(window_state, remove=arr[i-k], add=arr[i])
        result.append(process_window(window_state))
    
    return result

# Variable size maximum template
def variable_max_template(arr, condition):
    left = 0
    max_result = 0
    window_state = initialize_state()
    
    for right in range(len(arr)):
        update_state(window_state, arr[right])
        
        while violates_condition(window_state, condition):
            remove_from_state(window_state, arr[left])
            left += 1
        
        max_result = max(max_result, right - left + 1)
    
    return max_result

# Variable size minimum template
def variable_min_template(arr, condition):
    left = 0
    min_result = float('inf')
    window_state = initialize_state()
    
    for right in range(len(arr)):
        update_state(window_state, arr[right])
        
        while satisfies_condition(window_state, condition):
            min_result = min(min_result, right - left + 1)
            remove_from_state(window_state, arr[left])
            left += 1
    
    return min_result if min_result != float('inf') else 0
```

### 3. **Common Optimizations**

#### At Most K Trick
```python
def exactly_k(arr, k):
    return at_most_k(arr, k) - at_most_k(arr, k - 1)
```

#### Early Termination
```python
# For minimum problems
if current_window_size > min_found:
    break  # Can't get smaller
```

#### State Caching
```python
# Cache expensive computations
if state not in cache:
    cache[state] = expensive_computation(state)
```

## Practice Schedule

### Week 1: Fixed Size Windows
1. Maximum Average Subarray I
2. Find All Anagrams in a String  
3. Sliding Window Maximum
4. Maximum Number of Vowels

### Week 2: Variable Size Windows
1. Longest Substring Without Repeating Characters
2. Minimum Size Subarray Sum
3. Max Consecutive Ones III
4. Fruit Into Baskets

### Week 3: Advanced Problems
1. Minimum Window Substring
2. Longest Repeating Character Replacement
3. Subarrays with K Different Integers
4. Minimum Window Subsequence

## Common Pitfalls

### 1. **Window State Management**
```python
# Wrong: Not updating state correctly
window_sum += arr[right]
# Missing: window_sum -= arr[left] when shrinking

# Correct: Update state for both expand and shrink
window_sum += arr[right]  # Expand
while condition_violated:
    window_sum -= arr[left]  # Shrink
    left += 1
```

### 2. **Boundary Conditions**
```python
# Check for empty input
if not arr or k <= 0:
    return appropriate_default

# Ensure k doesn't exceed array length
if k > len(arr):
    return handle_edge_case()
```

### 3. **State Consistency**
```python
# Ensure hash map state is consistent
char_count[char] -= 1
if char_count[char] == 0:
    del char_count[char]  # Important for distinct count problems
```

## Next Topics

- [[Fixed_Size_Window]] - Deep dive into fixed window techniques
- [[Variable_Size_Window]] - Advanced variable window patterns
- [[Two_Pointers_Overview]] - Related technique for array problems
