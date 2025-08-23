# Hash Table Problems

This section covers common coding problems that can be efficiently solved using hash tables. These problems frequently appear in technical interviews and demonstrate the power of hash-based data structures.

## Problem Categories

### 1. Frequency Counting
### 2. Two-Sum Variants
### 3. Substring Problems
### 4. Set Operations
### 5. Caching and Memoization

---

## 1. Frequency Counting Problems

### Character Frequency
**Problem**: Count the frequency of each character in a string.

```python
def char_frequency(s):
    """
    Time: O(n), Space: O(k) where k is number of unique characters
    """
    freq = {}
    for char in s:
        freq[char] = freq.get(char, 0) + 1
    return freq

# Example
text = "hello world"
print(char_frequency(text))
# Output: {'h': 1, 'e': 1, 'l': 3, 'o': 2, ' ': 1, 'w': 1, 'r': 1, 'd': 1}
```

### Most Frequent Element
**Problem**: Find the most frequently occurring element in an array.

```python
def most_frequent(arr):
    """
    Time: O(n), Space: O(n)
    """
    if not arr:
        return None
    
    freq = {}
    for num in arr:
        freq[num] = freq.get(num, 0) + 1
    
    return max(freq, key=freq.get)

# Example
numbers = [1, 3, 2, 3, 4, 3, 2]
print(most_frequent(numbers))  # Output: 3
```

---

## 2. Two-Sum Variants

### Classic Two Sum
**Problem**: Find two numbers in array that add up to target.

```python
def two_sum(nums, target):
    """
    Time: O(n), Space: O(n)
    """
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

# Example
nums = [2, 7, 11, 15]
target = 9
print(two_sum(nums, target))  # Output: [0, 1]
```

### Three Sum
**Problem**: Find all unique triplets that sum to zero.

```python
def three_sum(nums):
    """
    Time: O(n²), Space: O(1) excluding output
    """
    nums.sort()
    result = []
    
    for i in range(len(nums) - 2):
        # Skip duplicates for first number
        if i > 0 and nums[i] == nums[i-1]:
            continue
            
        left, right = i + 1, len(nums) - 1
        
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            
            if total == 0:
                result.append([nums[i], nums[left], nums[right]])
                
                # Skip duplicates
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                    
                left += 1
                right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1
                
    return result
```

---

## 3. Substring Problems

### Longest Substring Without Repeating Characters
**Problem**: Find length of longest substring without repeating characters.

```python
def length_of_longest_substring(s):
    """
    Time: O(n), Space: O(min(m,n)) where m is character set size
    """
    char_index = {}
    left = 0
    max_length = 0
    
    for right, char in enumerate(s):
        if char in char_index and char_index[char] >= left:
            left = char_index[char] + 1
        
        char_index[char] = right
        max_length = max(max_length, right - left + 1)
    
    return max_length

# Example
s = "abcabcbb"
print(length_of_longest_substring(s))  # Output: 3 ("abc")
```

### Group Anagrams
**Problem**: Group strings that are anagrams of each other.

```python
def group_anagrams(strs):
    """
    Time: O(n * k log k) where n is number of strings, k is max string length
    Space: O(n * k)
    """
    from collections import defaultdict
    
    anagram_groups = defaultdict(list)
    
    for s in strs:
        # Sort characters to create key
        key = ''.join(sorted(s))
        anagram_groups[key].append(s)
    
    return list(anagram_groups.values())

# Example
words = ["eat", "tea", "tan", "ate", "nat", "bat"]
print(group_anagrams(words))
# Output: \[\[['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']\]
```

---

## 4. Set Operations

### Intersection of Two Arrays
**Problem**: Find intersection of two arrays.

```python
def intersection(nums1, nums2):
    """
    Time: O(n + m), Space: O(min(n,m))
    """
    set1 = set(nums1)
    result = set()
    
    for num in nums2:
        if num in set1:
            result.add(num)
    
    return list(result)

# Example
nums1 = [1, 2, 2, 1]
nums2 = [2, 2]
print(intersection(nums1, nums2))  # Output: [2]
```

### Find Missing Number
**Problem**: Find missing number in array containing n distinct numbers from 0 to n.

```python
def missing_number(nums):
    """
    Multiple approaches - Hash table version
    Time: O(n), Space: O(n)
    """
    num_set = set(nums)
    n = len(nums)
    
    for i in range(n + 1):
        if i not in num_set:
            return i
    
    return -1  # Should never reach here

# More efficient approaches:
def missing_number_math(nums):
    """
    Time: O(n), Space: O(1)
    """
    n = len(nums)
    expected_sum = n * (n + 1) // 2
    actual_sum = sum(nums)
    return expected_sum - actual_sum

def missing_number_xor(nums):
    """
    Time: O(n), Space: O(1)
    """
    missing = len(nums)
    for i, num in enumerate(nums):
        missing ^= i ^ num
    return missing
```

---

## 5. Caching and Memoization

### LRU Cache
**Problem**: Implement Least Recently Used cache.

```python
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}  # key -> node
        
        # Create dummy head and tail nodes
        self.head = Node(0, 0)
        self.tail = Node(0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def get(self, key):
        if key in self.cache:
            node = self.cache[key]
            # Move to head (most recently used)
            self._remove(node)
            self._add_to_head(node)
            return node.value
        return -1
    
    def put(self, key, value):
        if key in self.cache:
            # Update existing
            node = self.cache[key]
            node.value = value
            self._remove(node)
            self._add_to_head(node)
        else:
            # Add new
            if len(self.cache) >= self.capacity:
                # Remove least recently used
                lru = self.tail.prev
                self._remove(lru)
                del self.cache[lru.key]
            
            node = Node(key, value)
            self.cache[key] = node
            self._add_to_head(node)
    
    def _remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev
    
    def _add_to_head(self, node):
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
        node.prev = self.head

class Node:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None
```

### Fibonacci with Memoization
**Problem**: Compute Fibonacci numbers efficiently.

```python
def fibonacci_memo(n, memo={}):
    """
    Time: O(n), Space: O(n)
    """
    if n in memo:
        return memo[n]
    
    if n <= 1:
        return n
    
    memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)
    return memo[n]

# Using functools.lru_cache decorator
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci_lru(n):
    if n <= 1:
        return n
    return fibonacci_lru(n-1) + fibonacci_lru(n-2)
```

---

## Problem-Solving Patterns

### When to Use Hash Tables

1. **Fast Lookups**: Need O(1) average case access
2. **Counting**: Frequency analysis problems
3. **Caching**: Store computed results
4. **Set Operations**: Intersection, union, difference
5. **Mapping**: Key-value relationships

### Common Techniques

1. **Use as Counter**: `collections.Counter` or manual counting
2. **Use as Set**: Fast membership testing
3. **Use as Cache**: Store expensive computations
4. **Two-Pointer with Hash**: Combine with other techniques
5. **Sliding Window with Hash**: Track elements in window

### Time/Space Trade-offs

- **Time**: Usually improves from O(n²) to O(n)
- **Space**: Additional O(n) space for hash table
- **Worth it**: When lookup speed is critical

## Practice Problems

### Easy
1. Contains Duplicate
2. Valid Anagram
3. Two Sum
4. Intersection of Two Arrays

### Medium
1. Group Anagrams
2. Top K Frequent Elements
3. Longest Substring Without Repeating Characters
4. 4Sum II

### Hard
1. Substring with Concatenation of All Words
2. LRU Cache
3. First Missing Positive
4. Longest Consecutive Sequence

## Related Topics
- [Python Dictionaries](Python_Dictionaries.md) - Implementation details
- [Python Sets](Python_Sets.md) - Set operations
- [Time Complexity Guide](../../Resources/Time_Complexity_Guide.md) - Analysis techniques
