# Opposite Direction Pointers

Opposite direction pointers (also called convergent pointers) start from opposite ends of the data structure and move toward each other. This technique is particularly effective for problems on sorted arrays and palindrome checking.

## Core Concept

In opposite direction pointer problems:
1. **Left pointer** starts at the beginning (index 0)
2. **Right pointer** starts at the end (index n-1)
3. **Pointers move toward each other** based on conditions
4. **Stop when pointers meet** or cross

## Basic Template

```python
def opposite_pointers_template(arr):
    """Template for opposite direction pointers."""
    left, right = 0, len(arr) - 1
    
    while left < right:
        # Process current pair
        if condition_met(arr[left], arr[right]):
            # Found solution or update result
            process_result(left, right)
            left += 1
            right -= 1
        elif arr[left] < arr[right]:  # or custom condition
            left += 1
        else:
            right -= 1
    
    return result
```

## Target Sum Problems

### 1. **Two Sum in Sorted Array**
```python
def two_sum_sorted(numbers, target):
    """Find two numbers that add up to target in sorted array."""
    left, right = 0, len(numbers) - 1
    
    while left < right:
        current_sum = numbers[left] + numbers[right]
        
        if current_sum == target:
            return [left + 1, right + 1]  # 1-indexed
        elif current_sum < target:
            left += 1  # Need larger sum
        else:
            right -= 1  # Need smaller sum
    
    return []  # No solution found

# Example: numbers = [2,7,11,15], target = 9
# Result: [1, 2] (2 + 7 = 9)
```

### 2. **Three Sum**
```python
def three_sum(nums):
    """Find all unique triplets that sum to zero."""
    nums.sort()  # Essential for this approach
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

# Example: nums = [-1,0,1,2,-1,-4]
# Result: triplets \[\[-1, -1, 2], [-1, 0, 1\]\]
```

### 3. **Three Sum Closest**
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
            
            # Update closest sum if current is closer
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

### 4. **Four Sum**
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

## Palindrome Problems

### 1. **Valid Palindrome**
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

# Example: s = "A man, a plan, a canal: Panama"
# Result: True
```

### 2. **Valid Palindrome II**
```python
def valid_palindrome_deletion(s):
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
    
    return True  # Already a palindrome

# Example: s = "aba" → True, s = "abca" → True (delete 'c')
```

### 3. **Palindromic Substrings**
```python
def count_palindromic_substrings(s):
    """Count all palindromic substrings."""
    def expand_around_center(left, right):
        count = 0
        while left >= 0 and right < len(s) and s[left] == s[right]:
            count += 1
            left -= 1
            right += 1
        return count
    
    total_count = 0
    
    for i in range(len(s)):
        # Odd length palindromes (center at i)
        total_count += expand_around_center(i, i)
        
        # Even length palindromes (center between i and i+1)
        total_count += expand_around_center(i, i + 1)
    
    return total_count

# Example: s = "abc" → 3, s = "aaa" → 6
```

## Container and Trapping Problems

### 1. **Container With Most Water**
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

# Example: height = [1,8,6,2,5,4,8,3,7]
# Result: 49
```

### 2. **Trapping Rain Water**
```python
def trap_rain_water(height):
    """Calculate trapped rainwater."""
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

# Example: height = [0,1,0,2,1,0,1,3,2,1,2,1]
# Result: 6
```

## Array Manipulation

### 1. **Reverse Array**
```python
def reverse_array(arr):
    """Reverse array in-place using two pointers."""
    left, right = 0, len(arr) - 1
    
    while left < right:
        arr[left], arr[right] = arr[right], arr[left]
        left += 1
        right -= 1
    
    return arr

# Time: O(n), Space: O(1)
```

### 2. **Remove Duplicates from Sorted Array**
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

# Example: nums = [1,1,2] → returns 2, nums becomes [1,2,...]
```

### 3. **Sort Colors (Dutch Flag)**
```python
def sort_colors(nums):
    """Sort array of 0s, 1s, and 2s in-place."""
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
    
    return nums

# Time: O(n), Space: O(1)
```

## String Problems

### 1. **Reverse Words in String**
```python
def reverse_words(s):
    """Reverse words in string while preserving word order."""
    def reverse_range(arr, start, end):
        while start < end:
            arr[start], arr[end] = arr[end], arr[start]
            start += 1
            end -= 1
    
    # Convert to list for in-place operations
    chars = list(s.strip())
    
    # Reverse entire string
    reverse_range(chars, 0, len(chars) - 1)
    
    # Reverse each word
    start = 0
    for i in range(len(chars) + 1):
        if i == len(chars) or chars[i] == ' ':
            reverse_range(chars, start, i - 1)
            start = i + 1
    
    return ''.join(chars)

# Example: "the sky is blue" → "blue is sky the"
```

### 2. **Compare Version Numbers**
```python
def compare_version(version1, version2):
    """Compare two version numbers."""
    v1_parts = version1.split('.')
    v2_parts = version2.split('.')
    
    i, j = 0, 0
    
    while i < len(v1_parts) or j < len(v2_parts):
        num1 = int(v1_parts[i]) if i < len(v1_parts) else 0
        num2 = int(v2_parts[j]) if j < len(v2_parts) else 0
        
        if num1 < num2:
            return -1
        elif num1 > num2:
            return 1
        
        i += 1
        j += 1
    
    return 0

# Example: version1 = "1.01", version2 = "1.001" → 0
```

## Advanced Techniques

### 1. **Partition Array**
```python
def partition_array(nums, pivot):
    """Partition array around pivot value."""
    left, right = 0, len(nums) - 1
    
    while left <= right:
        while left <= right and nums[left] < pivot:
            left += 1
        while left <= right and nums[right] > pivot:
            right -= 1
        
        if left <= right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1
    
    return left  # Partition point

# Time: O(n), Space: O(1)
```

### 2. **Find Pair with Given Difference**
```python
def find_pair_difference(nums, k):
    """Find pair with difference k in sorted array."""
    left, right = 0, 1
    
    while right < len(nums):
        diff = nums[right] - nums[left]
        
        if diff == k:
            return [nums[left], nums[right]]
        elif diff < k:
            right += 1
        else:
            left += 1
            if left == right:
                right += 1
    
    return []

# Time: O(n), Space: O(1)
```

## Common Patterns and Tips

### 1. **When to Use Opposite Pointers**
- **Sorted arrays**: Target sum, closest sum problems
- **Palindromes**: String or array palindrome checking
- **Optimization**: Container, trapping water problems
- **Partitioning**: Dutch flag, quicksort partition

### 2. **Movement Strategies**
```python
# Equal movement
left += 1
right -= 1

# Conditional movement
if condition:
    left += 1
else:
    right -= 1

# Greedy movement (container problem)
if height[left] < height[right]:
    left += 1
else:
    right -= 1
```

### 3. **Common Mistakes**
```python
# Wrong: May cause infinite loop
while left <= right:  # Should be left < right for most cases

# Wrong: Not handling duplicates
# Should skip duplicates in problems like 3Sum

# Wrong: Incorrect boundary conditions
left, right = 0, len(arr)  # Should be len(arr) - 1
```

## Practice Problems

### Easy
1. Two Sum II - Input Array Is Sorted (LeetCode 167)
2. Valid Palindrome (LeetCode 125)
3. Reverse String (LeetCode 344)
4. Remove Duplicates from Sorted Array (LeetCode 26)

### Medium
1. 3Sum (LeetCode 15)
2. Container With Most Water (LeetCode 11)
3. Sort Colors (LeetCode 75)
4. 3Sum Closest (LeetCode 16)
5. Valid Palindrome II (LeetCode 680)

### Hard
1. Trapping Rain Water (LeetCode 42)
2. 4Sum (LeetCode 18)
3. Palindromic Substrings (LeetCode 647)

## Next Topics

- [[Same_Direction_Pointers]] - Learn about fast and slow pointer techniques
- [[Two_Pointers_Problems]] - Practice problems using two pointers
- [[Sliding_Window_Overview]] - Related technique for subarray problems
