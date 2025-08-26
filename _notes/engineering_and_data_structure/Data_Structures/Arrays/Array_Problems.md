# Array Problems

Practice problems that focus on array data structure operations, manipulations, and algorithms.

## Basic Array Operations

### Easy Problems

#### 1. Two Sum (LeetCode 1)
```python
def two_sum(nums, target):
    """Find indices of two numbers that add up to target."""
    num_to_index = {}
    
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_to_index:
            return [num_to_index[complement], i]
        num_to_index[num] = i
    
    return []

# Time: O(n), Space: O(n)
```

#### 2. Best Time to Buy and Sell Stock (LeetCode 121)
```python
def max_profit(prices):
    """Find maximum profit from one buy and one sell."""
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

#### 3. Contains Duplicate (LeetCode 217)
```python
def contains_duplicate(nums):
    """Check if array contains duplicates."""
    return len(nums) != len(set(nums))

# Alternative: Using hash set
def contains_duplicate_v2(nums):
    seen = set()
    for num in nums:
        if num in seen:
            return True
        seen.add(num)
    return False

# Time: O(n), Space: O(n)
```

#### 4. Maximum Subarray (LeetCode 53)
```python
def max_subarray(nums):
    """Find maximum sum of contiguous subarray (Kadane's Algorithm)."""
    max_sum = current_sum = nums[0]
    
    for num in nums[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    
    return max_sum

# Time: O(n), Space: O(1)
```

### Medium Problems

#### 1. Product of Array Except Self (LeetCode 238)
```python
def product_except_self(nums):
    """Return array where output[i] = product of all elements except nums[i]."""
    n = len(nums)
    result = [1] * n
    
    # Forward pass: result[i] = product of all elements before i
    for i in range(1, n):
        result[i] = result[i - 1] * nums[i - 1]
    
    # Backward pass: multiply by product of all elements after i
    right_product = 1
    for i in range(n - 1, -1, -1):
        result[i] *= right_product
        right_product *= nums[i]
    
    return result

# Time: O(n), Space: O(1) excluding output array
```

#### 2. 3Sum (LeetCode 15)
```python
def three_sum(nums):
    """Find all unique triplets that sum to zero."""
    nums.sort()
    result = []
    
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue  # Skip duplicates
        
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

#### 3. Container With Most Water (LeetCode 11)
```python
def max_area(height):
    """Find two lines that form container with most water."""
    left, right = 0, len(height) - 1
    max_water = 0
    
    while left < right:
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

## Array Manipulation

### Easy Problems

#### 1. Rotate Array (LeetCode 189)
```python
def rotate(nums, k):
    """Rotate array to the right by k steps."""
    n = len(nums)
    k = k % n  # Handle k > n
    
    # Reverse entire array
    nums.reverse()
    
    # Reverse first k elements
    nums[:k] = reversed(nums[:k])
    
    # Reverse remaining elements
    nums[k:] = reversed(nums[k:])

# Alternative: Using extra space
def rotate_v2(nums, k):
    n = len(nums)
    k = k % n
    nums[:] = nums[-k:] + nums[:-k]

# Time: O(n), Space: O(1)
```

#### 2. Remove Duplicates from Sorted Array (LeetCode 26)
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

#### 3. Move Zeroes (LeetCode 283)
```python
def move_zeroes(nums):
    """Move all zeros to end while maintaining order of non-zeros."""
    write_index = 0
    
    # Move all non-zeros to front
    for read_index in range(len(nums)):
        if nums[read_index] != 0:
            nums[write_index] = nums[read_index]
            write_index += 1
    
    # Fill remaining positions with zeros
    while write_index < len(nums):
        nums[write_index] = 0
        write_index += 1

# Time: O(n), Space: O(1)
```

### Medium Problems

#### 1. Set Matrix Zeroes (LeetCode 73)
```python
def set_zeroes(matrix):
    """Set entire row and column to zero if element is zero."""
    m, n = len(matrix), len(matrix[0])
    first_row_zero = any(matrix[0][j] == 0 for j in range(n))
    first_col_zero = any(matrix[i][0] == 0 for i in range(m))
    
    # Use first row and column as markers
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][j] == 0:
                matrix[i][0] = 0
                matrix[0][j] = 0
    
    # Set zeros based on markers
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][0] == 0 or matrix[0][j] == 0:
                matrix[i][j] = 0
    
    # Handle first row and column
    if first_row_zero:
        for j in range(n):
            matrix[0][j] = 0
    
    if first_col_zero:
        for i in range(m):
            matrix[i][0] = 0

# Time: O(m*n), Space: O(1)
```

#### 2. Spiral Matrix (LeetCode 54)
```python
def spiral_order(matrix):
    """Return elements of matrix in spiral order."""
    if not matrix or not matrix[0]:
        return []
    
    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    
    while top <= bottom and left <= right:
        # Traverse right
        for col in range(left, right + 1):
            result.append(matrix[top][col])
        top += 1
        
        # Traverse down
        for row in range(top, bottom + 1):
            result.append(matrix[row][right])
        right -= 1
        
        # Traverse left (if we still have rows)
        if top <= bottom:
            for col in range(right, left - 1, -1):
                result.append(matrix[bottom][col])
            bottom -= 1
        
        # Traverse up (if we still have columns)
        if left <= right:
            for row in range(bottom, top - 1, -1):
                result.append(matrix[row][left])
            left += 1
    
    return result

# Time: O(m*n), Space: O(1) excluding result
```

## Searching in Arrays

### Easy Problems

#### 1. Binary Search (LeetCode 704)
```python
def search(nums, target):
    """Binary search in sorted array."""
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

# Time: O(log n), Space: O(1)
```

#### 2. Search Insert Position (LeetCode 35)
```python
def search_insert(nums, target):
    """Find position to insert target in sorted array."""
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return left

# Time: O(log n), Space: O(1)
```

### Medium Problems

#### 1. Search in Rotated Sorted Array (LeetCode 33)
```python
def search(nums, target):
    """Search target in rotated sorted array."""
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        
        # Check which half is sorted
        if nums[left] <= nums[mid]:  # Left half is sorted
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:  # Right half is sorted
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return -1

# Time: O(log n), Space: O(1)
```

#### 2. Find Peak Element (LeetCode 162)
```python
def find_peak_element(nums):
    """Find any peak element in the array."""
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = left + (right - left) // 2
        
        if nums[mid] > nums[mid + 1]:
            right = mid  # Peak is in left half (including mid)
        else:
            left = mid + 1  # Peak is in right half
    
    return left

# Time: O(log n), Space: O(1)
```

## Sorting Problems

### Easy Problems

#### 1. Merge Sorted Array (LeetCode 88)
```python
def merge(nums1, m, nums2, n):
    """Merge two sorted arrays in-place."""
    # Start from the end to avoid overwriting
    i, j, k = m - 1, n - 1, m + n - 1
    
    while i >= 0 and j >= 0:
        if nums1[i] > nums2[j]:
            nums1[k] = nums1[i]
            i -= 1
        else:
            nums1[k] = nums2[j]
            j -= 1
        k -= 1
    
    # Copy remaining elements from nums2
    while j >= 0:
        nums1[k] = nums2[j]
        j -= 1
        k -= 1

# Time: O(m + n), Space: O(1)
```

#### 2. Squares of Sorted Array (LeetCode 977)
```python
def sorted_squares(nums):
    """Return sorted squares of sorted array."""
    left, right = 0, len(nums) - 1
    result = [0] * len(nums)
    pos = len(nums) - 1
    
    while left <= right:
        left_square = nums[left] ** 2
        right_square = nums[right] ** 2
        
        if left_square > right_square:
            result[pos] = left_square
            left += 1
        else:
            result[pos] = right_square
            right -= 1
        
        pos -= 1
    
    return result

# Time: O(n), Space: O(n)
```

### Medium Problems

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

## Advanced Array Problems

### Medium Problems

#### 1. Next Permutation (LeetCode 31)
```python
def next_permutation(nums):
    """Find next lexicographically greater permutation."""
    # Find the largest index i such that nums[i] < nums[i + 1]
    i = len(nums) - 2
    while i >= 0 and nums[i] >= nums[i + 1]:
        i -= 1
    
    if i >= 0:
        # Find the largest index j such that nums[i] < nums[j]
        j = len(nums) - 1
        while nums[j] <= nums[i]:
            j -= 1
        
        # Swap nums[i] and nums[j]
        nums[i], nums[j] = nums[j], nums[i]
    
    # Reverse the suffix starting at nums[i + 1]
    nums[i + 1:] = reversed(nums[i + 1:])

# Time: O(n), Space: O(1)
```

#### 2. Jump Game (LeetCode 55)
```python
def can_jump(nums):
    """Check if you can reach the last index."""
    max_reach = 0
    
    for i in range(len(nums)):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + nums[i])
        if max_reach >= len(nums) - 1:
            return True
    
    return True

# Time: O(n), Space: O(1)
```

#### 3. Subarray Sum Equals K (LeetCode 560)
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

# Time: O(n), Space: O(n)
```

### Hard Problems

#### 1. Trapping Rain Water (LeetCode 42)
```python
def trap(height):
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

# Time: O(n), Space: O(1)
```

#### 2. First Missing Positive (LeetCode 41)
```python
def first_missing_positive(nums):
    """Find the smallest missing positive integer."""
    n = len(nums)
    
    # Mark presence of numbers
    for i in range(n):
        while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            # Place nums[i] at its correct position
            nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]
    
    # Find first missing positive
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
    
    return n + 1

# Time: O(n), Space: O(1)
```

## Problem-Solving Strategies

### 1. **Array Traversal Patterns**
```python
# Single pass
for i in range(len(arr)):
    process(arr[i])

# Two pointers
left, right = 0, len(arr) - 1
while left < right:
    process(arr[left], arr[right])

# Sliding window
left = 0
for right in range(len(arr)):
    while condition_violated:
        left += 1
    process_window(left, right)
```

### 2. **Space Optimization Techniques**
```python
# In-place modification
def modify_in_place(arr):
    write_index = 0
    for read_index in range(len(arr)):
        if should_keep(arr[read_index]):
            arr[write_index] = arr[read_index]
            write_index += 1
    return write_index

# Using array indices as hash
def use_indices_as_hash(nums):
    for i in range(len(nums)):
        index = abs(nums[i]) - 1
        if index < len(nums):
            nums[index] = -abs(nums[index])
```

### 3. **Common Array Algorithms**
- **Kadane's Algorithm**: Maximum subarray sum
- **Dutch Flag Algorithm**: Partition array into three parts
- **Binary Search**: Search in sorted array
- **Two Pointers**: Various problems on sorted arrays
- **Sliding Window**: Subarray problems with conditions

## Practice Schedule

### Week 1: Basic Operations
1. Two Sum
2. Best Time to Buy and Sell Stock
3. Maximum Subarray
4. Contains Duplicate

### Week 2: Array Manipulation
1. Rotate Array
2. Move Zeroes
3. Product of Array Except Self
4. Set Matrix Zeroes

### Week 3: Advanced Problems
1. 3Sum
2. Container With Most Water
3. Trapping Rain Water
4. Next Permutation

## Next Topics

- [[Binary_Search_Fundamentals]] - Searching in sorted arrays
- [[Two_Pointers_Overview]] - Two pointer techniques for arrays
- [[Sliding_Window_Overview]] - Subarray problems with conditions
