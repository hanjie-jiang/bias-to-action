# Search Algorithm Problems

This section contains practice problems that use various search algorithms. Problems are organized by difficulty and algorithm type.

## Linear Search Problems

### Easy Problems

#### 1. Find Maximum Element
```python
def find_maximum(arr):
    """Find the maximum element in an unsorted array."""
    if not arr:
        return None
    
    max_element = arr[0]
    for element in arr[1:]:
        if element > max_element:
            max_element = element
    
    return max_element

# Time: O(n), Space: O(1)
```

#### 2. Count Occurrences
```python
def count_occurrences(arr, target):
    """Count how many times target appears in array."""
    count = 0
    for element in arr:
        if element == target:
            count += 1
    return count

# Time: O(n), Space: O(1)
```

#### 3. Find All Indices
```python
def find_all_indices(arr, target):
    """Find all indices where target appears."""
    indices = []
    for i, element in enumerate(arr):
        if element == target:
            indices.append(i)
    return indices

# Time: O(n), Space: O(k) where k is number of occurrences
```

## Binary Search Problems

### Easy Problems

#### 1. Search Insert Position (LeetCode 35)
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

# Example: nums = [1,3,5,6], target = 5 → return 2
# Example: nums = [1,3,5,6], target = 2 → return 1
```

#### 2. First Bad Version (LeetCode 278)
```python
def first_bad_version(n):
    """Find first bad version using binary search."""
    def is_bad_version(version):
        # This function is provided by the problem
        pass
    
    left, right = 1, n
    
    while left < right:
        mid = left + (right - left) // 2
        
        if is_bad_version(mid):
            right = mid  # First bad could be mid
        else:
            left = mid + 1  # First bad is after mid
    
    return left

# Time: O(log n), Space: O(1)
```

#### 3. Perfect Square (LeetCode 367)
```python
def is_perfect_square(num):
    """Check if number is perfect square using binary search."""
    if num < 1:
        return False
    
    left, right = 1, num
    
    while left <= right:
        mid = left + (right - left) // 2
        square = mid * mid
        
        if square == num:
            return True
        elif square < num:
            left = mid + 1
        else:
            right = mid - 1
    
    return False

# Time: O(log n), Space: O(1)
```

### Medium Problems

#### 1. Find First and Last Position (LeetCode 34)
```python
def search_range(nums, target):
    """Find first and last position of target in sorted array."""
    def find_boundary(is_first):
        left, right = 0, len(nums) - 1
        result = -1
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if nums[mid] == target:
                result = mid
                if is_first:
                    right = mid - 1  # Continue left for first
                else:
                    left = mid + 1   # Continue right for last
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return result
    
    first = find_boundary(True)
    if first == -1:
        return [-1, -1]
    
    last = find_boundary(False)
    return [first, last]

# Time: O(log n), Space: O(1)
```

#### 2. Search in Rotated Sorted Array (LeetCode 33)
```python
def search_rotated(nums, target):
    """Search target in rotated sorted array."""
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        
        # Determine which half is sorted
        if nums[left] <= nums[mid]:  # Left half sorted
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:  # Right half sorted
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return -1

# Time: O(log n), Space: O(1)
```

#### 3. Find Peak Element (LeetCode 162)
```python
def find_peak_element(nums):
    """Find any peak element in the array."""
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = left + (right - left) // 2
        
        if nums[mid] > nums[mid + 1]:
            right = mid  # Peak in left half (including mid)
        else:
            left = mid + 1  # Peak in right half
    
    return left

# Time: O(log n), Space: O(1)
```

#### 4. Find Minimum in Rotated Array (LeetCode 153)
```python
def find_min(nums):
    """Find minimum element in rotated sorted array."""
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = left + (right - left) // 2
        
        if nums[mid] > nums[right]:
            left = mid + 1  # Min in right half
        else:
            right = mid     # Min in left half (including mid)
    
    return nums[left]

# Time: O(log n), Space: O(1)
```

### Hard Problems

#### 1. Median of Two Sorted Arrays (LeetCode 4)
```python
def find_median_sorted_arrays(nums1, nums2):
    """Find median of two sorted arrays."""
    # Ensure nums1 is shorter
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    
    m, n = len(nums1), len(nums2)
    left, right = 0, m
    
    while left <= right:
        partition1 = (left + right) // 2
        partition2 = (m + n + 1) // 2 - partition1
        
        # Handle edge cases
        max_left1 = float('-inf') if partition1 == 0 else nums1[partition1 - 1]
        min_right1 = float('inf') if partition1 == m else nums1[partition1]
        
        max_left2 = float('-inf') if partition2 == 0 else nums2[partition2 - 1]
        min_right2 = float('inf') if partition2 == n else nums2[partition2]
        
        if max_left1 <= min_right2 and max_left2 <= min_right1:
            # Found correct partition
            if (m + n) % 2 == 0:
                return (max(max_left1, max_left2) + min(min_right1, min_right2)) / 2
            else:
                return max(max_left1, max_left2)
        elif max_left1 > min_right2:
            right = partition1 - 1
        else:
            left = partition1 + 1

# Time: O(log(min(m,n))), Space: O(1)
```

## Binary Search on Answer Space

#### 1. Capacity to Ship Packages (LeetCode 1011)
```python
def ship_within_days(weights, days):
    """Find minimum ship capacity to ship all packages within days."""
    def can_ship(capacity):
        current_weight = 0
        days_needed = 1
        
        for weight in weights:
            if current_weight + weight > capacity:
                days_needed += 1
                current_weight = weight
            else:
                current_weight += weight
        
        return days_needed <= days
    
    left, right = max(weights), sum(weights)
    
    while left < right:
        mid = left + (right - left) // 2
        
        if can_ship(mid):
            right = mid
        else:
            left = mid + 1
    
    return left

# Time: O(n * log(sum - max)), Space: O(1)
```

#### 2. Koko Eating Bananas (LeetCode 875)
```python
def min_eating_speed(piles, h):
    """Find minimum eating speed to finish all bananas in h hours."""
    import math
    
    def can_finish(speed):
        hours = 0
        for pile in piles:
            hours += math.ceil(pile / speed)
        return hours <= h
    
    left, right = 1, max(piles)
    
    while left < right:
        mid = left + (right - left) // 2
        
        if can_finish(mid):
            right = mid
        else:
            left = mid + 1
    
    return left

# Time: O(n * log(max_pile)), Space: O(1)
```

## Two Pointer Search Problems

#### 1. Two Sum II - Sorted Array (LeetCode 167)
```python
def two_sum_sorted(numbers, target):
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
    
    return []  # No solution found

# Time: O(n), Space: O(1)
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

## Problem-Solving Strategies

### 1. **Identify the Pattern**
- **Sorted array** → Binary Search
- **Find pair/triplet with sum** → Two Pointers
- **Search in range** → Binary Search on Answer
- **Unsorted array** → Linear Search or Hash Map

### 2. **Binary Search Decision Tree**
```
Is array sorted?
├── Yes
│   ├── Find exact element? → Basic Binary Search
│   ├── Find boundary? → Modified Binary Search
│   └── Find in rotated? → Rotated Array Search
└── No
    ├── Can sort? → Sort + Binary Search
    ├── Small array? → Linear Search
    └── Need fast lookup? → Hash Map
```

### 3. **Common Mistakes**
- Off-by-one errors in boundary conditions
- Infinite loops due to incorrect updates
- Not handling edge cases (empty arrays, single elements)
- Using wrong comparison operators

## Practice Recommendations

### Week 1: Basic Binary Search
1. Binary Search (LeetCode 704)
2. Search Insert Position (LeetCode 35)
3. First Bad Version (LeetCode 278)

### Week 2: Binary Search Variations
1. Find First and Last Position (LeetCode 34)
2. Search in Rotated Sorted Array (LeetCode 33)
3. Find Peak Element (LeetCode 162)

### Week 3: Advanced Applications
1. Median of Two Sorted Arrays (LeetCode 4)
2. Capacity to Ship Packages (LeetCode 1011)
3. Koko Eating Bananas (LeetCode 875)

## Next Topics

- [[Binary_Search_Fundamentals]] - Review binary search basics
- [[Two_Pointers_Overview]] - Learn two pointers technique
- [[Sorting_Algorithms_Overview]] - Understand sorting for search preparation
