# Search Algorithm Problems

This section contains practice problems that use various search algorithms. Problems are organized by difficulty and algorithm type.

## Toy Problem for Algorithmic Thinking - Peak Finder Problem

### One-dimensional List Problem

Given an array arr[] where no two adjacent elements are same, find the index of a peak element. An element is considered to be a peak element if it is strictly greater than its adjacent elements. If there are multiple peak elements, return the index of any one of them.

Note: Consider the element before the first element and the element after the last element to be negative infinity.

#### Naive Linear Search Solution

```python
def peakElement(arr):
    n = len(arr)

    for i in range(n):
        left = True
        right = True

        # Check for element to the left
        if i > 0 and arr[i] <= arr[i - 1]:
            left = False

        # Check for element to the right
        if i < n - 1 and arr[i] <= arr[i + 1]:
            right = False

        # If arr[i] is greater than its left as well as
        # its right element, return its index
        if left and right:
            return i

if __name__ == "__main__":
	arr = [1, 2, 4, 5, 7, 8, 3]
	print(peakElement(arr))
```

#### Binary Search Approach and Motivation Explained

The key insight for the binary search approach is that we can eliminate half of the search space at each step by comparing the middle element with its neighbor.

**General Idea**
- start in the middle:  the running time $\Theta(1)$
- look at neighbor and see if local peak: the running time $\Theta(1)$
- if not figure out which way to go: the running time $\Theta(N/2)$

The overall run time for this problem is: $T(N) = \Theta(1) + T(N/2) = c + T(N/2) = nc + T(N/2^c)$. Due to that we know $T(1) = \Theta(1)$, we could then make $nc + T(N/2^c) = T(1)$, i.e. $N/2^c = 1$, therefore, $i = \log(n)$. Plugging back in, $nc +  T(N/2^c) = c \cdot \log(N) + \Theta(1) = \Theta(1) \cdot \log(N) + \Theta(1) = \Theta(\log(N))$.

![Peak Finder Binary Search](/ml-learning-notes/assets/images/peak_finder_binary.png)

**Binary Search Implementation:**
 
```python
def peakElement_binary_search(arr):
    """
    Find a peak element using binary search approach.
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    left, right = 0, len(arr) - 1
    
    while left < right:
        mid = left + (right - left) // 2
        
        # If mid element is greater than its right neighbor,
        # then peak lies on the left side (including mid)
        if arr[mid] > arr[mid + 1]:
            right = mid
        else:
            # Peak lies on the right side
            left = mid + 1
    
    return left

# Example usage
if __name__ == "__main__":
    arr = [1, 2, 4, 5, 7, 8, 3]
    print(f"Peak element index (binary search): {peakElement_binary_search(arr)}")
    print(f"Peak element value: {arr[peakElement_binary_search(arr)]}")
```

### 2D Peak Finder Problem

The 2D peak finder extends the concept to a 2D matrix where a peak element is greater than or equal to its four neighbors (up, down, left, right).

![2D Peak Finder](/ml-learning-notes/assets/images/peak_finder_2d.png)

**Problem Statement:** Given a 2D matrix where no two adjacent elements are the same, find any peak element. An element is a peak if it's greater than or equal to all its neighbors.

Borrowing the idea from one-dimensional list problem solution, we could find some ideas, but not all of them are working: 

#### Solution 1: Efficient Algorithm but does not work
First pick the middle column `j = m/2` and find the column peak at position `(i,j)`, and then use `(i,j)` as a start to find the one-dimensional peak at row `i`. 

But when the column / row does not have a peak, it would not return anything, i.e. a 2-D peak may not exist on row `i`.

#### Solution 2: 
Pick middle column `j = m/2` and find the global maximum on column `j` at `(i,j)` (1. via the 1-D peak finder ($\Theta(\log(M))$) or 2. find max of column ($\Theta(M)$)) and compare the `(i,j-1)`, `(i,j)` and `(i,j+1)`. 

From here, pick the left columns if `(i,j-1)` > `(i,j)` and similarly for the right. If `(i,j)` is both larger than its adjacent neighbors, then it is the 2-D peak. Solve the new problem with half of the columns / rows. When you have a single column, find the global maximum and the peak is found.

Run time for this solution 2:
base case:

$$T(1, N) = \Theta(N)$$

recursively:

$$T(M, N) = \Theta(1) + \Theta(N) + T(M/2, N) = \Theta(N) + T(M/2, N) = \Theta(M) \cdot \Theta(\log(N)) = \Theta(M\log(N))$$

**Algorithmic Approaches:**

#### Naive O(nm) Solution:
```python
def find_2d_peak_naive(matrix):
    """
    Naive approach: Check every element
    Time Complexity: O(nm) where n,m are dimensions
    """
    rows, cols = len(matrix), len(matrix[0])
    
    for i in range(rows):
        for j in range(cols):
            is_peak = True
            current = matrix[i][j]
            
            # Check all four neighbors
            directions = [(-1,0), (1,0), (0,-1), (0,1)]
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    if matrix[ni][nj] > current:
                        is_peak = False
                        break
            
            if is_peak:
                return (i, j)
    
    return None  # No peak found
```

#### Optimized O(n log m) Solution:
```python
def find_2d_peak_optimized(matrix):
    """
    Divide and conquer approach on columns
    Time Complexity: O(n log m) where n=rows, m=cols
    """
    def find_max_in_column(matrix, col):
        max_val = matrix[0][col]
        max_row = 0
        for i in range(1, len(matrix)):
            if matrix[i][col] > max_val:
                max_val = matrix[i][col]
                max_row = i
        return max_row
    
    def find_peak_recursive(matrix, start_col, end_col):
        if start_col == end_col:
            max_row = find_max_in_column(matrix, start_col)
            return (max_row, start_col)
        
        mid_col = (start_col + end_col) // 2
        max_row = find_max_in_column(matrix, mid_col)
        
        # Check left neighbor
        left_val = matrix[max_row][mid_col - 1] if mid_col > 0 else -1
        # Check right neighbor  
        right_val = matrix[max_row][mid_col + 1] if mid_col < len(matrix[0]) - 1 else -1
        
        current_val = matrix[max_row][mid_col]
        
        if current_val >= left_val and current_val >= right_val:
            return (max_row, mid_col)  # Found peak
        elif left_val > current_val:
            return find_peak_recursive(matrix, start_col, mid_col - 1)
        else:
            return find_peak_recursive(matrix, mid_col + 1, end_col)
    
    if not matrix or not matrix[0]:
        return None
    
    return find_peak_recursive(matrix, 0, len(matrix[0]) - 1)

# Example usage
if __name__ == "__main__":
    matrix = [
        [10, 8, 10, 10],
        [14, 13, 12, 11],
        [15, 9, 11, 21],
        [16, 17, 19, 20]
    ]
    
    peak_naive = find_2d_peak_naive(matrix)
    peak_optimized = find_2d_peak_optimized(matrix)
    
    print(f"Naive approach found peak at: {peak_naive}")
    print(f"Optimized approach found peak at: {peak_optimized}")
    if peak_optimized:
        row, col = peak_optimized
        print(f"Peak value: {matrix[row][col]}")
```

**Key Insights:**
- **Divide and Conquer Strategy**: Focus on middle column, find max element
- **Recursive Elimination**: Compare with left/right neighbors to eliminate half the columns
- **Guaranteed Peak Existence**: Algorithm ensures a peak will always be found
- **Time Complexity**: O(n log m) - much better than naive O(nm) approach
- **Space Complexity**: O(log m) due to recursion stack

**Applications:**
- **Image Processing**: Finding local maxima in intensity matrices
- **Optimization Problems**: Peak finding in 2D objective functions  
- **Data Analysis**: Identifying outliers or significant points in 2D datasets

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
