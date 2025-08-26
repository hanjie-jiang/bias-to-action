# Sorting Algorithm Problems

Practice problems that utilize various sorting algorithms and sorting-based techniques. Problems are organized by difficulty and approach.

## Basic Sorting Problems

### Easy Problems

#### 1. Sort Array (LeetCode 912)
```python
def sort_array(nums):
    """Sort array using different algorithms."""
    
    # Merge Sort Implementation
    def merge_sort(arr):
        if len(arr) <= 1:
            return arr
        
        mid = len(arr) // 2
        left = merge_sort(arr[:mid])
        right = merge_sort(arr[mid:])
        
        return merge(left, right)
    
    def merge(left, right):
        result = []
        i = j = 0
        
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        
        result.extend(left[i:])
        result.extend(right[j:])
        return result
    
    return merge_sort(nums)

# Time: O(n log n), Space: O(n)
```

#### 2. Merge Sorted Array (LeetCode 88)
```python
def merge_sorted_arrays(nums1, m, nums2, n):
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

#### 3. Squares of Sorted Array (LeetCode 977)
```python
def sorted_squares(nums):
    """Return sorted squares of sorted array."""
    # Two pointers approach
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
    """Sort array of 0s, 1s, and 2s in-place."""
    # Dutch National Flag Algorithm
    left, current, right = 0, 0, len(nums) - 1
    
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

#### 2. Kth Largest Element (LeetCode 215)
```python
def find_kth_largest(nums, k):
    """Find kth largest element using quickselect."""
    import random
    
    def quickselect(left, right, k_smallest):
        if left == right:
            return nums[left]
        
        # Random pivot for better average performance
        pivot_index = random.randint(left, right)
        pivot_index = partition(left, right, pivot_index)
        
        if k_smallest == pivot_index:
            return nums[k_smallest]
        elif k_smallest < pivot_index:
            return quickselect(left, pivot_index - 1, k_smallest)
        else:
            return quickselect(pivot_index + 1, right, k_smallest)
    
    def partition(left, right, pivot_index):
        pivot = nums[pivot_index]
        # Move pivot to end
        nums[pivot_index], nums[right] = nums[right], nums[pivot_index]
        
        store_index = left
        for i in range(left, right):
            if nums[i] < pivot:
                nums[store_index], nums[i] = nums[i], nums[store_index]
                store_index += 1
        
        # Move pivot to final position
        nums[right], nums[store_index] = nums[store_index], nums[right]
        return store_index
    
    return quickselect(0, len(nums) - 1, len(nums) - k)

# Average: O(n), Worst: O(n²), Space: O(1)
```

#### 3. Merge Intervals (LeetCode 56)
```python
def merge_intervals(intervals):
    """Merge overlapping intervals."""
    if not intervals:
        return []
    
    # Sort by start time
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        last = merged[-1]
        
        if current[0] <= last[1]:  # Overlapping
            merged[-1] = [last[0], max(last[1], current[1])]
        else:  # Non-overlapping
            merged.append(current)
    
    return merged

# Time: O(n log n), Space: O(1) excluding result
```

#### 4. Top K Frequent Elements (LeetCode 347)
```python
def top_k_frequent(nums, k):
    """Find k most frequent elements."""
    from collections import Counter
    import heapq
    
    # Count frequencies
    count = Counter(nums)
    
    # Use heap to find top k
    return heapq.nlargest(k, count.keys(), key=count.get)

# Alternative: Bucket sort approach
def top_k_frequent_bucket(nums, k):
    from collections import Counter
    
    count = Counter(nums)
    # Bucket sort by frequency
    buckets = [[] for _ in range(len(nums) + 1)]
    
    for num, freq in count.items():
        buckets[freq].append(num)
    
    result = []
    for i in range(len(buckets) - 1, 0, -1):
        for num in buckets[i]:
            result.append(num)
            if len(result) == k:
                return result
    
    return result

# Heap: O(n log k), Bucket: O(n)
```

## Advanced Sorting Problems

### Hard Problems

#### 1. Count of Smaller Numbers After Self (LeetCode 315)
```python
def count_smaller(nums):
    """Count smaller numbers after each element."""
    def merge_sort_count(arr):
        if len(arr) <= 1:
            return arr
        
        mid = len(arr) // 2
        left = merge_sort_count(arr[:mid])
        right = merge_sort_count(arr[mid:])
        
        return merge_count(left, right)
    
    def merge_count(left, right):
        result = []
        i = j = 0
        
        while i < len(left) and j < len(right):
            if left[i][0] > right[j][0]:
                result.append(right[j])
                j += 1
            else:
                # All elements in right[j:] are larger
                counts[left[i][1]] += len(right) - j
                result.append(left[i])
                i += 1
        
        while i < len(left):
            result.append(left[i])
            i += 1
        
        while j < len(right):
            result.append(right[j])
            j += 1
        
        return result
    
    # Create (value, original_index) pairs
    indexed_nums = [(nums[i], i) for i in range(len(nums))]
    counts = [0] * len(nums)
    
    merge_sort_count(indexed_nums)
    return counts

# Time: O(n log n), Space: O(n)
```

#### 2. Reverse Pairs (LeetCode 493)
```python
def reverse_pairs(nums):
    """Count reverse pairs where i < j and nums[i] > 2 * nums[j]."""
    def merge_sort_count(left, right):
        if left >= right:
            return 0
        
        mid = (left + right) // 2
        count = merge_sort_count(left, mid) + merge_sort_count(mid + 1, right)
        
        # Count reverse pairs
        j = mid + 1
        for i in range(left, mid + 1):
            while j <= right and nums[i] > 2 * nums[j]:
                j += 1
            count += j - (mid + 1)
        
        # Merge
        temp = []
        i, j = left, mid + 1
        
        while i <= mid and j <= right:
            if nums[i] <= nums[j]:
                temp.append(nums[i])
                i += 1
            else:
                temp.append(nums[j])
                j += 1
        
        while i <= mid:
            temp.append(nums[i])
            i += 1
        
        while j <= right:
            temp.append(nums[j])
            j += 1
        
        # Copy back
        for i in range(len(temp)):
            nums[left + i] = temp[i]
        
        return count
    
    return merge_sort_count(0, len(nums) - 1)

# Time: O(n log n), Space: O(n)
```

## Sorting-Based Techniques

### 1. Meeting Rooms Problems

#### Meeting Rooms (LeetCode 252)
```python
def can_attend_meetings(intervals):
    """Check if person can attend all meetings."""
    intervals.sort(key=lambda x: x[0])
    
    for i in range(1, len(intervals)):
        if intervals[i][0] < intervals[i-1][1]:
            return False
    
    return True

# Time: O(n log n), Space: O(1)
```

#### Meeting Rooms II (LeetCode 253)
```python
def min_meeting_rooms(intervals):
    """Find minimum meeting rooms needed."""
    if not intervals:
        return 0
    
    import heapq
    
    intervals.sort(key=lambda x: x[0])
    heap = []  # Min heap of end times
    
    for interval in intervals:
        # Remove meetings that have ended
        while heap and heap[0] <= interval[0]:
            heapq.heappop(heap)
        
        # Add current meeting's end time
        heapq.heappush(heap, interval[1])
    
    return len(heap)

# Time: O(n log n), Space: O(n)
```

### 2. Array Transformation Problems

#### Wiggle Sort (LeetCode 280)
```python
def wiggle_sort(nums):
    """Reorder array so nums[0] < nums[1] > nums[2] < nums[3]..."""
    for i in range(len(nums) - 1):
        if (i % 2 == 0 and nums[i] > nums[i + 1]) or \
           (i % 2 == 1 and nums[i] < nums[i + 1]):
            nums[i], nums[i + 1] = nums[i + 1], nums[i]

# Time: O(n), Space: O(1)
```

#### Wiggle Sort II (LeetCode 324)
```python
def wiggle_sort_ii(nums):
    """Wiggle sort with strict inequality."""
    nums.sort()
    n = len(nums)
    
    # Split into two halves
    small = nums[:(n + 1) // 2]
    large = nums[(n + 1) // 2:]
    
    # Reverse to avoid adjacent equal elements
    small.reverse()
    large.reverse()
    
    # Interleave
    for i in range(n):
        if i % 2 == 0:
            nums[i] = small[i // 2]
        else:
            nums[i] = large[i // 2]

# Time: O(n log n), Space: O(n)
```

## Custom Sorting Problems

#### 1. Largest Number (LeetCode 179)
```python
def largest_number(nums):
    """Arrange numbers to form largest possible number."""
    from functools import cmp_to_key
    
    def compare(x, y):
        # Compare x+y vs y+x
        if x + y > y + x:
            return -1
        elif x + y < y + x:
            return 1
        else:
            return 0
    
    # Convert to strings
    str_nums = [str(num) for num in nums]
    
    # Sort with custom comparator
    str_nums.sort(key=cmp_to_key(compare))
    
    # Handle edge case of all zeros
    result = ''.join(str_nums)
    return '0' if result[0] == '0' else result

# Time: O(n log n), Space: O(n)
```

#### 2. Custom Sort String (LeetCode 791)
```python
def custom_sort_string(order, s):
    """Sort string s according to order."""
    from collections import Counter
    
    count = Counter(s)
    result = []
    
    # Add characters in order
    for char in order:
        if char in count:
            result.extend([char] * count[char])
            del count[char]
    
    # Add remaining characters
    for char, freq in count.items():
        result.extend([char] * freq)
    
    return ''.join(result)

# Time: O(n), Space: O(n)
```

## Problem-Solving Strategies

### 1. **Choose the Right Algorithm**
- **Small array (< 50)**: Insertion sort
- **Nearly sorted**: Insertion sort, bubble sort
- **Guaranteed O(n log n)**: Merge sort, heap sort
- **Average case optimization**: Quick sort
- **Stable sorting needed**: Merge sort, insertion sort
- **Memory constrained**: Heap sort, quick sort

### 2. **Sorting Decision Tree**
```
What's the constraint?
├── Time critical & large data → Quick sort
├── Stability required → Merge sort
├── Memory limited → Heap sort
├── Small data → Insertion sort
└── Special properties → Custom algorithm
```

### 3. **Common Patterns**
- **Interval problems**: Sort by start time
- **Meeting rooms**: Sort + greedy or heap
- **Kth element**: Quickselect or heap
- **Custom order**: Custom comparator
- **Counting inversions**: Merge sort

## Practice Schedule

### Week 1: Basic Sorting
1. Sort Array (multiple algorithms)
2. Merge Sorted Array
3. Squares of Sorted Array

### Week 2: Sorting Applications
1. Sort Colors
2. Kth Largest Element
3. Merge Intervals
4. Top K Frequent Elements

### Week 3: Advanced Problems
1. Meeting Rooms II
2. Largest Number
3. Count of Smaller Numbers After Self

## Next Topics

- [[Binary_Search_Fundamentals]] - Use sorting to enable binary search
- [[Two_Pointers_Overview]] - Techniques that work well with sorted data
- [[Sliding_Window_Overview]] - Another technique for array problems
