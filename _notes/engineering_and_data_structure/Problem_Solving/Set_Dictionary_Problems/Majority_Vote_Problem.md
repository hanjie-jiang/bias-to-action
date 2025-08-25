---
title: Majority Vote Problem
---
Our first problem is about identifying the "majority" element in a list. The "majority element" in a list is an element that appears more than `n / 2` times. Given a list of integers, our aim is to identify the majority element.

```Python
def majority_vote(listA):
    count_dict = {}
    for element in listA:
        count_dict[element] = count_dict.get(element, 0) + 1
        if count_dict[element] > len(listA) // 2:
            return element
    return -1
```