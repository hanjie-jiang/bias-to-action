---
title: Engineering_and_Data_Structure_Questions
---
## Dictionary & Sets
### Leetcode 1056: Confusing Number
#### How to convert a string of digits in dictionary to a number 

```
int("".join([value for _, value in dictionary.items()]))
```
#### How to sort a dictionary in python
##### based on key

```
my_dict = {'apple': 3, 'orange': 1, 'banana': 2}  
sorted_by_key = dict(sorted(my_dict.items()))  
print(sorted_by_key)  
# Output: {'apple': 3, 'banana': 2, 'orange': 1}
```
##### based on value

```
my_dict = {'apple': 3, 'orange': 1, 'banana': 2}  
sorted_by_value = dict(sorted(my_dict.items(), key=lambda item: item[1]))  
print(sorted_by_value)  
# Output: {'orange': 1, 'banana': 2, 'apple': 3}
```
