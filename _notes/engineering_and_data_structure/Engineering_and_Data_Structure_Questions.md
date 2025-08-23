---
title: Engineering_and_Data_Structure_Questions
---
## Dictionary & Sets
### Python Sets
 A set in Python is an unordered collection of unique objects, ensuring the absence of duplicate values. Furthermore, it allows us to perform several operations on such collections, **such as intersection (identifying common elements), union (combining all unique elements), and difference (detecting unique items in a set).**
#### array intersection

```
def array_intersection(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = set1 & set2
    return sorted(list(intersection))
```
 
 The set operations run at a time complexity of O(n), but the sorting step has a time complexity of O(nlog⁡n). Therefore, the overall time complexity of the solution is O(nlog⁡n), dominated by the sorting step.
#### non-repeating elements

```
seen, repeated = set(), set()
for num in nums:
    if num in seen:
        repeated.add(num)
    else: 
        seen.add(num)
return list(seen - repeated)
```

This approach results again in a time complexity of O(n) and a memory complexity of O(n) due to the constant time operations provided by the Python `set`.

#### unique elements

```
def unique_elements(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    unique_to_1 = sorted(list(set1 - set2))
    unique_to_2 = sorted(list(set2 - set1))
    return (unique_to_1, unique_to_2)
```

This solution is considerably more efficient than the naive approach, operating at a time complexity of O(n), or O(max⁡(len(list1),len(list2))) to be more precise.
#### unique string in list
##### using 2 sets 

```
def find_unique_string(words):
    seen = set()
    duplicates = set()
    for word in words:
        if word in seen:
            duplicates.add(word)
        seen.add(word)
    for word in words:
	    if word not in duplicates:
	        return word
return ""
```

##### using dictionary

```
def find_unique_string(words):
    count_dict = {}
    for word in words:
        if word in count_dict:
            count_dict[word] = count_dict[word] + 1
        else:
            count_dict[word] = 1
    for word in words:
        if count_dict[word] == 1:
            return word
    return ""
```

#### Anagram Pairs in Two Lists (`medium`)

##### using tuples & sets

```
def anagram(list_1, list_2):
	# convert every word from both lists to a sorted tuple of its characters to have a unified form for all anagram words
	sorted_tuples_1 = set(tuple(sorted(word)) for word in list_1)
	sorted_tuples_2 = set(tuple(sorted(word)) for word in list_2)
	
	#  find the common tuples between the two
	common_tuples = sorted_tuples_1 & sorted_tuples_2
	
	# iterate over the words in the original lists to filter for the words that are anagram
	list_1_output = [word for word in list_1 if tuple(sorted(word)) in common_tuples] # contains anagrams from the first list
	list_2_output = [word for word in list_2 if tuple(sorted(word)) in common_tuples] # contains anagrams from the second list
	
	# check for the words pairs in the filtered list
	output = []
	for word1 in list_1_output:
	    for word2 in list_2_output:
	        # traversing every pair of words in filtered lists
	        if tuple(sorted(word1)) == tuple(sorted(word2)):
	            # If words in the pair are anagrams, add them to the output list
	            output.append((word1, word2))
	return output
```

##### use of dictionary

```
from collections import defaultdict

# Create mapping for `list_1`
mapping_1 = defaultdict(list)
# mapping_1 stores (sorted anagram) -> list[anagrams] mapping for `list_1`
for word in list_1:
    sorted_tuple = tuple(sorted(word)) # unique identifier of the anagram
    mapping_1[sorted_tuple].append(word)
    # `mapping_1[sorted_tuple]` stores all anagrams under the same identifier for `list_1`

# Create mapping for `list_2`
mapping_2 = defaultdict(list)
# mapping_2 stores (sorted anagram) -> list[anagrams] mapping for `list_2`
for word in list_2:
    sorted_tuple = tuple(sorted(word)) # unique identifier of the anagram
    mapping_2[sorted_tuple].append(word)
    # `mapping_2[sorted_tuple]` stores all anagrams under the same identifier for `list_2`

# Intersect keys from mapping_1 and mapping_2 to get common sorted tuples
# Every element in `common_tuples` is an anagram identifier that exists in both lists
common_tuples = set(mapping_1.keys()) & set(mapping_2.keys())

output = []
for anagram_tuple in common_tuples:
    for word1 in mapping_1[anagram_tuple]:
        for word2 in mapping_2[anagram_tuple]:
            # Both word1 and word2 have the same anagram identifier, so are anagrams
            output.append((word1, word2))

return output
```
### Exercises

#### CodeSignal Mastering Python Sets: Intersection, Non-Repeating Elements, and Unique Elements

##### How to ignoring capital vs lower cases in python

```
    inventory1 = [string.upper() for string in inventory1] # .lower() for lower 
    inventory2 = [string.upper() for string in inventory2] # .lower() for lower 
```
#### Leetcode 1056: Confusing Number

##### How to initiate a set and how to add elements to set

```
set_sample = set()
set_sample.add(1)
```
##### How to convert a string of digits in dictionary to a number 

```
int("".join([value for _, value in dictionary.items()]))
```
##### How to sort a dictionary in python
###### based on key

```
my_dict = {'apple': 3, 'orange': 1, 'banana': 2}  
sorted_by_key = dict(sorted(my_dict.items()))  
print(sorted_by_key)  
# Output: {'apple': 3, 'banana': 2, 'orange': 1}
```
###### based on value

```
my_dict = {'apple': 3, 'orange': 1, 'banana': 2}  
sorted_by_value = dict(sorted(my_dict.items(), key=lambda item: item[1]))  
print(sorted_by_value)  
# Output: {'orange': 1, 'banana': 2, 'apple': 3}
```


