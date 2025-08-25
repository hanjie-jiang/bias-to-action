---
title: Anagram Pairs
---
# Anagram Pairs

Find all anagram pairs between two lists using Python sets and dictionaries.

## Problem Description

Given two lists of strings, find all pairs of words that are anagrams of each other (words that can be formed by rearranging the letters of another word).

## Solution Using Tuples and Sets

```python
def anagram(list_1, list_2):
    # convert every word from both lists to a sorted tuple of its characters to have a unified form for all anagram words
    sorted_tuples_1 = set(tuple(sorted(word)) for word in list_1)
    sorted_tuples_2 = set(tuple(sorted(word)) for word in list_2)
    
    # find the common tuples between the two
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

## Solution Using Dictionary

```python
from collections import defaultdict

def anagram_dict(list_1, list_2):
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

## How It Works

### Set Approach
1. **Create Sorted Tuples**: Convert each word to a sorted tuple of characters
2. **Find Common Anagrams**: Use set intersection to find common anagram identifiers
3. **Filter Words**: Get words from both lists that match the common anagrams
4. **Generate Pairs**: Create all possible pairs between the filtered words

### Dictionary Approach
1. **Create Mappings**: Build dictionaries mapping sorted tuples to lists of anagrams
2. **Find Common Keys**: Use set intersection to find common anagram identifiers
3. **Generate Pairs**: Create all pairs between words with the same anagram identifier

## Time Complexity Analysis

- **Set Approach**: O(n²) - Due to nested loops for generating pairs
- **Dictionary Approach**: O(n²) - Due to nested loops for generating pairs
- **Space Complexity**: O(n) - We need to store the sets/dictionaries and result

## Example Usage

```python
# Example 1
list_1 = ["cat", "dog", "tac"]
list_2 = ["act", "god", "dog"]
result = anagram(list_1, list_2)
print(result)  # [('cat', 'act'), ('cat', 'tac'), ('dog', 'god')]

# Example 2
list_1 = ["hello", "world"]
list_2 = ["olleh", "dlrow"]
result = anagram(list_1, list_2)
print(result)  # [('hello', 'olleh'), ('world', 'dlrow')]

# Example 3
list_1 = ["a", "b", "c"]
list_2 = ["d", "e", "f"]
result = anagram(list_1, list_2)
print(result)  # []
```

## Key Insights

1. **Anagram Identifier**: Sorted tuple of characters serves as a unique identifier for anagrams
2. **Set Operations**: Using set intersection to find common anagram identifiers
3. **Dictionary Mapping**: Efficiently grouping anagrams by their sorted representation
4. **Pair Generation**: Creating all possible pairs between matching anagrams

## Edge Cases

- **Empty Lists**: Returns empty list
- **No Anagrams**: Returns empty list
- **Single Anagrams**: Returns single pair
- **Multiple Anagrams**: Returns all possible pairs
- **Case Sensitivity**: "Cat" and "cat" are considered different

## Related Problems

- **[Array Intersection](Array_Intersection.md)** - Finding common elements
- **[String Operations](../../Data_Structures/Hash_Tables/String_Operations.md)** - String manipulation techniques
- **[Dictionary Operations](../../Data_Structures/Hash_Tables/Python_Dictionary_Operations.md)** - Dictionary usage patterns
