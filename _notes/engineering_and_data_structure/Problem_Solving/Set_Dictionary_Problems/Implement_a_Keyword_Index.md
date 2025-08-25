---
title: Implement a Keyword Index
---
In this situation, we are given a list of strings, with each string representing a document. Our task is to generate an index of all the distinct words in the documents for quick reference. We need to create a dictionary where each unique word is a key, and the corresponding value is a list of indices pointing to the documents where the word can be found.

```Python
def keyword_index(docs):
    index = {}
    for doc_idx, doc in enumerate(docs):
        for word in doc.split():
            if word in index:
                index[word].append(doc_idx)
            else:
                index[word] = [doc_idx]
    return index
```

```Python
def keyword_index(docs):
    clean_text_list = [text.split() for text in docs]
    dict_search = {}
    for index, text in enumerate(clean_text_list):
        for position, string in enumerate(text):
            if string not in dict_search:
                dict_search[string] = {}
                dict_search[string][index] = 1
            else:
                dict_search[string][index] = dict_search[string].get(index, 0) + 1
    return dict_search

docs = ["Hello world", "world of python", "python is a snake"]
print(keyword_index(docs))  # Expected output: {'Hello': {0: 1}, 'world': {0: 1, 1: 1}, 'of': {1: 1}, 'python': {1: 1, 2: 1}, 'is': {2: 1}, 'a': {2: 1}, 'snake': {2: 1}}
```