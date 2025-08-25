---
title: Frequent Words Find
---
# Frequent Words Find

Given a large body of text, we need to identify the three most frequently occurring words. Imagine working with large documents, such as news articles, thesis manuscripts, or even books. Identifying the most common words could give us an overview of the main themes or topics in the text.

A Python dictionary allows us to store data in key-value pairs. In this case, we can use each unique word in the text as a key and the frequency of the word as its corresponding value. As we traverse the document once, we can record the count of each word on the go, avoiding the need for multiple full-document checks. Hence, using a dictionary would drastically reduce our algorithm's time complexity and boost its efficiency.

```Python
def frequent_words_finder(text):
    from collections import defaultdict

    text = text.lower()
    word_list = text.split()
    
    word_counts = defaultdict(int)

    for word in word_list:
        word_counts[word] += 1
    top_three = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    return top_three
```
