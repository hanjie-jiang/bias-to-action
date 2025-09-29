# Hash Tables Overview

## What are Hash Tables?

Hash tables (also known as hash maps) are one of the most important and widely-used data structures in computer science. They provide extremely fast average-case performance for insertion, deletion, and lookup operations.

## Key Concepts

### Basic Structure
- **Key-Value Pairs**: Hash tables store data as key-value pairs
- **Hash Function**: A function that converts keys into array indices
- **Buckets/Slots**: Array positions where data is stored
- **Load Factor**: Ratio of stored elements to total capacity

### How Hash Tables Work
1. **Hashing**: Apply hash function to key to get array index
2. **Storage**: Store the key-value pair at the calculated index
3. **Collision Handling**: Deal with cases where multiple keys hash to the same index
4. **Dynamic Resizing**: Grow/shrink the table to maintain performance

## Time Complexity

| Operation | Average Case | Worst Case |
|-----------|--------------|------------|
| Search    | O(1)         | O(n)       |
| Insert    | O(1)         | O(n)       |
| Delete    | O(1)         | O(n)       |

## Space Complexity
- **Space**: O(n) where n is the number of key-value pairs

## Common Use Cases

### 1. Caching and Memoization
```python
# Simple cache implementation
cache = {}
def expensive_function(x):
    if x in cache:
        return cache[x]
    result = complex_calculation(x)
    cache[x] = result
    return result
```

### 2. Counting and Frequency Analysis
```python
# Count character frequencies
text = "hello world"
freq = {}
for char in text:
    freq[char] = freq.get(char, 0) + 1
```

### 3. Database Indexing
- Primary keys in databases
- Creating fast lookup tables
- Join operations

### 4. Symbol Tables
- Variable names in compilers
- Function lookup tables
- Configuration settings

## Hash Tables in Different Languages

### Python
- **dict**: Built-in hash table implementation
- **set**: Hash table storing only keys (no values)

### Java
- **HashMap**: General-purpose hash table
- **HashSet**: Set implementation using hash table
- **Hashtable**: Thread-safe version

### JavaScript
- **Object**: Properties stored as hash table
- **Map**: Modern hash table implementation
- **Set**: Collection of unique values

## Real-World Applications

1. **Web Browsers**: URL caching, DNS lookups
2. **Databases**: Index structures, query optimization
3. **Compilers**: Symbol tables, keyword recognition
4. **Operating Systems**: Process tables, file systems
5. **Networking**: Routing tables, MAC address tables

## Advantages
- **Fast Operations**: O(1) average case for basic operations
- **Flexible Keys**: Can use various data types as keys
- **Memory Efficient**: Good space utilization with proper load factor
- **Versatile**: Suitable for many different problems

## Disadvantages
- **Worst Case Performance**: Can degrade to O(n) with poor hash function
- **Memory Overhead**: Requires extra space for hash table structure
- **No Ordering**: Elements are not stored in any particular order
- **Hash Function Dependency**: Performance heavily depends on quality of hash function

## Next Steps

1. [Hash Functions and Collisions](Hash_Functions_and_Collisions.md) - Learn about hash function design and collision resolution
2. [Python Dictionaries](Python_Dictionaries.md) - Explore Python's dict implementation
3. [Python Sets](Python_Sets.md) - Understand Python's set implementation
4. [Hash Table Problems](Hash_Table_Problems.md) - Practice common coding problems

## Related Topics
o- [[Array_Intersection]] - Using hash tables for set operations  
- [[Non_Repeating_Elements]] - Hash table applications
- [[Time_Complexity_Guide]] - Understanding Big O notation

---

## 🎓 MIT 6.006 Theoretical Foundation

### Formal Definition
A hash table implements the dictionary Abstract Data Type (ADT) supporting:
- **insert(key, value)**: Insert a key-value pair into the dictionary
- **search(key)**: Return value associated with key, or null if not found  
- **delete(key)**: Remove key-value pair from the dictionary

**Invariants**: 
- Each key appears at most once in the dictionary
- Hash function h(key) consistently maps keys to valid table indices
- Load factor α = n/m where n = number of elements, m = table size

### Complexity Analysis
- **Time Complexity**:
  - Average case: O(1) for insert, search, delete (under simple uniform hashing)
  - Worst case: O(n) when all keys hash to same slot
- **Space Complexity**: O(n) where n is number of stored key-value pairs
- **Amortized Analysis**: Dynamic resizing maintains O(1) amortized cost per operation

### Theoretical Properties
- **Simple Uniform Hashing Assumption**: Each key equally likely to hash to any slot
- **Universal Hashing**: Family of hash functions where collision probability ≤ 1/m
- **Load Factor Impact**: Performance degrades as α approaches 1; typically resize when α > 0.75

### MIT 6.006 Context
- **Unit**: Unit 2 (Hashing and Amortization)
- **Lectures**: Lecture 8 (Hashing with Chaining), Lecture 9 (Table Doubling, Karp-Rabin)  
- **Key Concepts**: Universal hashing, amortized analysis, rolling hash

---

## 🤖 Machine Learning Applications

### Direct Applications
- **Feature Hashing**: Convert high-dimensional categorical features to fixed-size vectors
- **Caching Computations**: Store expensive feature computations to avoid recomputation
- **Vocabulary Management**: Map words/tokens to indices in NLP applications
- **Sparse Data Structures**: Efficiently represent sparse feature vectors

### Algorithmic Connections  
- **Online Learning**: Hash tables enable efficient updates in streaming algorithms
- **Ensemble Methods**: Fast lookup of weak learner predictions for combination
- **Hyperparameter Tuning**: Cache model evaluations to avoid redundant training

### Real-World Examples
- **scikit-learn HashingVectorizer**: Uses hash functions for text feature extraction
- **Word Embeddings**: Hash tables map vocabulary to embedding indices
- **Recommendation Systems**: User-item interaction matrices often use hash-based storage
- **A/B Testing**: Fast lookup of user experiment assignments

### Advanced Connections
- **Locality Sensitive Hashing**: Approximate nearest neighbor search for high-dimensional data
- **Consistent Hashing**: Distributed ML systems for load balancing across nodes
- **Bloom Filters**: Probabilistic data structures for membership testing in large datasets

---

## 🔗 Cross-References and Connections

### Prerequisites
- **Mathematical Background**: Basic probability, understanding of modular arithmetic
- **Algorithmic Concepts**: Arrays, linked lists, basic complexity analysis
- **Recommended Reading**: [Time_Complexity_Guide](../../Resources/Time_Complexity_Guide.md)

### Related Topics  
- **Within This Section**: [Hash_Functions_and_Collisions](Hash_Functions_and_Collisions.md), [Python_Dictionaries](Python_Dictionaries.md)
- **Other Engineering Sections**: [Data Structures Overview](../../Overview/Engineering_and_Data_Structure_Overview.md), [String_Problems](../../Problem_Solving/String_Problems/Unique_Strings.md)  
- **ML Fundamentals**: [Feature Engineering](../../../ml_fundamentals/feature_engineering/data_types_and_normalization.md), [Model Evaluation](../../../ml_fundamentals/model_evaluation/metrics_and_validation.md)
- **Mathematics**: [Probability Theory](../../../probability_and_markov/Probability_and_Markov_Overview.md)

### MIT 6.006 References
- **Lecture Numbers**: 8 (Hashing with Chaining), 9 (Table Doubling, Karp-Rabin)
- **Problem Sets**: Problem Set 2 (Rolling Hash String Matching)  
- **Recitation Materials**: Hash function analysis, collision resolution strategies
- **Further Reading**: CLRS Chapter 11 (Hash Tables)

### Extension Topics
- **Advanced Hashing**: Cuckoo hashing, Robin Hood hashing
- **Research Directions**: Perfect hashing, minimal perfect hashing
- **Distributed Systems**: Consistent hashing, distributed hash tables
