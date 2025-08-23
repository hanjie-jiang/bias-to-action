# Hash Functions and Collisions

## Hash Functions

A hash function is the heart of any hash table. It takes a key as input and produces an integer (hash code) that determines where the key-value pair should be stored in the underlying array.

### Properties of Good Hash Functions

#### 1. Deterministic
- Same input always produces same output
- Essential for consistent lookups

#### 2. Uniform Distribution
- Keys should be spread evenly across the hash table
- Minimizes clustering and collisions

#### 3. Fast Computation
- Should be O(1) time complexity
- Avoid expensive operations

#### 4. Avalanche Effect
- Small changes in input cause large changes in output
- Helps distribute similar keys

### Common Hash Function Techniques

#### 1. Division Method
```python
def hash_function(key, table_size):
    return key % table_size
```
- Simple and fast
- Works well with prime table sizes
- Poor performance with certain key patterns

#### 2. Multiplication Method
```python
def hash_function(key, table_size):
    A = 0.6180339887  # Golden ratio - 1
    return int(table_size * ((key * A) % 1))
```
- More uniform distribution
- Table size doesn't need to be prime
- Slightly more expensive computation

#### 3. String Hashing
```python
def hash_string(s, table_size):
    hash_value = 0
    for char in s:
        hash_value = (hash_value * 31 + ord(char)) % table_size
    return hash_value
```
- Polynomial rolling hash
- 31 is a common multiplier (prime number)
- Used in Java's String.hashCode()

## Collisions

A collision occurs when two different keys hash to the same index. Since hash tables have finite size, collisions are inevitable (pigeonhole principle).

### Collision Resolution Strategies

#### 1. Separate Chaining (Open Hashing)

Store multiple key-value pairs at each array index using a secondary data structure.

```python
class HashTableChaining:
    def __init__(self, size):
        self.size = size
        self.table = [[] for _ in range(size)]
    
    def _hash(self, key):
        return hash(key) % self.size
    
    def insert(self, key, value):
        index = self._hash(key)
        bucket = self.table[index]
        
        # Check if key already exists
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)  # Update existing
                return
        
        bucket.append((key, value))  # Add new
    
    def get(self, key):
        index = self._hash(key)
        bucket = self.table[index]
        
        for k, v in bucket:
            if k == key:
                return v
        
        raise KeyError(key)
```

**Advantages:**
- Simple to implement
- Never runs out of space (can always add to chain)
- Good performance with good hash function

**Disadvantages:**
- Extra memory overhead for pointers/lists
- Cache performance can be poor
- Worst case: O(n) if all keys hash to same bucket

#### 2. Open Addressing (Closed Hashing)

Store all key-value pairs directly in the hash table array. When collision occurs, probe for next available slot.

##### Linear Probing
```python
class HashTableLinearProbing:
    def __init__(self, size):
        self.size = size
        self.keys = [None] * size
        self.values = [None] * size
    
    def _hash(self, key):
        return hash(key) % self.size
    
    def insert(self, key, value):
        index = self._hash(key)
        
        # Linear probing
        while self.keys[index] is not None:
            if self.keys[index] == key:
                self.values[index] = value  # Update existing
                return
            index = (index + 1) % self.size
        
        # Insert at empty slot
        self.keys[index] = key
        self.values[index] = value
    
    def get(self, key):
        index = self._hash(key)
        
        # Linear probing for search
        while self.keys[index] is not None:
            if self.keys[index] == key:
                return self.values[index]
            index = (index + 1) % self.size
        
        raise KeyError(key)
```

**Advantages:**
- Better cache performance (data locality)
- No extra memory for pointers
- Simple implementation

**Disadvantages:**
- Primary clustering (consecutive occupied slots)
- Requires careful deletion handling
- Performance degrades as load factor increases

##### Quadratic Probing
```python
def quadratic_probe(self, key):
    index = self._hash(key)
    i = 0
    while self.keys[index] is not None:
        if self.keys[index] == key:
            return index
        i += 1
        index = (self._hash(key) + i*i) % self.size
    return index
```

**Advantages:**
- Reduces primary clustering
- Better distribution than linear probing

**Disadvantages:**
- Secondary clustering
- May not probe all slots
- More complex deletion

##### Double Hashing
```python
def double_hash_probe(self, key):
    hash1 = hash(key) % self.size
    hash2 = 7 - (hash(key) % 7)  # Second hash function
    
    index = hash1
    i = 0
    while self.keys[index] is not None:
        if self.keys[index] == key:
            return index
        i += 1
        index = (hash1 + i * hash2) % self.size
    return index
```

**Advantages:**
- Eliminates clustering
- Good distribution with proper second hash function

**Disadvantages:**
- More complex implementation
- Requires two hash functions

## Load Factor and Resizing

### Load Factor (α)
```
α = n / m
where n = number of elements, m = table size
```

### Performance Impact
- **Separate Chaining**: Performance degrades linearly with load factor
- **Open Addressing**: Performance degrades exponentially after α > 0.7

### Dynamic Resizing
```python
def resize(self):
    old_keys = self.keys
    old_values = self.values
    
    # Double the size
    self.size *= 2
    self.keys = [None] * self.size
    self.values = [None] * self.size
    
    # Rehash all existing elements
    for i in range(len(old_keys)):
        if old_keys[i] is not None:
            self.insert(old_keys[i], old_values[i])
```

## Python's Hash Implementation

### Python dict
- Uses open addressing with random probing
- Maintains insertion order (since Python 3.7)
- Automatically resizes when load factor exceeds 2/3
- Uses sophisticated hash functions for different types

### Python set
- Similar to dict but stores only keys
- Uses dummy values internally
- Same collision resolution strategy

## Common Hash Function Pitfalls

1. **Poor Distribution**: Using simple modulo with non-prime table sizes
2. **Predictable Patterns**: Hash functions that don't handle similar keys well
3. **Integer Overflow**: Not handling large hash values properly
4. **Security Issues**: Hash functions vulnerable to collision attacks

## Best Practices

1. **Choose Prime Table Sizes**: Better distribution with division method
2. **Monitor Load Factor**: Resize before performance degrades
3. **Use Quality Hash Functions**: Consider built-in functions for complex types
4. **Handle Collisions Appropriately**: Choose strategy based on use case
5. **Consider Security**: Use cryptographic hash functions for security-critical applications

## Next Steps
- [Hash Tables Overview](Hash_Tables_Overview.md) - Basic concepts
- [Python Dictionaries](Python_Dictionaries.md) - Implementation details
- [Hash Table Problems](Hash_Table_Problems.md) - Practice problems
