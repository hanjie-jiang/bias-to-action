---
title: Multi-Head Attention Overview
---

## What is Multi-Head Attention?

**Multi-head attention** runs multiple attention mechanisms in parallel, each with different learned projections. Instead of having one set of Query, Key, Value weight matrices, we have **h** independent sets, allowing the model to attend to different aspects of the input simultaneously.

**Core Intuition**: Just as humans process information through multiple perspectives (grammar, meaning, context, emotion), multi-head attention lets the model focus on different representation subspaces simultaneously.

## Single-Head Limitations

With single-head attention:
- One attention mechanism tries to capture all relationships
- Query/Key/Value projections are fixed to one learned transformation
- Model might miss important patterns that require different perspectives

**Example**: In "The cat sat on the mat"
- Single head must choose: syntax? semantics? positional relationships?
- Hard to capture all aspects simultaneously

## Multi-Head Solution

Run **h** attention heads in parallel, each learning different patterns:

- **Head 1**: Might learn syntactic dependencies (subject-verb-object)
- **Head 2**: Might learn semantic similarity (related concepts)
- **Head 3**: Might learn positional patterns (adjacent words)
- **Head 4**: Might learn long-range dependencies

Each head has independent learned weights, capturing different relationship types.

## Mathematical Formulation

### Complete Multi-Head Attention

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

Where each head is:

$$\text{head}_i = \text{Attention}(QW^Q_i, KW^K_i, VW^V_i)$$

### Dimensions

For input dimension $d_{model}$ and $h$ heads:

- Typically set: $d_k = d_v = d_{model} / h$
- Each head operates in a **lower-dimensional space**
- Concatenation recovers $d_{model}$ dimensions

**Weight Matrices per Head**:

$$W^Q_i, W^K_i \in \mathbb{R}^{d_{model} \times d_k}$$
$$W^V_i \in \mathbb{R}^{d_{model} \times d_v}$$

**Output Projection**:

$$W^O \in \mathbb{R}^{hd_v \times d_{model}}$$

### Example: 8 Heads, 512-dim Model

- $d_{model} = 512$
- $h = 8$ heads
- $d_k = d_v = 512 / 8 = 64$ per head

Each head:
- $W^Q_i, W^K_i, W^V_i$: Projects 512 → 64 dimensions
- Computes attention in 64-dimensional space
- Outputs 64-dimensional representation

After all heads:
- Concat 8 heads: $8 \times 64 = 512$ dimensions
- Project back through $W^O$: $512 \times 512$

## Step-by-Step Computation

Given input $X \in \mathbb{R}^{n \times d_{model}}$ (sequence length $n$):

### Step 1: Linear Projections for Each Head

For head $i = 1, ..., h$:

$$Q_i = XW^Q_i \in \mathbb{R}^{n \times d_k}$$
$$K_i = XW^K_i \in \mathbb{R}^{n \times d_k}$$
$$V_i = XW^V_i \in \mathbb{R}^{n \times d_v}$$

### Step 2: Compute Attention for Each Head

$$\text{head}_i = \text{Attention}(Q_i, K_i, V_i) = \text{softmax}\left(\frac{Q_iK_i^T}{\sqrt{d_k}}\right)V_i$$

Output: $\text{head}_i \in \mathbb{R}^{n \times d_v}$

### Step 3: Concatenate All Heads

$$\text{Concat} = [\text{head}_1; \text{head}_2; ...; \text{head}_h] \in \mathbb{R}^{n \times hd_v}$$

(Concatenation along the feature dimension)

### Step 4: Final Linear Projection

$$\text{Output} = \text{Concat} \cdot W^O \in \mathbb{R}^{n \times d_{model}}$$

Projects concatenated heads back to model dimension.

## Concrete Example

Input: "The cat sat" (3 tokens, simplified)
- $d_{model} = 8$
- $h = 2$ heads
- $d_k = d_v = 4$ per head

### Head 1 (Syntactic Attention)

Attention pattern might look like:
```
       The   cat   sat
The  [ 0.8   0.1   0.1 ]  ← "The" attends to itself (determiner)
cat  [ 0.3   0.6   0.1 ]  ← "cat" to itself and "The"
sat  [ 0.1   0.7   0.2 ]  ← "sat" to "cat" (subject-verb)
```

### Head 2 (Semantic Attention)

Different attention pattern:
```
       The   cat   sat
The  [ 0.2   0.6   0.2 ]  ← "The" to "cat" (semantic unit)
cat  [ 0.2   0.3   0.5 ]  ← "cat" to "sat" (action relationship)
sat  [ 0.1   0.4   0.5 ]  ← "sat" to both "cat" and itself
```

### Concatenation

Each head outputs 4-dim vectors (3 tokens × 4 dims):
- Head 1: `[h1_t1, h1_t2, h1_t3]` (each 4-dim)
- Head 2: `[h2_t1, h2_t2, h2_t3]` (each 4-dim)

Concatenate: `[h1_t1+h2_t1, h1_t2+h2_t2, h1_t3+h2_t3]` (each 8-dim)

### Final Projection

$W^O$ (8×8) projects concatenated 8-dim vectors to final 8-dim output.

## Why Does This Work?

### 1. Ensemble Effect

Like ensemble methods in ML:
- Multiple "experts" (heads) learn different patterns
- Averaging/combining reduces variance
- More robust representation

### 2. Representation Subspaces

Each head can specialize in a **different representation subspace**:

$$\text{head}_1: \text{space}_1 \subset \mathbb{R}^{d_{model}}$$
$$\text{head}_2: \text{space}_2 \subset \mathbb{R}^{d_{model}}$$

Even though each head has lower dimension ($d_k < d_{model}$), jointly they span the full model space.

### 3. Different Relationship Types

Research shows different heads learn:
- **Positional**: Adjacent words, positional patterns
- **Syntactic**: Grammatical relationships (subject-verb, etc.)
- **Semantic**: Meaning-based relationships
- **Long-range**: Dependencies across distant tokens

See: [[Attention_Patterns|What Do Attention Heads Learn?]]

## Implementation Details

### Efficient Batching

Instead of computing heads sequentially, we can compute all heads in parallel using batch operations:

**Reshape Strategy**:
```python
# Instead of h separate computations:
# for i in range(h):
#     head_i = attention(Q_i, K_i, V_i)

# Do one batched computation:
Q_all = linear_Q(X).view(batch, seq_len, h, d_k).transpose(1, 2)
# Shape: (batch, h, seq_len, d_k)

# Compute attention with h as batch dimension
attention_output = scaled_dot_product(Q_all, K_all, V_all)
# Shape: (batch, h, seq_len, d_v)

# Reshape back
output = attention_output.transpose(1, 2).contiguous().view(batch, seq_len, h*d_v)
```

This leverages GPU parallelism efficiently.

### Parameter Count

For $d_{model} = 512$, $h = 8$, $d_k = d_v = 64$:

Per head:
- $W^Q_i$: $512 \times 64 = 32,768$
- $W^K_i$: $512 \times 64 = 32,768$
- $W^V_i$: $512 \times 64 = 32,768$

All 8 heads: $8 \times 3 \times 32,768 = 786,432$ parameters

Output projection $W^O$: $512 \times 512 = 262,144$ parameters

**Total**: ~1M parameters for one multi-head attention layer

## Complexity Analysis

### Time Complexity

For sequence length $n$, model dimension $d$, and $h$ heads:

1. **Projections**: $O(nd^2)$ (for all $Q, K, V$ projections)
2. **Attention per head**: $O(n^2 d/h) \times h = O(n^2 d)$
3. **Output projection**: $O(nd^2)$

**Total**: $O(n^2d + nd^2)$ (same as single-head!)

### Space Complexity

- **Attention matrices**: $O(hn^2)$ (h separate $n \times n$ matrices)
- **Intermediate activations**: $O(hnd_k) = O(nd)$

**Key Insight**: Multi-head attention has **same asymptotic complexity** as single-head, but with better representational power.

## Advantages of Multi-Head Attention

### 1. Richer Representations

Multiple heads capture complementary information:
```
Head 1: syntax
Head 2: semantics
Head 3: position
...
Combined: holistic understanding
```

### 2. Improved Gradient Flow

With $h$ heads, gradients flow through $h$ different paths:
- More stable training
- Less likely to get stuck in local minima
- Better convergence

### 3. Specialization

Different heads can specialize:
- Some become "syntax experts"
- Others become "semantic experts"
- Emergent division of labor

### 4. Redundancy and Robustness

If one head fails to learn useful patterns:
- Other heads can compensate
- More robust to initialization
- Better generalization

## Empirical Observations

Research on what heads learn (Voita et al., 2019; Clark et al., 2019):

### BERT Attention Patterns

- **Layer 1-2** (shallow): Positional patterns, adjacent tokens
- **Layer 3-6** (middle): Syntactic relationships, phrase boundaries
- **Layer 7-12** (deep): Semantic similarity, coreference

### Some Heads are "Experts"

- **Delimiter heads**: Attend to [CLS], [SEP] tokens
- **Positional heads**: Fixed positional patterns
- **Syntactic heads**: Grammatical relationships
- **Rare heads**: Specialized for rare phenomena

### Head Pruning

Many heads can be removed without significant performance loss:
- Suggests redundancy
- But some heads are critical
- Task-dependent importance

## Comparison: 1 Head vs. Many Heads

| Aspect | Single-Head | Multi-Head (h=8) |
|--------|-------------|------------------|
| Representation | One perspective | h perspectives |
| Parameter count | Lower | Higher (~h times) |
| Computational cost | Lower | Same asymptotically |
| Expressiveness | Limited | Richer |
| Specialization | Generic | Specialized heads |
| Robustness | Lower | Higher (redundancy) |

## Hyperparameter: Number of Heads

Common choices:
- **BERT-base**: 12 heads (768-dim model → 64-dim per head)
- **GPT-3**: 96 heads (12,288-dim model → 128-dim per head)
- **ViT**: 12 heads (768-dim model)

**Guidelines**:
- More heads → more specialization, but diminishing returns
- Typically: $h \in \{8, 12, 16, 32\}$
- Must divide $d_{model}$ evenly: $d_{model} = h \cdot d_k$

## Visualization

### Attention Pattern Example

For "The cat sat on the mat" with 4 heads:

**Head 1** (syntactic):
```
Focuses on: The→cat, cat→sat, mat→on
```

**Head 2** (positional):
```
Focuses on: adjacent tokens (bigrams)
```

**Head 3** (semantic):
```
Focuses on: cat→mat (both nouns)
```

**Head 4** (long-range):
```
Focuses on: first→last token connections
```

Tools for visualization:
- [BertViz](https://github.com/jessevig/bertviz)
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)

## Code Structure Preview

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear layers for Q, K, V projections (all heads)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1. Linear projections for all heads (batched)
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 2. Apply attention for all heads in parallel
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # 3. Concatenate heads
        concat = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 4. Final linear projection
        output = self.W_o(concat)
        return output
```

Full implementation: [[Multi_Head_Implementation|Multi-Head Attention Implementation]]

## When to Use Multi-Head vs Single-Head

**Use Multi-Head** (almost always):
- Standard transformers (BERT, GPT, T5)
- Rich representation needed
- Sufficient computational resources

**Use Single-Head** (rare):
- Extremely resource-constrained
- Very simple tasks
- Ablation studies

In practice, multi-head is the default for transformers.

## Related Concepts

### Multi-Query Attention (MQA)

Shares keys and values across heads, only queries are separate:
- Reduces parameters
- Faster inference
- Used in some large language models

### Grouped-Query Attention (GQA)

Middle ground between multi-head and multi-query:
- Group heads, share K/V within groups
- Used in LLaMA 2

## Summary

Multi-head attention:
- Runs **h** attention mechanisms in parallel
- Each head has independent $W^Q, W^K, W^V$ projections
- Heads operate in **lower-dimensional subspaces** ($d_k = d_{model}/h$)
- Concatenate and project outputs back to $d_{model}$
- Enables learning **diverse relationship patterns**
- **Same complexity** as single-head, but **richer representations**

$$\boxed{\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O}$$

## Next Steps

Ready to implement it?
- [[Multi_Head_Implementation|Build Multi-Head Attention from Scratch]]

Want to understand what heads learn?
- [[Attention_Patterns|Analyzing Attention Patterns]]

Ready for the full architecture?
- [[Transformer_Overview|Transformer Architecture]]

## Related Topics

- [[Self_Attention_Overview|Self-Attention Mechanism]]
- [[Positional_Encoding_Overview|Positional Encoding]]
- [[Transformer_Overview|Transformer Architecture]]
- [[Linear_Algebra_for_ML|Matrix Operations]]
