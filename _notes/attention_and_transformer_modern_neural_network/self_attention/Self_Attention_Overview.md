---
title: Self-Attention Overview
---

## What is Self-Attention?

**Self-attention** (also called intra-attention) is an attention mechanism where a sequence attends to **itself**. Instead of computing attention between two different sequences (like encoder-decoder attention), each position in a sequence can attend to all positions in the **same** sequence, including itself.

**Core Idea**: Allow every word to directly look at every other word to capture relationships and dependencies within the sequence.

## Self-Attention vs. Encoder-Decoder Attention

### Encoder-Decoder Attention (Cross-Attention)
- **Two sequences**: Source sequence (encoder) and target sequence (decoder)
- **Queries** come from decoder
- **Keys and Values** come from encoder
- **Purpose**: Align target with source (e.g., translation alignment)

### Self-Attention
- **One sequence**: The sequence attends to itself
- **Queries, Keys, Values** all come from the same sequence
- **Purpose**: Capture internal relationships and dependencies

## Why Self-Attention?

### Problem: Sequential Dependencies in Text

Consider: "The animal didn't cross the street because **it** was too tired"

To understand what "it" refers to, the model needs to:
1. Look back at "animal"
2. Understand "animal" is more relevant than "street"
3. Connect "tired" with animate objects

Traditional RNNs:
- Process sequentially: word 1 → word 2 → ... → word 10
- Information about "animal" must propagate through many timesteps
- Gradient vanishing makes long-range dependencies hard

**Self-Attention Solution**:
- Every word can directly attend to "animal"
- Single computation hop from "it" to "animal"
- Parallel processing of all positions

## How Self-Attention Works

Given an input sequence $X = [x_1, x_2, ..., x_n]$:

### Step 1: Create Q, K, V from the Same Input

$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

All three projections use the **same input** $X$, but **different** learned weight matrices.

### Step 2: Compute Self-Attention

$$\text{SelfAttention}(X) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

The output is a new representation where each position has "looked at" all other positions.

## Concrete Example

Input sentence: "The cat sat"

Tokens as vectors (simplified 3D):
```
"The" → [0.1, 0.2, 0.3]
"cat" → [0.4, 0.5, 0.6]
"sat" → [0.7, 0.8, 0.9]
```

After projection to Q, K, V and computing attention:

**Attention Matrix** (row $i$ attends to column $j$):
```
       The   cat   sat
The  [ 0.5   0.3   0.2 ]
cat  [ 0.1   0.6   0.3 ]
sat  [ 0.2   0.4   0.4 ]
```

**Interpretation**:
- "The" attends mostly to itself (0.5) - captures article identity
- "cat" attends mostly to itself (0.6) - captures noun identity
- "sat" attends to both "cat" (0.4) and itself (0.4) - verb-subject relationship

Each output is a weighted mix of all input vectors based on learned relevance.

## Attention Patterns

Self-attention learns to capture various linguistic phenomena:

### 1. Syntactic Dependencies
```
"The quick brown fox jumps"
"fox" → high attention to "quick", "brown" (adjective modification)
"jumps" → high attention to "fox" (subject-verb agreement)
```

### 2. Semantic Relationships
```
"Alice loves Bob, and he loves her too"
"he" → high attention to "Bob" (coreference)
"her" → high attention to "Alice" (coreference)
```

### 3. Long-Range Dependencies
```
"The keys, which were on the table yesterday, are missing"
"are" → high attention to "keys" (agreement across clause)
```

### 4. Positional Relationships
- Adjacent words often have high mutual attention
- Captures local context (n-gram-like patterns)

## Key Properties

### 1. Permutation Equivariance

Without positional encoding, self-attention is **permutation equivariant**:

$$\text{SelfAttention}(P \cdot X) = P \cdot \text{SelfAttention}(X)$$

Where $P$ is any permutation matrix.

**Implication**: Self-attention is insensitive to word order without positional information!
**Solution**: Add [[Positional_Encoding_Overview|positional encodings]] to the input.

### 2. Parallel Computation

Unlike RNNs that process $x_1$, then $x_2$, then $x_3$:
- Self-attention computes all outputs **simultaneously**
- Massive speedup on GPUs
- Enables training on longer sequences

### 3. Constant Path Length

Information can flow from any position to any other in **one step**:
- RNN: Path length proportional to distance (can be $O(n)$)
- Self-Attention: Path length always 1 ($O(1)$)

Better gradient flow for long-range dependencies.

### 4. Explicit Relationship Modeling

The attention matrix $A$ explicitly represents relationships:
- Can visualize which words attend to which
- Provides interpretability
- Can be analyzed for linguistic structure

## Mathematical Details

### Input

$$X \in \mathbb{R}^{n \times d_{model}}$$

Where:
- $n$: sequence length
- $d_{model}$: embedding dimension

### Projections

$$W^Q, W^K \in \mathbb{R}^{d_{model} \times d_k}$$
$$W^V \in \mathbb{R}^{d_{model} \times d_v}$$

Typically: $d_k = d_v = d_{model}$ (in transformers)

### Computation

$$Q = XW^Q \in \mathbb{R}^{n \times d_k}$$
$$K = XW^K \in \mathbb{R}^{n \times d_k}$$
$$V = XW^V \in \mathbb{R}^{n \times d_v}$$

$$\text{Attention Weights} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{n \times n}$$

$$\text{Output} = \text{Attention Weights} \cdot V \in \mathbb{R}^{n \times d_v}$$

### Complexity

- **Time**: $O(n^2 \cdot d)$ - quadratic in sequence length
- **Space**: $O(n^2)$ - must store $n \times n$ attention matrix

This quadratic complexity becomes a bottleneck for very long sequences (thousands of tokens).

## Visualizing Self-Attention

### Attention Matrix Heatmap

For sentence: "The cat sat on the mat"

```
          The  cat  sat  on   the  mat
The      [0.6  0.1  0.1  0.1  0.05 0.05]  ← "The" mostly attends to itself
cat      [0.2  0.5  0.2  0.05 0.03 0.02]  ← "cat" to itself and neighbors
sat      [0.1  0.4  0.3  0.1  0.05 0.05]  ← "sat" to "cat" (subject)
on       [0.1  0.1  0.2  0.4  0.1  0.1 ]  ← "on" to context
the      [0.1  0.05 0.05 0.1  0.6  0.1 ]  ← "the" to itself
mat      [0.05 0.05 0.05 0.2  0.3  0.35]  ← "mat" to "on the mat"
```

Darker = higher attention weight

### Attention Graphs

Represent attention as directed graph:
- Nodes = words
- Edge weight = attention score
- Threshold edges below certain value for clarity

Can reveal:
- Syntactic parse tree structure
- Semantic relationship graphs
- Coreference chains

## Self-Attention Variants

### 1. Scaled Dot-Product (Standard)

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Used in Transformers

### 2. Multi-Head Self-Attention

Run multiple self-attention operations in parallel:

$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

Where each head:
$$\text{head}_i = \text{Attention}(XW^Q_i, XW^K_i, XW^V_i)$$

See: [[Multi_Head_Overview|Multi-Head Attention]]

### 3. Masked Self-Attention

Prevent attending to future positions (for autoregressive models like GPT):

$$\text{Mask}_{ij} = \begin{cases}
0 & \text{if } i \geq j \\
-\infty & \text{if } i < j
\end{cases}$$

After softmax, future positions have 0 attention weight.

### 4. Local Self-Attention

Attend only to nearby positions (window-based):
- Reduces complexity from $O(n^2)$ to $O(n \cdot w)$ where $w$ is window size
- Used in long-document models

### 5. Sparse Self-Attention

Attend to specific patterns (e.g., every 64th position + local neighborhood):
- Used in: Sparse Transformers, Longformer
- Reduces complexity for very long sequences

## Advantages Over RNNs

| RNN | Self-Attention |
|-----|----------------|
| Sequential processing | Parallel processing |
| $O(n)$ path between distant tokens | $O(1)$ path between any tokens |
| Hidden state must compress all context | Direct access to all context |
| Gradient vanishing for long sequences | Direct gradients between any positions |
| Hard to parallelize | Fully parallelizable |

## Limitations

### 1. Quadratic Complexity
- Memory: $O(n^2)$ for attention matrix
- Computation: $O(n^2 d)$ for attention scores
- Becomes prohibitive for $n > 10,000$

### 2. No Inherent Position Information
- Must add positional encodings
- Unlike RNNs which process sequentially

### 3. Interpretability Challenges
- Attention weights ≠ explanation
- Multiple heads make interpretation complex
- See: [[Attention_Patterns|Understanding Attention Patterns]]

## Applications

Self-attention is the core building block of:

- **Transformers**: [[Transformer_Overview|Transformer Architecture]]
- **BERT**: Bidirectional encoder using self-attention
- **GPT**: Autoregressive decoder using masked self-attention
- **Vision Transformers**: Self-attention over image patches
- **Protein Structure Prediction**: AlphaFold uses self-attention
- **Graph Neural Networks**: Attention over graph neighbors

## Implementation Preview

Basic PyTorch implementation structure:

```python
class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_k)
        self.W_k = nn.Linear(d_model, d_k)
        self.W_v = nn.Linear(d_k, d_k)
        self.scale = math.sqrt(d_k)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        Q = self.W_q(x)  # (batch, seq_len, d_k)
        K = self.W_k(x)  # (batch, seq_len, d_k)
        V = self.W_v(x)  # (batch, seq_len, d_k)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(scores, dim=-1)

        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        return output
```

Full implementation: [[Self_Attention_Implementation|Self-Attention Implementation]]

## Learning Path

1. **Start here**: Understand the concept
2. [[Scaled_Dot_Product_Attention|Scaled Dot-Product Details]] - Mathematical deep dive
3. [[Self_Attention_Implementation|Implement from Scratch]] - Build it in code
4. [[Self_Attention_Visualization|Visualize Attention]] - See what it learns
5. [[Multi_Head_Overview|Multi-Head Attention]] - Next level

## Next Steps

Ready for the detailed mechanics?
- [[Scaled_Dot_Product_Attention|Scaled Dot-Product Attention Mathematics]]

Want to code it yourself?
- [[Self_Attention_Implementation|Build Self-Attention from Scratch]]

Curious about multiple attention heads?
- [[Multi_Head_Overview|Multi-Head Attention]]

## Related Topics

- [[Attention_Mechanism_Overview|General Attention Mechanism]]
- [[Transformer_Overview|Transformer Architecture]] - Uses self-attention extensively
- [[Positional_Encoding_Overview|Positional Encoding]] - Essential for self-attention
- [[BERT|BERT]] - Bidirectional self-attention
- [[GPT|GPT]] - Masked self-attention
