---
title: Attention Mathematics
---

## Query, Key, Value Framework

The modern attention mechanism operates on three learned transformations of the input:

- **Query (Q)**: What information am I looking for?
- **Key (K)**: What information do I have to offer?
- **Value (V)**: The actual information content

### Mathematical Formulation

Given input sequences, we project them through learned weight matrices:

$$Q = X W^Q$$
$$K = X W^K$$
$$V = X W^V$$

Where:
- $X \in \mathbb{R}^{n \times d_{model}}$: Input sequence ($n$ tokens, $d_{model}$ dimensions)
- $W^Q \in \mathbb{R}^{d_{model} \times d_k}$: Query projection matrix
- $W^K \in \mathbb{R}^{d_{model} \times d_k}$: Key projection matrix
- $W^V \in \mathbb{R}^{d_{model} \times d_v}$: Value projection matrix
- $d_k$: Dimension of queries and keys
- $d_v$: Dimension of values

## Scaled Dot-Product Attention

The complete attention mechanism:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Step-by-Step Breakdown

#### Step 1: Compute Attention Scores

$$S = QK^T$$

Where $S \in \mathbb{R}^{n_q \times n_k}$ (score matrix)

- Each element $S_{ij}$ represents how much query $i$ should attend to key $j$
- Dot product measures similarity between query and key vectors
- Higher dot product = more similar = more attention

**Example**: For sequence length $n=4$:

$$S = \begin{bmatrix}
s_{11} & s_{12} & s_{13} & s_{14} \\
s_{21} & s_{22} & s_{23} & s_{24} \\
s_{31} & s_{32} & s_{33} & s_{34} \\
s_{41} & s_{42} & s_{43} & s_{44}
\end{bmatrix}$$

Each row shows one query's attention to all keys.

#### Step 2: Scale the Scores

$$S_{scaled} = \frac{QK^T}{\sqrt{d_k}}$$

**Why scale?** When $d_k$ is large, dot products grow large in magnitude, pushing softmax into regions with extremely small gradients (saturation).

**Intuition**:
- If $Q$ and $K$ have variance 1, their dot product has variance $d_k$
- Dividing by $\sqrt{d_k}$ normalizes variance back to 1
- Keeps gradients stable during training

#### Step 3: Apply Softmax

$$A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$$

Where $A \in \mathbb{R}^{n_q \times n_k}$ (attention weights matrix)

Softmax is applied **row-wise**:

$$A_{ij} = \frac{\exp(S_{ij} / \sqrt{d_k})}{\sum_{j'=1}^{n_k} \exp(S_{ij'} / \sqrt{d_k})}$$

Properties:
- Each row sums to 1: $\sum_j A_{ij} = 1$
- All values between 0 and 1: $0 \leq A_{ij} \leq 1$
- Can be interpreted as a probability distribution over keys

#### Step 4: Weighted Sum of Values

$$\text{Output} = AV$$

Where Output $\in \mathbb{R}^{n_q \times d_v}$

Each output position is a weighted combination of all value vectors:

$$\text{Output}_i = \sum_{j=1}^{n_k} A_{ij} V_j$$

## Concrete Example

Let's walk through a tiny example with:
- Sequence length: $n = 3$
- Model dimension: $d_{model} = 4$
- Key/Query dimension: $d_k = 2$
- Value dimension: $d_v = 2$

### Input Sequence

$$X = \begin{bmatrix}
1.0 & 0.5 & 0.2 & 0.8 \\
0.3 & 0.9 & 0.1 & 0.4 \\
0.7 & 0.2 & 0.6 & 0.3
\end{bmatrix}$$

### Projection Matrices (Simplified)

$$W^Q = \begin{bmatrix}
0.5 & 0.3 \\
0.2 & 0.4 \\
0.1 & 0.6 \\
0.7 & 0.2
\end{bmatrix}, \quad
W^K = \begin{bmatrix}
0.4 & 0.1 \\
0.3 & 0.5 \\
0.6 & 0.2 \\
0.2 & 0.4
\end{bmatrix}, \quad
W^V = \begin{bmatrix}
0.6 & 0.3 \\
0.2 & 0.7 \\
0.4 & 0.1 \\
0.3 & 0.5
\end{bmatrix}$$

### Step 1: Compute Q, K, V

$$Q = XW^Q = \begin{bmatrix}
1.07 & 0.77 \\
0.47 & 0.57 \\
0.61 & 0.55
\end{bmatrix}$$

$$K = XW^K = \begin{bmatrix}
0.79 & 0.79 \\
0.35 & 0.51 \\
0.62 & 0.41
\end{bmatrix}$$

$$V = XW^V = \begin{bmatrix}
1.09 & 0.97 \\
0.44 & 0.59 \\
0.71 & 0.52
\end{bmatrix}$$

### Step 2: Compute Scores

$$QK^T = \begin{bmatrix}
1.45 & 0.76 & 1.09 \\
0.66 & 0.46 & 0.52 \\
0.92 & 0.49 & 0.61
\end{bmatrix}$$

### Step 3: Scale Scores

With $\sqrt{d_k} = \sqrt{2} \approx 1.414$:

$$\frac{QK^T}{\sqrt{d_k}} = \begin{bmatrix}
1.03 & 0.54 & 0.77 \\
0.47 & 0.33 & 0.37 \\
0.65 & 0.35 & 0.43
\end{bmatrix}$$

### Step 4: Apply Softmax (Row-wise)

$$A = \begin{bmatrix}
0.48 & 0.25 & 0.27 \\
0.35 & 0.31 & 0.34 \\
0.40 & 0.29 & 0.31
\end{bmatrix}$$

Verification: Each row sums to 1.0

### Step 5: Compute Output

$$\text{Output} = AV = \begin{bmatrix}
0.81 & 0.78 \\
0.64 & 0.65 \\
0.71 & 0.68
\end{bmatrix}$$

**Interpretation**:
- Position 1's output is mostly influenced by its own value (weight 0.48) and position 3 (weight 0.27)
- Each output is a context-aware representation incorporating information from all positions

## Why This Design?

### Dot Product Similarity

The dot product $q \cdot k$ measures similarity:

$$q \cdot k = \|q\| \|k\| \cos(\theta)$$

Where $\theta$ is the angle between vectors.

- **Aligned vectors** ($\theta \approx 0$): Large positive dot product → High attention
- **Orthogonal vectors** ($\theta = 90°$): Zero dot product → No attention
- **Opposite vectors** ($\theta = 180°$): Large negative dot product → Negative attention (suppressed by softmax)

### Separate Q, K, V Projections

Why not use the same matrix for all three?

**Flexibility**: Different projections allow:
- **Query**: Specialized for "asking questions"
- **Key**: Specialized for "being searched"
- **Value**: Specialized for "actual content"

**Analogy**: Database queries
- **Query**: Your SQL SELECT statement
- **Key**: Indexed columns for fast lookup
- **Value**: Actual data rows returned

In practice, this separation provides more expressive power and better performance.

## Attention Score Interpretation

Given attention weights $A_{ij}$:

- **$A_{ij}$ close to 1**: Position $i$ strongly attends to position $j$
- **$A_{ij}$ close to 0**: Position $i$ ignores position $j$
- **Uniform distribution**: Position $i$ attends equally to all positions (no strong focus)

### Attention Patterns

Different tasks learn different attention patterns:

- **Translation**: Diagonal or near-diagonal (source-target alignment)
- **Reading Comprehension**: Focuses on relevant context passages
- **Summarization**: Attends to salient sentences
- **Syntactic Tasks**: May learn to attend along dependency edges

## Computational Complexity

For sequence length $n$ and dimension $d$:

1. **Q, K, V Projections**: $O(n \cdot d^2)$ each → $O(3nd^2)$ total
2. **Score Computation** ($QK^T$): $O(n^2 d)$
3. **Softmax**: $O(n^2)$
4. **Output** ($AV$): $O(n^2 d)$

**Total Complexity**: $O(n^2 d + nd^2)$

**Bottleneck**:
- For short sequences ($n < d$): Projections dominate $O(nd^2)$
- For long sequences ($n > d$): Attention matrix dominates $O(n^2 d)$

This quadratic dependence on sequence length is why efficient attention variants exist for very long sequences.

## Memory Requirements

Storing the attention matrix $A \in \mathbb{R}^{n \times n}$ requires $O(n^2)$ memory.

For $n = 512$ and float32:
- $512 \times 512 \times 4 \text{ bytes} = 1 \text{ MB}$ per attention matrix
- For 12 layers × 12 heads: $144 \text{ MB}$ just for attention weights!

## Gradients and Backpropagation

The gradient of attention with respect to its inputs involves:

$$\frac{\partial \text{Attention}}{\partial Q}, \quad \frac{\partial \text{Attention}}{\partial K}, \quad \frac{\partial \text{Attention}}{\partial V}$$

Key insight: All paths are differentiable
- Softmax is differentiable
- Matrix multiplications are differentiable
- Enables end-to-end training via backpropagation

The gradients flow through:
1. Value multiplication
2. Softmax normalization
3. Scaled dot product
4. Projection matrices

## Practical Considerations

### Numerical Stability

When computing softmax, subtract the max for numerical stability:

$$\text{softmax}(x_i) = \frac{\exp(x_i - \max(x))}{\sum_j \exp(x_j - \max(x))}$$

### Masking

For certain positions, we may want to prevent attention:

$$\text{Attention}(Q, K, V, M) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

Where $M$ is a mask matrix:
- $M_{ij} = 0$ if position $i$ can attend to position $j$
- $M_{ij} = -\infty$ otherwise (becomes 0 after softmax)

**Use cases**:
- **Padding mask**: Ignore padding tokens
- **Causal mask**: Future tokens can't attend to past (for autoregressive models)

## Comparison with Other Similarity Functions

### Additive Attention (Bahdanau)

$$\text{score}(q, k) = v^T \tanh(W_1 q + W_2 k)$$

- More parameters than dot product
- Theoretically more expressive
- Slower in practice

### Multiplicative Attention (Luong)

$$\text{score}(q, k) = q^T W k$$

- Adds learnable weight matrix
- Middle ground between additive and dot product

### Scaled Dot Product (Transformer)

$$\text{score}(q, k) = \frac{q^T k}{\sqrt{d_k}}$$

- Simplest and fastest
- Works well in practice
- Industry standard

## Summary

The attention mechanism computes a weighted sum of values based on query-key similarity:

$$\boxed{\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V}$$

**Key Properties**:
- **Parallelizable**: All positions computed simultaneously
- **Flexible**: Variable-length inputs and outputs
- **Differentiable**: End-to-end training via backprop
- **Interpretable**: Attention weights show what the model focuses on

## Next Steps

Now that you understand the mathematics:

- [[Attention_Implementation|Implement Attention from Scratch]] - Build it in Python
- [[Self_Attention_Overview|Self-Attention]] - Apply attention within a sequence
- [[Multi_Head_Overview|Multi-Head Attention]] - Run multiple attentions in parallel

## Related Topics

- [[Linear_Algebra_for_ML|Matrix Multiplication]] - Core operation in attention
- [[Calculus_and_Gradient_Descent|Backpropagation]] - Training attention models
- [[Probability_and_Markov_Overview|Softmax Function]] - Normalization in attention
