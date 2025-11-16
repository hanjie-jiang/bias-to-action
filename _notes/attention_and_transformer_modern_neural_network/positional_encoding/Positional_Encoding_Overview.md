---
title: Positional Encoding Overview
---

## The Position Problem

**Self-attention is permutation-invariant**: If you shuffle the input tokens, you get the same output (just shuffled).

Consider these two sentences:
1. "The cat chased the mouse"
2. "The mouse chased the cat"

Without positional information, self-attention would produce identical representations (just reordered), even though the meanings are opposite!

**The Problem**: Attention mechanisms have no inherent notion of sequence order.

**The Solution**: **Positional encoding** - inject position information into the model.

## Why RNNs Don't Need This

Recurrent Neural Networks (RNNs) process sequentially:
```
h_1 = f(x_1, h_0)
h_2 = f(x_2, h_1)  ← knows it comes after x_1
h_3 = f(x_3, h_2)  ← knows it comes after x_2
```

Position information is **implicit** in the recurrence structure.

**Transformers**: Process all positions in parallel → must **explicitly** encode positions.

## Positional Encoding Methods

### 1. Sinusoidal Positional Encoding (Original Transformer)

Add position-dependent patterns using sine and cosine functions:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

Where:
- $pos$: position in sequence (0, 1, 2, ...)
- $i$: dimension index (0 to $d_{model}/2$)
- $d_{model}$: model dimension

### 2. Learned Positional Embeddings (BERT, GPT)

Learn position embeddings like word embeddings:

$$PE_{pos} = \text{Embedding}(pos)$$

Where each position has a learnable vector.

## Sinusoidal Positional Encoding

### Intuition

Use different frequencies for different dimensions:
- **Low frequencies**: Slowly varying across positions (capture global structure)
- **High frequencies**: Rapidly varying (capture fine-grained position differences)

Think of it like a **binary clock** with smooth transitions:
- Least significant bit: Alternates every position
- Most significant bit: Changes very slowly

### Mathematical Formula

For position $pos$ and dimension $i$:

$$\omega_i = \frac{1}{10000^{2i/d_{model}}}$$

$$PE(pos, 2i) = \sin(pos \cdot \omega_i)$$
$$PE(pos, 2i+1) = \cos(pos \cdot \omega_i)$$

**Even dimensions**: Use sine
**Odd dimensions**: Use cosine

### Example: 4-Dimensional Encoding

For $d_{model} = 4$:

Position 0:
```
PE[0] = [sin(0/10000^0),   cos(0/10000^0),   sin(0/10000^{1/2}), cos(0/10000^{1/2})]
      = [0.000,            1.000,            0.000,              1.000]
```

Position 1:
```
PE[1] = [sin(1/1),         cos(1/1),         sin(1/100),         cos(1/100)]
      = [0.841,            0.540,            0.010,              0.999]
```

Position 2:
```
PE[2] = [sin(2/1),         cos(2/1),         sin(2/100),         cos(2/100)]
      = [0.909,            -0.416,           0.020,              0.998]
```

### Visualization

For a 128-dim model and 50 positions, positional encoding looks like a **heatmap**:

```
         Dim 0   Dim 1   Dim 2   Dim 3   ...  Dim 127
Pos 0   [0.00    1.00    0.00    1.00    ...   0.71 ]
Pos 1   [0.84    0.54    0.01    1.00    ...   0.71 ]
Pos 2   [0.91   -0.42    0.02    1.00    ...   0.71 ]
...
Pos 49  [-0.95  -0.30    0.48    0.88    ...   0.73 ]
```

Each row is a unique "fingerprint" for that position.

## Why Sinusoidal Works

### 1. Unique Encodings

Each position gets a unique vector - no two positions have the same encoding.

### 2. Relative Position Information

For any fixed offset $k$, the encoding at position $pos + k$ is a **linear function** of the encoding at position $pos$:

$$PE(pos+k) = T_k \cdot PE(pos)$$

Where $T_k$ is a linear transformation matrix.

This means the model can learn to attend based on **relative positions**, not just absolute positions.

### 3. Generalization to Longer Sequences

Sinusoidal encodings generalize to sequence lengths **not seen during training**:
- Trained on sequences up to 512 tokens
- Can handle 1024+ tokens at inference
- Encodings are deterministic, not learned

### 4. Smooth Transitions

Adjacent positions have similar encodings:
- $PE(pos)$ and $PE(pos+1)$ are close in embedding space
- Gradual changes across the sequence
- Helps model learn smooth position-dependent functions

## Learned Positional Embeddings

### Approach

Treat positions like vocabulary tokens:

```python
# Create learnable embedding for each position
self.position_embedding = nn.Embedding(max_seq_length, d_model)

# At forward pass
pos_ids = torch.arange(seq_len)  # [0, 1, 2, ..., seq_len-1]
pos_encodings = self.position_embedding(pos_ids)
```

Each position index (0, 1, 2, ...) has a learned $d_{model}$-dimensional vector.

### Advantages

- **Flexibility**: Can learn arbitrary position patterns
- **Task-specific**: Adapts to specific task needs
- **Empirically strong**: Often performs as well or better than sinusoidal

### Disadvantages

- **Fixed maximum length**: Can't generalize beyond training length
- **More parameters**: $O(\text{max\_seq\_length} \times d_{model})$
- **Less interpretable**: No mathematical structure

### Used In

- **BERT**: Learned positions (max 512 tokens)
- **GPT**: Learned positions
- **T5**: Relative position biases (variant)

## How to Add Positional Encoding

### Option 1: Addition (Standard)

Add positional encoding to token embeddings:

$$\text{Input} = \text{TokenEmbedding}(x) + \text{PositionalEncoding}(pos)$$

**Why addition?**
- Simple and effective
- Preserves dimensionality
- Token and position information mix naturally

### Option 2: Concatenation (Rare)

Concatenate token and positional embeddings:

$$\text{Input} = [\text{TokenEmbedding}(x); \text{PositionalEncoding}(pos)]$$

**Drawbacks**:
- Doubles dimension
- More parameters
- Less commonly used

**Standard practice**: Use addition.

## Implementation: Sinusoidal

```python
import torch
import math

def create_sinusoidal_positional_encoding(seq_len, d_model):
    """
    Create sinusoidal positional encoding matrix.

    Args:
        seq_len: Maximum sequence length
        d_model: Model dimension

    Returns:
        PE: (seq_len, d_model) positional encoding matrix
    """
    PE = torch.zeros(seq_len, d_model)

    position = torch.arange(0, seq_len).unsqueeze(1).float()  # (seq_len, 1)

    div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                         -(math.log(10000.0) / d_model))

    # Even dimensions: sin
    PE[:, 0::2] = torch.sin(position * div_term)

    # Odd dimensions: cos
    PE[:, 1::2] = torch.cos(position * div_term)

    return PE

# Usage
seq_len = 100
d_model = 512
PE = create_sinusoidal_positional_encoding(seq_len, d_model)

# Add to token embeddings
token_embeddings = ...  # (batch, seq_len, d_model)
input_with_position = token_embeddings + PE[:token_embeddings.size(1), :]
```

## Implementation: Learned

```python
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_seq_len, d_model)

    def forward(self, seq_len):
        positions = torch.arange(seq_len, device=self.position_embeddings.weight.device)
        return self.position_embeddings(positions)

# Usage
pos_encoder = LearnedPositionalEncoding(max_seq_len=512, d_model=512)

token_embeddings = ...  # (batch, seq_len, d_model)
pos_encodings = pos_encoder(token_embeddings.size(1))  # (seq_len, d_model)
input_with_position = token_embeddings + pos_encodings
```

## Relative Positional Encoding (Advanced)

Instead of absolute positions, encode **relative distances** between tokens.

### T5 Relative Position Biases

Add learned biases to attention scores based on distance:

$$\text{Attention Score}_{ij} = q_i \cdot k_j + b_{i-j}$$

Where $b_{i-j}$ is a learned bias for relative distance $i-j$.

**Advantages**:
- Focuses on relative positions (often more important than absolute)
- Better generalization to longer sequences
- Used in T5, DeBERTa

### Rotary Position Embedding (RoPE)

Rotate query and key vectors based on position:

$$q_m' = R_m q_m, \quad k_n' = R_n k_n$$

Where $R_m$ is a rotation matrix for position $m$.

**Properties**:
- Dot product $q_m' \cdot k_n'$ depends on relative position $m - n$
- Very efficient
- Used in: GPT-Neo, PaLM, LLaMA

## Comparing Methods

| Method | Pros | Cons | Used In |
|--------|------|------|---------|
| **Sinusoidal** | No extra params, generalizes beyond training length, interpretable | May underperform learned | Original Transformer |
| **Learned Absolute** | Flexible, task-adaptive, empirically strong | Fixed max length, more params | BERT, GPT |
| **Relative (T5)** | Generalizes well, focuses on relative distance | More complex, added computation | T5, DeBERTa |
| **RoPE** | Efficient, strong performance, relative position | More complex implementation | LLaMA, GPT-Neo |

## When Position Matters Most

Position information is crucial for:

1. **Word Order**: "dog bites man" vs "man bites dog"
2. **Syntax**: Subject-verb-object order
3. **Temporal Sequences**: "before" vs "after"
4. **Positional References**: "first", "last", "next"

Less crucial for:
- Bag-of-words tasks (sentiment on keywords)
- Set-based reasoning (where order doesn't matter)

## Ablation Studies

What happens without positional encoding?

**Results** (on various tasks):
- Translation: ~5-10 BLEU score drop
- Question Answering: ~10-15% accuracy drop
- Language Modeling: ~0.5-1.0 perplexity increase

**Conclusion**: Positional encoding is **essential** for most NLP tasks.

## Visualizing Position Information

### Similarity Matrix

Compute cosine similarity between positional encodings:

```
Similarity[i, j] = cos(PE[i], PE[j])
```

For sinusoidal encodings, you'll see:
- High similarity for nearby positions
- Periodic patterns (due to sine/cosine waves)
- Gradual decrease with distance

### Attention Pattern Changes

With positional encoding:
- Attention patterns become position-aware
- Models can learn position-specific behaviors
- Enables local vs. global attention strategies

## Summary

**The Problem**: Transformers process all tokens in parallel → no inherent position information

**The Solution**: Add positional encodings to input embeddings

**Two Main Approaches**:
1. **Sinusoidal** (Original Transformer)
   - Deterministic sine/cosine functions
   - Generalizes to unseen lengths
   - No extra parameters

2. **Learned** (BERT, GPT)
   - Learnable embedding per position
   - Task-adaptive
   - Fixed maximum length

**Modern Variants**:
- Relative position biases (T5)
- Rotary Position Embeddings (RoPE) - LLaMA

**Usage**:
$$\text{Input to Transformer} = \text{Token Embeddings} + \text{Positional Encodings}$$

## Next Steps

Ready to see the full architecture?
- [[Transformer_Overview|Transformer Architecture]]

Want to implement positional encoding?
- [[Position_Encoding_Implementation|Positional Encoding Implementation]]

Interested in advanced position methods?
- [[Relative_Positional_Encoding|Relative Position Encoding]]

## Related Topics

- [[Self_Attention_Overview|Self-Attention]] - Why we need position information
- [[Transformer_Overview|Transformer Architecture]] - Where positional encoding fits
- [[Linear_Algebra_for_ML|Vector Operations]] - Understanding embeddings
