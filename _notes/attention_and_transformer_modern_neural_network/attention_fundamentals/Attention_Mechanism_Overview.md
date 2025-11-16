---
title: Attention Mechanism Overview
---

## What is Attention?

The **attention mechanism** is a technique that allows neural networks to dynamically focus on different parts of the input when producing each part of the output. Instead of compressing all information into a fixed-size representation, attention computes a weighted combination of input features based on their relevance to the current task.

**Core Intuition**: When reading a sentence, you don't give equal weight to every word. Similarly, attention mechanisms learn which parts of the input deserve more "attention" at each step.

## The Sequence-to-Sequence Problem

Before attention, encoder-decoder models for tasks like machine translation worked as follows:

1. **Encoder**: Process entire source sentence into a single fixed-size vector (context/thought vector)
2. **Decoder**: Generate target sentence from this single vector

**The Bottleneck Problem**:
- All source information must be compressed into one vector
- Long sentences lose information (the "forgetting problem")
- Distant dependencies are hard to capture

## How Attention Solves This

Instead of using a single context vector, attention:

1. **Keeps all encoder hidden states** (one per input token)
2. **Computes attention weights** for each decoder step
3. **Creates dynamic context vectors** as weighted sums of encoder states

This allows the decoder to "look back" at the entire source sequence and focus on relevant parts when generating each output token.

## High-Level Mechanism

For each decoder step:

```
Step 1: Compute similarity scores
  - How relevant is each encoder state to the current decoder state?

Step 2: Normalize scores into weights
  - Apply softmax to get attention distribution (sums to 1)

Step 3: Compute weighted sum
  - Combine encoder states using attention weights

Step 4: Use context vector
  - Feed the weighted combination to decoder
```

## Visual Example: Machine Translation

Translating "The cat sat on the mat" → "Le chat était sur le tapis"

When generating "chat" (cat):
- High attention on "cat" (0.7)
- Low attention on other words (0.05 each)

When generating "tapis" (mat):
- High attention on "mat" (0.8)
- Low attention on other words

**Key Insight**: The model learns these alignments automatically from data, without explicit alignment labels!

## Types of Attention

### 1. Encoder-Decoder Attention (Cross-Attention)
- Decoder attends to encoder outputs
- Used in: Original seq2seq with attention, Transformer decoder

### 2. Self-Attention
- Sequence attends to itself
- Each position can look at all other positions
- Used in: Transformer encoder, BERT, GPT

### 3. Variants
- **Additive (Bahdanau) Attention**: Uses a feed-forward network to compute scores
- **Multiplicative (Luong) Attention**: Uses dot product for scores
- **Scaled Dot-Product**: Transformer's version (scales by $\sqrt{d_k}$)

## Components of Attention

Modern attention mechanisms use three learned transformations:

### Query (Q)
- **What we're looking for**
- Represents the current decoder state or position asking "what should I focus on?"

### Key (K)
- **What each position offers**
- Represents each encoder state or position advertising "here's what I have"

### Value (V)
- **The actual content**
- The information to aggregate based on attention weights

**Analogy**:
- Query = Your search term on YouTube
- Key = Video titles/tags
- Value = Actual video content
- Attention scores = Relevance ranking
- Output = Weighted combination of top videos

## The Attention Formula

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Breaking it down:
1. $QK^T$: Compute similarity scores (dot product of query with all keys)
2. $\frac{1}{\sqrt{d_k}}$: Scale to prevent small gradients
3. $\text{softmax}$: Normalize to probability distribution
4. Multiply by $V$: Weighted sum of values

## Why Attention Works

### 1. Handles Variable-Length Dependencies
- Can connect any two positions directly
- No intermediate hidden state compression

### 2. Enables Parallelization
- Unlike RNNs, attention can be computed in parallel
- All positions processed simultaneously

### 3. Provides Interpretability
- Attention weights show which inputs influenced which outputs
- Useful for debugging and understanding model decisions

### 4. Flexible and Powerful
- Works for any sequence-to-sequence task
- Generalizes beyond NLP (images, graphs, etc.)

## Learning Path

This section contains:

1. [[Attention_Intuition|Attention Intuition]] - Deep dive into motivation and use cases
2. [[Attention_Math|Attention Mathematics]] - Detailed mathematical formulation
3. [[Attention_Implementation|Attention Implementation]] - Build from scratch in Python
4. [[Attention_Variations|Attention Variations]] - Different attention mechanisms

## Prerequisites

- [[Neural_Networks_and_Deep_Learning_Overview|Basic neural networks]]
- [[Linear_Algebra_for_ML|Matrix multiplication, dot products]]
- [[Probability_and_Markov_Overview|Softmax function]]

## Next Steps

Ready to understand **why** we need attention? Continue to:
- [[Attention_Intuition|Attention Intuition: The Motivation]]

Want to see the **math**? Jump to:
- [[Attention_Math|Attention Mathematics]]

Prefer to **code**? Go directly to:
- [[Attention_Implementation|Build Attention from Scratch]]

## Related Topics

- [[Self_Attention_Overview|Self-Attention]] - Attending within a sequence
- [[Multi_Head_Overview|Multi-Head Attention]] - Multiple attention mechanisms in parallel
- [[Transformer_Overview|Transformer Architecture]] - Where attention shines
- [[Ngram_Language_Modeling|N-gram Models]] - What attention replaced for language modeling
