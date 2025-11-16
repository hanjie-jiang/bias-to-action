---
title: Transformer Architecture Overview
---

## Introduction

The **Transformer** architecture, introduced in "Attention is All You Need" (Vaswani et al., 2017), revolutionized deep learning by replacing recurrence with pure attention mechanisms. It consists of an **encoder** that processes the input and a **decoder** that generates the output, both built entirely from attention and feed-forward layers.

**Key Innovation**: Parallelizable sequence processing with attention, eliminating the sequential bottleneck of RNNs.

## High-Level Architecture

```
Input Sequence → [Encoder] → Context → [Decoder] → Output Sequence
```

### Encoder
- Processes the input sequence (e.g., English sentence)
- Builds rich contextual representations
- Stack of $N$ identical layers (typically $N=6$)

### Decoder
- Generates output sequence (e.g., French translation)
- Attends to encoder output (cross-attention)
- Stack of $N$ identical layers

## Full Transformer Architecture Diagram

```
Input                                         Output (shifted right)
  ↓                                                  ↓
Input Embedding                            Output Embedding
  ↓                                                  ↓
+ Positional Encoding                      + Positional Encoding
  ↓                                                  ↓
┌─────────────────────┐                  ┌─────────────────────┐
│   ENCODER STACK     │                  │   DECODER STACK     │
│                     │                  │                     │
│  ┌──────────────┐   │                  │  ┌──────────────┐   │
│  │ Multi-Head   │   │                  │  │ Masked       │   │
│  │ Self-Attn    │   │                  │  │ Multi-Head   │   │
│  └──────────────┘   │                  │  │ Self-Attn    │   │
│         ↓           │                  │  └──────────────┘   │
│  Add & LayerNorm    │                  │         ↓           │
│         ↓           │                  │  Add & LayerNorm    │
│  ┌──────────────┐   │   Encoder        │         ↓           │
│  │ Feed-Forward │   │   Output ------> │  ┌──────────────┐   │
│  └──────────────┘   │                  │  │ Multi-Head   │   │
│         ↓           │                  │  │ Cross-Attn   │◄──┘
│  Add & LayerNorm    │                  │  └──────────────┘
│         ↓           │                  │         ↓
│  (repeat N times)   │                  │  Add & LayerNorm
└─────────────────────┘                  │         ↓
         ↓                               │  ┌──────────────┐
    Encoder Output ──────────────────────┤  │ Feed-Forward │
                                         │  └──────────────┘
                                         │         ↓
                                         │  Add & LayerNorm
                                         │         ↓
                                         │  (repeat N times)
                                         └─────────────────────┘
                                                   ↓
                                            Linear Layer
                                                   ↓
                                               Softmax
                                                   ↓
                                         Output Probabilities
```

## Encoder Architecture

### Single Encoder Layer

Each encoder layer contains two sub-layers:

#### 1. Multi-Head Self-Attention

$$\text{Attention}(Q, K, V) = \text{MultiHead}(X, X, X)$$

- Query, Key, Value all come from the **same** input (self-attention)
- Each position attends to all positions in the input sequence
- Captures relationships within the source sequence

#### 2. Position-Wise Feed-Forward Network

$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

- Applied to each position independently and identically
- Two linear transformations with ReLU activation
- Typically: $d_{ff} = 4 \times d_{model}$ (e.g., 2048 for 512-dim model)

### Residual Connections and Layer Normalization

Each sub-layer is wrapped with:

$$\text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

- **Residual connection** $(x + \cdots)$: Helps gradient flow
- **Layer normalization**: Stabilizes training

### Complete Encoder Layer

```python
def encoder_layer(x):
    # Multi-head self-attention
    attn_output = multi_head_self_attention(x, x, x)
    x = layer_norm(x + attn_output)  # Residual + LayerNorm

    # Feed-forward network
    ffn_output = feed_forward(x)
    x = layer_norm(x + ffn_output)  # Residual + LayerNorm

    return x
```

### Encoder Stack

Stack $N=6$ identical encoder layers:

$$\text{Encoder}(x) = \text{Layer}_6(\text{Layer}_5(...\text{Layer}_1(x)))$$

Each layer refines the representation, capturing increasingly abstract patterns.

## Decoder Architecture

### Single Decoder Layer

Each decoder layer contains **three** sub-layers:

#### 1. Masked Multi-Head Self-Attention

$$\text{MaskedAttention}(Q, K, V) = \text{MultiHead}(Y, Y, Y, \text{mask})$$

- Self-attention over the **output** sequence
- **Causal masking**: Position $i$ can only attend to positions $\leq i$
- Prevents looking at future tokens during training

**Mask Matrix**:
```
For sequence "A B C D", allow:
A can attend to: [A]
B can attend to: [A, B]
C can attend to: [A, B, C]
D can attend to: [A, B, C, D]
```

Implemented as:
$$\text{Mask}_{ij} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases}$$

#### 2. Multi-Head Cross-Attention (Encoder-Decoder Attention)

$$\text{CrossAttention}(Q, K, V) = \text{MultiHead}(Y, \text{EncoderOutput}, \text{EncoderOutput})$$

- **Query** comes from decoder (what decoder is looking for)
- **Key and Value** come from encoder output (what source provides)
- This is where decoder "reads" the input sequence
- Each decoder position can attend to **all encoder positions**

**This is the key mechanism for seq2seq tasks!**

Example (translation):
- Decoder generating "chat" (cat)
- Cross-attention focuses on encoder's "cat" representation
- Learns source-target alignment automatically

#### 3. Position-Wise Feed-Forward Network

Same as encoder:
$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

### Complete Decoder Layer

```python
def decoder_layer(y, encoder_output):
    # 1. Masked multi-head self-attention
    masked_attn_output = masked_multi_head_self_attention(y, y, y)
    y = layer_norm(y + masked_attn_output)

    # 2. Multi-head cross-attention
    cross_attn_output = multi_head_attention(
        query=y,
        key=encoder_output,
        value=encoder_output
    )
    y = layer_norm(y + cross_attn_output)

    # 3. Feed-forward network
    ffn_output = feed_forward(y)
    y = layer_norm(y + ffn_output)

    return y
```

### Decoder Stack

Stack $N=6$ identical decoder layers, each receiving encoder output:

$$\text{Decoder}(y, \text{enc}) = \text{Layer}_6(...\text{Layer}_1(y, \text{enc}), \text{enc})$$

## Three Types of Attention in Transformer

### 1. Encoder Self-Attention
- **Q, K, V** all from encoder input
- **Bidirectional**: Each position sees entire input sequence
- **Purpose**: Build contextual representations of source

### 2. Decoder Masked Self-Attention
- **Q, K, V** all from decoder input
- **Unidirectional**: Position $i$ only sees positions $\leq i$
- **Purpose**: Generate output autoregressively, prevent future peeking

### 3. Decoder Cross-Attention (Encoder-Decoder Attention)
- **Q** from decoder
- **K, V** from encoder output
- **Purpose**: Align target with source, enable translation/transformation

## Input and Output Processing

### Input Sequence (Encoder)

1. **Token Embedding**: Map tokens to $d_{model}$-dimensional vectors
2. **Positional Encoding**: Add position information
3. **Feed to Encoder Stack**

$$\text{EncoderInput} = \text{Embedding}(x) + \text{PositionalEncoding}$$

### Output Sequence (Decoder)

1. **Token Embedding**: Map output tokens to vectors
2. **Positional Encoding**: Add position information
3. **Feed to Decoder Stack**

$$\text{DecoderInput} = \text{Embedding}(y) + \text{PositionalEncoding}$$

### Final Output

After decoder stack:

1. **Linear Layer**: Project to vocabulary size
   $$\text{Logits} = \text{DecoderOutput} \cdot W_{output}$$

2. **Softmax**: Convert to probabilities
   $$P(\text{word}) = \text{softmax}(\text{Logits})$$

## Training

### Training Procedure

**Teacher Forcing**: During training, use ground truth previous tokens (not model predictions)

Input: "The cat sat"
Target: "<START> Le chat était <END>"

Decoder receives: "<START> Le chat était"
Decoder predicts: "Le chat était <END>"

### Loss Function

**Cross-Entropy Loss** over all positions:

$$\mathcal{L} = -\sum_{t=1}^{T} \log P(y_t | y_{<t}, x)$$

Where:
- $y_t$: True token at position $t$
- $y_{<t}$: All previous tokens
- $x$: Source sequence

### Optimization

- **Optimizer**: Adam with custom learning rate schedule
- **Learning Rate Warmup**: Increase LR for first 4000 steps, then decay

  $$\text{LR} = d_{model}^{-0.5} \cdot \min(\text{step}^{-0.5}, \text{step} \cdot \text{warmup}^{-1.5})$$

- **Regularization**: Dropout, label smoothing

## Inference (Generation)

### Autoregressive Generation

At test time, generate one token at a time:

```
1. Start with <START> token
2. Feed to decoder (with encoder output)
3. Get probability distribution over vocabulary
4. Sample or pick top token
5. Append to sequence
6. Repeat until <END> token or max length
```

**Greedy Decoding**: Pick most probable token each step

**Beam Search**: Maintain top-k hypotheses, better quality but slower

## Hyperparameters (Original Paper)

| Hyperparameter | Base Model | Big Model |
|----------------|------------|-----------|
| Layers ($N$) | 6 | 6 |
| $d_{model}$ | 512 | 1024 |
| $d_{ff}$ | 2048 | 4096 |
| Attention heads | 8 | 16 |
| $d_k = d_v$ | 64 | 64 |
| Dropout | 0.1 | 0.3 |
| Parameters | 65M | 213M |

## Model Size Breakdown

For base model ($d_{model}=512$, $h=8$, $N=6$):

**Per Layer**:
- Multi-head attention: ~1M params (Q, K, V, O projections)
- Feed-forward: ~4M params (2 linear layers with $d_{ff}=2048$)
- Layer norms: ~1K params (negligible)

**Total**:
- Encoder: 6 layers × 5M ≈ 30M params
- Decoder: 6 layers × 7.5M ≈ 45M params (extra cross-attention)
- Embeddings: ~20M params (vocab size × $d_{model}$)
- **Total**: ~100M parameters (varies with vocabulary size)

## Why Transformers Work So Well

### 1. Parallelization
- All positions processed simultaneously
- Massive speedup on GPUs compared to RNNs
- Enables training on huge datasets

### 2. Direct Connections
- Any position can attend to any other in $O(1)$ steps
- RNNs require $O(n)$ steps for distant dependencies
- Better gradient flow

### 3. Flexibility
- Same architecture for many tasks (translation, summarization, QA)
- Easy to adapt with minor changes
- Scales well with data and compute

### 4. Interpretability
- Attention weights show what model focuses on
- Can visualize source-target alignments
- Debugging and analysis easier than RNNs

## Limitations

### 1. Quadratic Complexity
- Self-attention: $O(n^2)$ time and space
- Prohibitive for very long sequences (>10k tokens)
- Solutions: Sparse attention, Linformer, Performer

### 2. No Positional Inductive Bias
- Must explicitly add positional encodings
- RNNs get position "for free"
- May be less sample-efficient for small data

### 3. Fixed Context Length
- Maximum sequence length set at training
- Can't easily extend to longer sequences
- Solutions: Relative positions, ALiBi

## Variants and Extensions

### Encoder-Only (BERT)
- Remove decoder
- Use only encoder stack
- For classification, named entity recognition, etc.
- Bidirectional context for every token

### Decoder-Only (GPT)
- Remove encoder
- Use only decoder stack (with masking)
- For text generation, language modeling
- Autoregressive generation

### Encoder-Decoder (T5, BART)
- Keep both encoder and decoder
- For seq2seq tasks: translation, summarization
- Most flexible but more parameters

## Comparison with RNNs

| Aspect | RNN/LSTM | Transformer |
|--------|----------|-------------|
| Processing | Sequential | Parallel |
| Long-range deps | Difficult ($O(n)$ path) | Easy ($O(1)$ path) |
| Training speed | Slow | Fast |
| Memory | $O(n)$ | $O(n^2)$ |
| Positional info | Implicit | Explicit |
| Interpretability | Hard | Easier (attention) |

## Applications

The transformer architecture enables:

- **Machine Translation**: Original use case (WMT datasets)
- **Text Summarization**: Encoder-decoder for abstractive summarization
- **Question Answering**: BERT-style encoder for SQuAD
- **Text Generation**: GPT-style decoder for creative writing
- **Code Generation**: GitHub Copilot (GPT-based)
- **Image Processing**: Vision Transformers (ViT)
- **Protein Folding**: AlphaFold2
- **Speech Recognition**: Whisper (OpenAI)

## Summary

The Transformer consists of:

**Encoder**:
- Multi-head self-attention (bidirectional)
- Feed-forward network
- Residual connections + Layer normalization
- Stacked $N$ times

**Decoder**:
- Masked multi-head self-attention (unidirectional)
- Multi-head cross-attention (attend to encoder)
- Feed-forward network
- Residual connections + Layer normalization
- Stacked $N$ times

**Key Innovations**:
- Pure attention (no recurrence)
- Parallel processing
- Direct connections between any positions
- Scales efficiently with data and compute

**Three Attention Types**:
1. Encoder self-attention: Understand source
2. Decoder masked self-attention: Generate target autoregressively
3. Cross-attention: Align target with source

## Next Steps

Ready to implement it?
- [[Full_Transformer_Implementation|Build Transformer from Scratch]]

Want to understand specific components?
- [[Encoder_Architecture|Detailed Encoder Architecture]]
- [[Decoder_Architecture|Detailed Decoder Architecture]]
- [[Encoder_Decoder_Attention|Cross-Attention Deep Dive]]

Interested in modern variants?
- [[BERT|BERT - Encoder-Only]]
- [[GPT|GPT - Decoder-Only]]
- [[T5|T5 - Encoder-Decoder]]

## Related Topics

- [[Self_Attention_Overview|Self-Attention Mechanism]]
- [[Multi_Head_Attention_Overview|Multi-Head Attention]]
- [[Positional_Encoding_Overview|Positional Encoding]]
- [[Neural_Networks_and_Deep_Learning_Overview|Neural Network Foundations]]
- [[Ngram_Language_Modeling|N-gram Models (what transformers replaced)]]
