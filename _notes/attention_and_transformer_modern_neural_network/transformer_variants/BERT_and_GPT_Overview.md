---
title: BERT and GPT Overview
---

## Introduction

While the original Transformer used both encoder and decoder, modern architectures often use **only one half** of the architecture, optimized for specific tasks:

- **BERT** (Bidirectional Encoder Representations from Transformers): **Encoder-only**
- **GPT** (Generative Pre-trained Transformer): **Decoder-only**

These two paradigms represent fundamentally different approaches to language modeling and have spawned entire families of models.

## The Pre-training Revolution

Both BERT and GPT leverage **transfer learning** through a two-stage process:

### Stage 1: Pre-training
- Train on massive unlabeled text corpora (billions of words)
- Learn general language representations
- Computationally expensive (weeks on many GPUs)

### Stage 2: Fine-tuning
- Adapt to specific downstream tasks (small labeled datasets)
- Task-specific training (hours to days)
- Achieves state-of-the-art on many benchmarks

**Key Insight**: Language understanding learned during pre-training transfers to many tasks!

---

## BERT: Bidirectional Encoder Representations

### Architecture

**Encoder-Only Transformer**:
- Remove decoder entirely
- Use only the encoder stack
- Bidirectional self-attention (each token sees all other tokens)

**Standard Configurations**:

| Model | Layers | Hidden Size | Attention Heads | Parameters |
|-------|--------|-------------|-----------------|------------|
| BERT-Base | 12 | 768 | 12 | 110M |
| BERT-Large | 24 | 1024 | 16 | 340M |

### Pre-training Objectives

BERT uses two unsupervised tasks:

#### 1. Masked Language Modeling (MLM)

**Idea**: Randomly mask some tokens, predict them from context

**Procedure**:
1. Randomly select 15% of tokens
2. Replace selected tokens:
   - 80% with [MASK] token
   - 10% with random token
   - 10% unchanged
3. Predict the original token

**Example**:
```
Original:  "The cat sat on the mat"
Masked:    "The [MASK] sat on the [MASK]"
Predict:   "cat" and "mat"
```

**Why this works**:
- Forces model to learn bidirectional context
- "cat" requires looking at both "The" (left) and "sat" (right)
- Learns rich representations of words in context

**Why 80/10/10 split?**
- 80% [MASK]: Main training signal
- 10% random: Prevents overfitting to [MASK]
- 10% unchanged: Reduces train-test mismatch

#### 2. Next Sentence Prediction (NSP)

**Idea**: Predict if sentence B follows sentence A

**Procedure**:
1. Create pairs of sentences
   - 50%: Consecutive sentences (label: IsNext)
   - 50%: Random sentences (label: NotNext)
2. Concatenate: [CLS] Sentence A [SEP] Sentence B [SEP]
3. Predict binary label from [CLS] token

**Example**:
```
IsNext:
  A: "The cat sat on the mat."
  B: "It was very comfortable."
  Label: IsNext

NotNext:
  A: "The cat sat on the mat."
  B: "Paris is the capital of France."
  Label: NotNext
```

**Purpose**: Learn sentence-level relationships for tasks like QA, NLI

**Note**: Later research (RoBERTa) showed NSP may not be necessary.

### Input Representation

BERT input combines three embeddings:

$$\text{Input} = \text{TokenEmbedding} + \text{SegmentEmbedding} + \text{PositionalEmbedding}$$

1. **Token Embeddings**: WordPiece vocabulary (~30k tokens)
2. **Segment Embeddings**: Distinguish sentence A vs. sentence B (0 or 1)
3. **Position Embeddings**: Learned positional encodings (max 512 tokens)

**Special Tokens**:
- `[CLS]`: Classification token (first position, used for sequence-level tasks)
- `[SEP]`: Separator between sentences
- `[MASK]`: Masked token placeholder

### Fine-tuning for Downstream Tasks

BERT can be adapted to many tasks by adding a small task-specific layer:

#### Text Classification (Sentiment Analysis, Spam Detection)
```
[CLS] Text [SEP] → BERT → [CLS] representation → Linear → Softmax
```

#### Token Classification (Named Entity Recognition, POS Tagging)
```
Token1 Token2 ... → BERT → Token representations → Linear → Per-token labels
```

#### Question Answering (SQuAD)
```
[CLS] Question [SEP] Context [SEP] → BERT → Span predictions (start, end)
```

#### Sentence Pair Classification (Natural Language Inference)
```
[CLS] Premise [SEP] Hypothesis [SEP] → BERT → [CLS] → Linear → Entailment/Contradiction/Neutral
```

### Strengths

1. **Bidirectional Context**: Sees full context (left and right)
2. **Transfer Learning**: Pre-trained representations work across tasks
3. **State-of-the-Art**: Dominated NLU benchmarks (GLUE, SQuAD) in 2018-2019
4. **Interpretable**: Attention patterns show syntactic structure

### Limitations

1. **Not Designed for Generation**: Encoder-only, no autoregressive generation
2. **[MASK] Token Mismatch**: [MASK] used in training but not in fine-tuning
3. **Computational Cost**: Large models require significant resources
4. **Fixed Maximum Length**: 512 tokens (longer sequences must be truncated)

### BERT Variants

- **RoBERTa** (2019): Remove NSP, larger batches, more data, better performance
- **ALBERT** (2019): Parameter sharing, factorized embeddings (smaller model, similar performance)
- **DistilBERT** (2019): Distilled version (40% smaller, 60% faster, 97% performance)
- **ELECTRA** (2020): Replace MLM with "replaced token detection" (more efficient)
- **DeBERTa** (2021): Disentangled attention, relative position encoding

---

## GPT: Generative Pre-trained Transformer

### Architecture

**Decoder-Only Transformer**:
- Remove encoder entirely
- Use only the decoder stack (without cross-attention)
- **Causal (masked) self-attention**: Token $i$ only sees tokens $\leq i$

**GPT Evolution**:

| Model | Layers | Hidden Size | Parameters | Training Data |
|-------|--------|-------------|------------|---------------|
| GPT-1 | 12 | 768 | 117M | BookCorpus (4.5GB) |
| GPT-2 | 48 | 1600 | 1.5B | WebText (40GB) |
| GPT-3 | 96 | 12288 | 175B | CommonCrawl (570GB) |
| GPT-4 | ? | ? | ~1.7T | ? |

### Pre-training Objective

**Causal Language Modeling (CLM)** / **Autoregressive Generation**:

**Idea**: Predict next token given all previous tokens

$$P(w_t | w_1, w_2, ..., w_{t-1})$$

**Training**:
```
Input:  "The cat sat on"
Target: "cat sat on the"

Predictions:
Position 1: Predict "cat" given "The"
Position 2: Predict "sat" given "The cat"
Position 3: Predict "on" given "The cat sat"
Position 4: Predict "the" given "The cat sat on"
```

**Loss**: Cross-entropy over all positions

$$\mathcal{L} = -\sum_{t=1}^{T} \log P(w_t | w_{<t})$$

**No masking during training** (unlike BERT's MLM), but causal attention ensures position $t$ can't see future.

### Causal Masking

Attention mask prevents attending to future positions:

```
For "A B C D":

       A  B  C  D
A    [ ✓  ✗  ✗  ✗ ]  ← A can only see A
B    [ ✓  ✓  ✗  ✗ ]  ← B can see A, B
C    [ ✓  ✓  ✓  ✗ ]  ← C can see A, B, C
D    [ ✓  ✓  ✓  ✓ ]  ← D can see all
```

Implemented with:
$$\text{Mask}_{ij} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases}$$

### Input Representation

Simpler than BERT:

$$\text{Input} = \text{TokenEmbedding} + \text{PositionalEmbedding}$$

- **Token Embeddings**: Byte-Pair Encoding (BPE) vocabulary (50k tokens in GPT-2/3)
- **Position Embeddings**: Learned positional encodings

No segment embeddings or special [CLS]/[SEP] tokens (though GPT-2 uses `<|endoftext|>`).

### Fine-tuning for Downstream Tasks

GPT uses a unified approach: **frame all tasks as text generation**

#### Text Classification
```
Input: "Review: This movie was great. Sentiment:"
Generate: " positive"
```

#### Question Answering
```
Input: "Context: ... Question: What is...? Answer:"
Generate: " The answer is..."
```

#### Translation
```
Input: "Translate English to French: The cat sat. French:"
Generate: " Le chat était assis."
```

**Key Idea**: Task is specified in the prompt (few-shot or zero-shot learning)

### GPT-3: In-Context Learning

GPT-3 introduced **few-shot prompting** - no fine-tuning needed!

**Zero-Shot**: Just describe the task
```
Translate "Hello" to Spanish:
[model generates] "Hola"
```

**Few-Shot**: Provide examples in the prompt
```
Translate English to Spanish:
English: Hello
Spanish: Hola
English: Goodbye
Spanish: Adiós
English: Thank you
Spanish: [model generates "Gracias"]
```

**Emergent Ability**: With enough scale, models perform tasks not explicitly trained for!

### Strengths

1. **Generative**: Excels at text generation, completion, creative writing
2. **Flexible**: All tasks framed as generation (unified interface)
3. **Scalability**: Performance improves with model size (scaling laws)
4. **Few-Shot Learning**: GPT-3+ can do tasks from examples (no fine-tuning)
5. **Coherent Long-Form Text**: Generates articles, stories, code

### Limitations

1. **Unidirectional Context**: Only sees left context, not right
2. **Less Suitable for Encoding Tasks**: No bidirectional understanding
3. **Expensive**: Large models require massive compute
4. **Hallucination**: Can generate plausible-sounding but incorrect information
5. **Lack of Grounding**: No inherent factual knowledge verification

### GPT Variants

- **GPT-2** (2019): Scaled up GPT-1, demonstrated zero-shot capabilities
- **GPT-3** (2020): 175B parameters, few-shot learning breakthrough
- **Codex** (2021): GPT-3 fine-tuned on code (powers GitHub Copilot)
- **InstructGPT** (2022): Aligned with human feedback (RLHF)
- **ChatGPT** (2022): InstructGPT optimized for conversation
- **GPT-4** (2023): Multimodal (text + images), improved reasoning

---

## BERT vs GPT: Head-to-Head Comparison

| Aspect | BERT (Encoder-Only) | GPT (Decoder-Only) |
|--------|---------------------|---------------------|
| **Architecture** | Encoder stack only | Decoder stack only (no cross-attn) |
| **Attention** | Bidirectional (full context) | Unidirectional (causal/left-to-right) |
| **Pre-training** | Masked Language Modeling | Causal Language Modeling |
| **Strengths** | Understanding, classification | Generation, completion |
| **Best For** | NLU: sentiment, NER, QA span | NLG: text gen, translation, summarization |
| **Fine-tuning** | Task-specific heads | Prompt-based (unified) |
| **Context** | Sees full sentence | Only sees previous tokens |
| **Output** | Token representations | Next-token probabilities |
| **Use Cases** | Search, classification, extraction | Chatbots, code gen, creative writing |

**Analogy**:
- **BERT**: Like a careful reader who can read a sentence multiple times, looking left and right
- **GPT**: Like a writer generating text word-by-word, only seeing what's been written

## When to Use Which?

### Use BERT (or Encoder-Only) for:
- **Text Classification**: Sentiment analysis, spam detection, topic classification
- **Named Entity Recognition**: Extract entities from text
- **Question Answering**: Span extraction (SQuAD-style)
- **Sentence Similarity**: Semantic search, duplicate detection
- **Natural Language Inference**: Entailment, contradiction

### Use GPT (or Decoder-Only) for:
- **Text Generation**: Creative writing, article generation
- **Code Generation**: Programming assistance (Copilot)
- **Summarization**: Abstractive summarization
- **Translation**: Machine translation
- **Dialogue**: Chatbots, conversational AI
- **Few-Shot Learning**: Tasks with limited examples

### Use Encoder-Decoder (T5, BART) for:
- **Seq2Seq Tasks**: Translation, summarization
- **Data-to-Text**: Generate text from structured data
- **Text Simplification**: Rewrite complex text
- **Flexible Generation**: When you want both understanding and generation

## Modern Landscape (2024+)

### Encoder-Only Evolution
- **DeBERTa**: Improved position encoding and attention
- **ELECTRA**: More efficient pre-training
- Still dominant for classification and NLU benchmarks

### Decoder-Only Dominance
- **LLaMA** (Meta): Efficient open-source models
- **PaLM** (Google): 540B parameters, strong reasoning
- **Claude** (Anthropic): Constitutional AI, long context
- **Mistral**: Sparse mixture of experts

**Trend**: Decoder-only models (GPT-style) have become dominant for most tasks due to:
- Unified generation interface
- Scaling laws favor generation
- Few-shot in-context learning
- Easier to prompt than fine-tune

### Instruction Tuning
Modern models add:
- **RLHF** (Reinforcement Learning from Human Feedback): Align with human preferences
- **Instruction Following**: Train on diverse instruction-response pairs
- **Constitutional AI**: Align with principles

Examples: ChatGPT, Claude, Bard

## The Scaling Hypothesis

**Observation**: Performance improves predictably with:
1. Model size (parameters)
2. Dataset size
3. Compute (FLOPs)

**Chinchilla Scaling Laws** (2022): Optimal compute allocation:
- For 10x more compute, increase both model size and data by ~3.2x

**Emergent Abilities**: At sufficient scale, models gain abilities not seen in smaller versions:
- Few-shot learning (GPT-3)
- Chain-of-thought reasoning (GPT-3.5+)
- Instruction following (InstructGPT)

## Summary

### BERT
- **Architecture**: Encoder-only transformer
- **Pre-training**: Masked Language Modeling (MLM)
- **Attention**: Bidirectional
- **Strength**: Understanding and classification
- **Use Cases**: NER, sentiment, QA span extraction

### GPT
- **Architecture**: Decoder-only transformer (no cross-attention)
- **Pre-training**: Causal Language Modeling (CLM)
- **Attention**: Unidirectional (causal masking)
- **Strength**: Generation and completion
- **Use Cases**: Text generation, code, chatbots

### Key Takeaway
- **BERT**: Reader (understands text deeply)
- **GPT**: Writer (generates text fluently)
- **Both**: Revolutionary transfer learning approaches

## Next Steps

Deep dive into specific models:
- [[BERT|BERT Architecture and Implementation]]
- [[GPT|GPT Architecture and Implementation]]
- [[T5|T5 - Unified Text-to-Text Framework]]

Explore training techniques:
- [[Training_Objectives|Pre-training Objectives in Detail]]
- [[Fine_Tuning_Strategies|Fine-tuning Best Practices]]

Understand applications:
- [[Machine_Translation|Machine Translation with Transformers]]
- [[Text_Generation|Text Generation Techniques]]
- [[Question_Answering|Question Answering Systems]]

## Related Topics

- [[Transformer_Architecture_Overview|Original Transformer Architecture]]
- [[Multi_Head_Attention_Overview|Multi-Head Attention]]
- [[Positional_Encoding_Overview|Positional Encoding]]
- [[Ngram_Language_Modeling|N-gram Models (what these replaced)]]
