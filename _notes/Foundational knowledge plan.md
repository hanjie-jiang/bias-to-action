---
title: Foundational knowledge plan
---
#### **Week 0: Overview of ML Fundamentals**
The [[ML_fundamentals|ML fundamentals]] section introduces model evaluation, classical algorithms and more, which becomes the building blocks of the topics in the weeks following. 

#### **Week 1-2: Probability Foundations + Markov Assumption**
- [[Probability_and_Markov_Overview]]
- Topics:
    - [[conditional_probability_and_bayes_rule]]
    - [[naive_bayes_and_gaussian_naive_bayes]]
    - [[joint_and_marginal_distributions]]
    - Markov Assumption: what it is and why it matters in NLP
- Resources:
    - [StatQuest: Conditional Probability (YouTube)](https://www.youtube.com/watch?v=_IgyaD7vOOA)
    - [StatQuest: Bayes' Rule](https://www.youtube.com/watch?v=9wCnvr7Xw4E)
    - [3Blue1Brown: Bayes theorem, the geometry of changing beliefs](https://www.youtube.com/watch?v=HZGCoVF3YvM)
	- [StatQuest: Naive Bayes](https://www.youtube.com/watch?v=O2L2Uv9pdDA)
    - [StatQuest: Gaussian Naive Bayes](https://www.youtube.com/watch?v=H3EjCKtlVog)
    - [Khan Academy - Probability & Statistics](https://www.khanacademy.org/math/statistics-probability)
    - Speech and Language Processing by Jurafsky & Martin Ch. 3 (Markov models)
#### **Week 3: N-gram Models & Language Modeling**
- [[Ngram_Language_Modeling]]
- Topics:
    - What is an n-gram?
    - How n-gram language models work
    - Perplexity and limitations of n-gram models
- Activities:
    - Implement a bigram/trigram model on a toy corpus
- Resources:
    - The Illustrated Transformer - start with n-gram part
    - [Happy-LLM intro chapter](language_model/resources/Happy-LLM-v1.0.pdf)
    - Optional: n-gram language model notebook
#### **Week 4: Intro to Information Theory**
- [[Information_Theory]]

- Topics:
    - Entropy, Cross-Entropy, KL Divergence
    - Why they matter in language modeling
- Activities:
    - Manually compute entropy of a simple probability distribution
    - Implement cross-entropy loss
- Resources:
    - [3Blue1Brown â€“ But what is entropy?](https://www.youtube.com/watch?v=H3QBX2Zyb-U)
    - Stanford CS224n Lecture 1
#### **Week 5-6: Linear Algebra for ML**
- [[Linear_Algebra_for_ML]]
- Topics:
    - Vectors, Matrices, Matrix Multiplication
    - Dot product, norms, projections 
    - Eigenvalues & Singular Value Decomposition (SVD)  
- Activities:
    - Practice via small matrix coding problems (NumPy or PyTorch)
- Resources:
    - [3Blue1Brown: Essence of Linear Algebra](https://www.youtube.com/watch?v=kjBOesZCoqc&list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)   
    - Stanford CS229 Linear Algebra Review
        

#### **Week 7: Calculus + Gradient Descent**
- [[Calculus_and_Gradient_Descent]]
- Topics:
    - Partial Derivatives    
    - Chain Rule    
    - Gradients and optimization intuition    
- Activities:
    - Derive gradients of simple functions    
    - Visualize gradient descent in 2D    
- Resources:
    - Khan Academy Calculus (focus on multivariable sections)    
    - [Gradient Descent Visualization (YouTube)](https://www.youtube.com/watch?v=IHZwWFHWa-w)    
#### **Week 8-9: Neural Networks & Backpropagation**
- [[Neural_Networks_and_Deep_Learning_Overview]]
- Topics:
    - [[Introduction_to_Perceptron_Algorithm|Introduction to Perceptron Algorithm]]
    - Structure of a feedforward neural network
    - Activation functions (ReLU, softmax)
    - Backpropagation algorithm
- Activities:
    - Implement a simple NN from scratch (e.g., on MNIST or XOR)
    - Derive gradient of softmax + cross-entropy
- Resources:
    - Michael Nielsenâ€™s NN book: http://neuralnetworksanddeeplearning.com/
    - CS231n lecture on backprop
#### **Week 10: Integration and Project**
- [[Integration_and_Project]]
- Goal:
    - Build a mini-project combining n-gram + neural net ideas
    - Example: Predict the next word using both n-gram and a small MLP
- Outcome:
    - Review all learned concepts
    - Prepare to transition to Happy-LLMâ€™s transformer section

---

## Phase 2: Modern Deep Learning - Attention & Transformers

#### **Week 11-12: Attention Mechanisms**
- [[Attention_and_Transformers_Overview|Attention & Transformers Overview]]
- Topics:
    - Why attention? The sequence-to-sequence motivation
    - [[Attention_Mechanism_Overview|Attention fundamentals]]: Query, Key, Value framework
    - [[Attention_Math|Attention mathematics]]: Scaled dot-product attention
    - Attention vs. RNNs: parallelization and long-range dependencies
- Activities:
    - Implement basic attention mechanism from scratch
    - Visualize attention weights on simple sequences
    - Compare attention to fixed-context approaches
- Resources:
    - [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Jay Alammar
    - [3Blue1Brown: Attention in transformers](https://www.youtube.com/watch?v=eMlx5fFNoYc)
    - [Stanford CS224N: Attention](https://www.youtube.com/watch?v=5vcj8kSwBCY)

#### **Week 13-14: Self-Attention & Multi-Head Attention**
- [[Self_Attention_Overview|Self-Attention Overview]]
- [[Multi_Head_Attention_Overview|Multi-Head Attention Overview]]
- Topics:
    - [[Self_Attention_Overview|Self-attention mechanism]]: attending within a sequence
    - Permutation equivariance and the need for positional encoding
    - [[Positional_Encoding_Overview|Positional encodings]]: sinusoidal vs. learned
    - [[Multi_Head_Attention_Overview|Multi-head attention]]: parallel attention heads
    - What do different attention heads learn?
- Activities:
    - Implement self-attention from scratch (NumPy or PyTorch)
    - Implement multi-head attention
    - Visualize attention patterns across multiple heads
    - Add positional encodings and observe effects
- Resources:
    - [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) - Lilian Weng
    - [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - Harvard NLP
    - Original paper: [Attention is All You Need](https://arxiv.org/abs/1706.03762)

#### **Week 15-16: Transformer Architecture**
- [[Transformer_Architecture_Overview|Transformer Architecture Overview]]
- Topics:
    - Full transformer architecture: encoder and decoder stacks
    - Encoder: multi-head self-attention + feed-forward networks
    - Decoder: masked self-attention + cross-attention + feed-forward
    - Residual connections and layer normalization
    - Training transformers: learning rate warmup, label smoothing
- Activities:
    - Implement a complete transformer from scratch
    - Train on a small machine translation task
    - Experiment with different hyperparameters (heads, layers, dimensions)
    - Analyze attention patterns in trained model
- Resources:
    - [Andrej Karpathy: Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)
    - [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) - Interactive visualization
    - [BertViz](https://github.com/jessevig/bertviz) - Attention visualization tool

#### **Week 17-18: BERT and GPT - Modern Applications**
- [[BERT_and_GPT_Overview|BERT & GPT Overview]]
- Topics:
    - **BERT** (encoder-only): Masked language modeling, bidirectional context
    - **GPT** (decoder-only): Causal language modeling, autoregressive generation
    - Pre-training vs. fine-tuning paradigm
    - Transfer learning with transformers
    - Prompt engineering and few-shot learning (GPT-3)
    - Instruction tuning and RLHF (ChatGPT, InstructGPT)
- Activities:
    - Fine-tune a pre-trained BERT model for text classification
    - Generate text with GPT-2/GPT-3 using different prompting strategies
    - Compare encoder-only vs. decoder-only architectures
    - Experiment with prompt engineering for various tasks
- Resources:
    - [BERT paper](https://arxiv.org/abs/1810.04805) - Devlin et al.
    - [GPT-2 paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - Radford et al.
    - [GPT-3 paper](https://arxiv.org/abs/2005.14165) - Brown et al.
    - [Hugging Face Transformers Course](https://huggingface.co/course)
    - [Stanford CS224N: Pre-training](https://www.youtube.com/watch?v=nTv_Pu_RM5s)

#### **Week 19-20: Advanced Topics & Integration Project**
- Topics:
    - Efficient transformers: Sparse attention, Linformer, Reformer
    - Long-context models: Relative position encoding, ALiBi
    - Vision transformers (ViT): applying transformers to images
    - Multimodal transformers: CLIP, DALL-E, GPT-4
    - State-space models and alternatives to attention
- Integration Project:
    - Build an end-to-end NLP application using transformers
    - Examples:
        - Question answering system with BERT
        - Text summarization with BART/T5
        - Chatbot with GPT-2 fine-tuning
        - Code generation assistant
- Resources:
    - [Vision Transformers paper](https://arxiv.org/abs/2010.11929)
    - [CLIP paper](https://arxiv.org/abs/2103.00020)
    - [State Space Models survey](https://arxiv.org/abs/2312.00752)
