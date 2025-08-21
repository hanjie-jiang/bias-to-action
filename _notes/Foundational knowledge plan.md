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
    - â€œSpeech and Language Processingâ€ by Jurafsky & Martin â€” Ch. 3 (Markov models)
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
