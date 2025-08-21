---
title: Home
hide:
  - toc
---

<style>
/* Pastel hero + cards */
.hero {
  --bg1: #f7e9ff;
  --bg2: #e6f7ff;
  --bg3: #eaffe6;
  background:
    radial-gradient(1200px 600px at 10% 20%, var(--bg1) 0, transparent 60%),
    radial-gradient(1200px 600px at 90% 10%, var(--bg2) 0, transparent 60%),
    radial-gradient(1200px 600px at 50% 90%, var(--bg3) 0, transparent 60%);
  border-radius: 1.5rem;
  padding: 5rem 2rem;
  margin-bottom: 2rem;
  text-align: center;
}
.hero__content { max-width: 900px; margin: 0 auto; }
.hero h1 { margin: 0 0 .5rem; font-size: clamp(2rem, 4vw, 3rem); }
.hero .subtitle { font-size: 1.125rem; opacity: .8; margin-bottom: 1.25rem; }
.hero__actions a { margin: .25rem .4rem; }

/* Cards */
.cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 14px;
}
.card {
  display: block;
  background: var(--md-surface, #fff);
  border: 1px solid rgba(0,0,0,.06);
  border-radius: 16px;
  padding: 16px 16px 18px;
  text-decoration: none !important;
  color: inherit;
  box-shadow: 0 2px 10px rgba(0,0,0,.04);
  transition: transform .12s ease, box-shadow .12s ease, border-color .12s ease;
}
.card:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 18px rgba(0,0,0,.06);
  border-color: rgba(0,0,0,.12);
}
.card h3 { margin: 4px 0 6px; font-size: 1.05rem; }

/* Use Material's button look without Markdown helpers */
.md-button { display: inline-block; padding: .5rem .9rem; border-radius: .5rem; border: 1px solid rgba(0,0,0,.12); }
.md-button--primary { background: var(--md-primary-fg-color, #6200ee); color: #fff; border-color: transparent; }
.md-button:hover { filter: brightness(0.98); }
</style>

<div class="hero">
  <div class="hero__content">
    <h1>Machine Learning Notes</h1>
    <p class="subtitle">Curated notes on probability, foundations, and neural networks.</p>
    <p class="hero__actions">
      <a class="md-button md-button--primary" href="/ml-learning-notes/ml_fundamentals/ML_fundamentals/">ML Fundamentals</a>
      <a class="md-button" href="/ml-learning-notes/probability_and_markov/Probability_and_Markov_Overview/">Probability &amp; Markov</a>
      <a class="md-button" href="/ml-learning-notes/language_model/Ngram_Language_Modeling/">Language Model</a>
    </p>
  </div>
</div>

<section class="cards">
  <a class="card" href="/ml-learning-notes/Information_Theory/">
    <h3>Information Theory</h3>
    <p>Entropy, cross-entropy and KL divergence.</p>
  </a>
  <a class="card" href="/ml-learning-notes/Neural_Networks_and_Backprop/">
    <h3>Neural Networks</h3>
    <p>Backpropagation, activations, training tips.</p>
  </a>
  <a class="card" href="/ml-learning-notes/Linear_Algebra_for_ML/">
    <h3>Linear Algebra</h3>
    <p>Vectors, matrices, SVD and more.</p>
  </a>
</section>