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
.hero__actions .md-button { margin: .25rem .4rem; }

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
</style>

<div class="hero">
  <div class="hero__content">
    <h1>Machine Learning Notes</h1>
    <p class="subtitle">Curated notes on probability, foundations, and neural networks.</p>
    <p class="hero__actions">
      [ML Fundamentals](ml_fundamentals/ML_fundamentals.md){ .md-button .md-button--primary }
      [Probability & Markov](probability_and_markov/Probability_and_Markov_Overview.md){ .md-button }
      [Language Model](language_model/Ngram_Language_Modeling.md){ .md-button }
    </p>
  </div>
</div>

<section class="cards">
  [<span><h3>Information Theory</h3><p>Entropy, cross‑entropy and KL divergence.</p></span>](Information_Theory.md){ .card }
  [<span><h3>Neural Networks</h3><p>Backpropagation, activations, training tips.</p></span>](Neural_Networks_and_Backprop.md){ .card }
  [<span><h3>Linear Algebra</h3><p>Vectors, matrices, SVD and more.</p></span>](Linear_Algebra_for_ML.md){ .card }
</section>
