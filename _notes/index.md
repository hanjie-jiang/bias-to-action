---
title: Home
hide:
  - navigation
  - toc
---

<style>
/* Pastel hero */
.hero {
  --bg1: #f7e9ff;   /* lavender */
  --bg2: #e6f7ff;   /* powder blue */
  --bg3: #eaffe6;   /* mint */
  background:
    radial-gradient(1200px 600px at 10% 20%, var(--bg1) 0, transparent 60%),
    radial-gradient(1200px 600px at 90% 10%, var(--bg2) 0, transparent 60%),
    radial-gradient(1200px 600px at 50% 90%, var(--bg3) 0, transparent 60%);
  border-radius: 1.5rem;
  padding: 5rem 2rem;
  margin: 0 0 2rem 0;
  text-align: center;
}
.hero__content { max-width: 900px; margin: 0 auto; }
.hero h1 { margin: 0 0 .5rem 0; font-size: clamp(2rem, 4vw, 3rem); }
.hero .subtitle { font-size: 1.125rem; opacity: .8; margin: 0 0 1.25rem 0; }
.hero__actions .md-button { margin: .25rem .4rem; }

/* Card grid */
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
    <h1>Obsidian Notes</h1>
    <p class="subtitle">Curated notes on ML, probability, and foundations.</p>

    <p class="hero__actions">
      <a href="week1-2_probability_and_markov/Week1/Week1P1-conditional_probability_and_bayes_rule.md" class="md-button md-button--primary">Bayes’ Rule</a>
      <a href="week1-2_probability_and_markov/Week1/Week1P2-naive_bayes_and_gaussian_naive_bayes.md" class="md-button">Naive Bayes</a>
      <a href="week1-2_probability_and_markov/Week1/Week1P3-joint_and_marginal_distributions.md" class="md-button">Joint & Marginal</a>
      <a href="week1-2_probability_and_markov/Week1/Week1P4-ML_fundamentals.md" class="md-button">ML Fundamentals</a>
    </p>
  </div>
</div>

<section class="cards">
  <a class="card" href="Entropy.md">
    <h3>Information Theory</h3>
    <p>Entropy, cross-entropy, KL divergence.</p>
  </a>
  <a class="card" href="Week8_9_Neural_Networks_and_Backprop.md">
    <h3>Neural Networks</h3>
    <p>Backpropagation, activations, training tips.</p>
  </a>
  <a class="card" href="Foundational knowledge plan.md">
    <h3>Study Plans</h3>
    <p>Structured roadmaps & resources.</p>
  </a>
</section>
