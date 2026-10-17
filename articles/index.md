---
title: Articles
layout: page
lede: Methods reviews, tutorials and case studies from my learning journey.
---

<div class="paper">
  <div class="paper-title"><a href="{{ '/articles/explainable-multisource-lulc-framework/' | relative_url }}">An Explainable Multi-Source Machine Learning Framework for Urban Land Cover Classification</a></div>
  <div class="paper-meta"><span class="tag">Methods review</span><span class="tag">Tutorial</span> Working draft · October 2026</div>
  <p class="paper-summary">A step-by-step framework for combining Sentinel-2, terrain, population density and spectral indices with Random Forest, XGBoost, LightGBM and CatBoost, then explaining the model with SHAP. It includes the equations, accuracy assessment, reporting templates and a Python appendix. It reports no new results.</p>
</div>

<div class="paper">
  <div class="paper-title"><a href="{{ '/articles/lugbe-case-study/' | relative_url }}">Reference study (external): Explainable Ensemble Machine Learning Approach for Land Cover Classification in Lugbe, Abuja, Nigeria</a></div>
  <div class="paper-meta"><span class="tag">External reference</span> Ekweozoh, Musa, Adejoro, Nwokolo, Ahmed &amp; Salaudeen · <em>Global Scientific Journal</em> 14(8), August 2026 · <a href="https://www.globalscientificjournal.com/researchpaper/EXPLAINABLE_ENSEMBLE_MACHINE_LEARNING_APPROACH_FOR_LAND_COVER_CLASSIFICATION_IN_LUGBE_ABUJA_NIGERIA.pdf">PDF</a></div>
  <p class="paper-summary">A published case study I came across and am using as a reference (not my work); summary checked against the paper. Sentinel-2 combined with terrain, population density and five spectral indices; four tree ensembles; all four within 59.3–59.6% accuracy, with LightGBM selected for its much shorter training time; SHAP highlights NDVI, population density and NDBI.</p>
</div>

<hr>
<p class="note">To add an article, create a Markdown file in <code>articles/</code> with front matter (<code>title</code>, <code>kind</code>, <code>authors</code>, <code>status</code>, <code>updated</code>, and <code>math: true</code> if it has equations), then add a card for it on this page. See <code>README.md</code>.</p>
