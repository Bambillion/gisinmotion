---
title: GIS in Motion
description: A learning library on land use / land cover classification with GIS, remote sensing and machine learning.
---

<p class="lede">A public notebook for my learning journey in <strong>land use and land cover (LULC) classification</strong>: how satellites, GIS layers and machine learning combine to map what covers the ground in fast-growing cities, and how to explain what the models learn.</p>

I started this site to keep everything in one place as I learn: the papers I am reading, the methods I am practising, and the articles I write along the way. One study I keep coming back to is a published case study I came across and am using as a reference: an explainable ensemble machine learning study of land cover in **Lugbe, Abuja, Nigeria** by Ekweozoh et al. (*Global Scientific Journal*, 2026). It combines Sentinel-2 imagery, terrain, population density and spectral indices with tree-ensemble classifiers and SHAP. It is not my work.

<div class="cards">
  <div class="card">
    <h3><a href="{{ '/reading-list/' | relative_url }}">Reading list</a></h3>
    <p>{{ site.data.papers.count }} arXiv papers grouped by topic, from Sentinel-2 classification and gradient boosting to SHAP, data fusion and African case studies.</p>
  </div>
  <div class="card">
    <h3><a href="{{ '/articles/' | relative_url }}">Articles</a></h3>
    <p>Methods write-ups and case studies, including a step-by-step framework for explainable multi-source LULC classification.</p>
  </div>
  <div class="card">
    <h3><a href="{{ '/books/' | relative_url }}">Books &amp; texts</a></h3>
    <p>Textbooks, free online books, classification standards and documentation worth keeping close.</p>
  </div>
</div>

## What I am learning

1. **Data**: Sentinel-2 multispectral imagery, elevation, slope and aspect, population density, and spectral indices (NDVI, NDBI, MNDWI, BSI, SAVI).
2. **Models**: Random Forest, XGBoost, LightGBM and CatBoost, tuned with cross-validated grid search.
3. **Evaluation**: confusion matrices, overall accuracy, kappa and F1, and why spatial validation matters.
4. **Explainability**: SHAP values for understanding which features drive each land cover class.
5. **Context**: what makes land cover mapping in African and other rapidly urbanising cities different.

## Start here

- New to the topic? Read the [methods framework article]({{ '/articles/explainable-multisource-lulc-framework/' | relative_url }}). It walks through a full workflow with the equations and a short Python appendix.
- Want a worked example? See my notes on the external [Lugbe, Abuja reference study]({{ '/articles/lugbe-case-study/' | relative_url }}) by Ekweozoh et al. (2026), with a summary checked against the published paper.
- Looking for sources? Browse the [reading list]({{ '/reading-list/' | relative_url }}) or download it as [BibTeX]({{ '/papers.bib' | relative_url }}).
