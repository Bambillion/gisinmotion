---
title: "Explainable Ensemble Machine Learning Approach for Land Cover Classification in Lugbe, Abuja, Nigeria"
kind: External reference study · Summary
authors: "Cheta Franklin Ekweozoh, Muawiyya Modibbo Musa, Olasehinde Michael Adejoro, Chukwunwike Patrick Nwokolo, Almakura Fatimah Ahmed, Idiat Idris Salaudeen"
status: "GSJ 14(8), August 2026"
updated: 2026-10-07
description: Notes on a published Lugbe, Abuja land cover study by Ekweozoh et al. (Global Scientific Journal, 2026), which uses Sentinel-2, terrain, population density, spectral indices, tree ensembles and SHAP.
---

<div class="note"><strong>About this study.</strong> This is a published case study I came across and am using as a reference while I learn. <strong>It is not my work.</strong><br>
<strong>Authors:</strong> Cheta Franklin Ekweozoh, Muawiyya Modibbo Musa, Olasehinde Michael Adejoro, Chukwunwike Patrick Nwokolo, Almakura Fatimah Ahmed, Idiat Idris Salaudeen.<br>
<strong>Published in:</strong> <em>Global Scientific Journal (GSJ)</em>, Volume 14, Issue 8, August 2026. Online ISSN 2320-9186. 32 pages (the PDF is numbered "Page 1 of 32" onwards, and no journal page range is printed).<br>
<strong>Links:</strong> <a href="https://www.globalscientificjournal.com/researchpaper/EXPLAINABLE_ENSEMBLE_MACHINE_LEARNING_APPROACH_FOR_LAND_COVER_CLASSIFICATION_IN_LUGBE_ABUJA_NIGERIA.pdf?id=247642">article page</a> · <a href="https://www.globalscientificjournal.com/researchpaper/EXPLAINABLE_ENSEMBLE_MACHINE_LEARNING_APPROACH_FOR_LAND_COVER_CLASSIFICATION_IN_LUGBE_ABUJA_NIGERIA.pdf">PDF</a><br>
I checked this summary against the published PDF. The version I first came across carried a different title, <em>"An Explainable Multi-Source Machine Learning Framework for Urban Land Cover Classification Using GIS and Remote Sensing Data: A Case Study of Lugbe, Abuja, Nigeria"</em>, but its abstract is the same as the published one. Page numbers below refer to the PDF.</div>

**Cite as:** Ekweozoh, C. F., Musa, M. M., Adejoro, O. M., Nwokolo, C. P., Ahmed, A. F., & Salaudeen, I. I. (2026). Explainable ensemble machine learning approach for land cover classification in Lugbe, Abuja, Nigeria. *Global Scientific Journal (GSJ)*, 14(8). ISSN 2320-9186.

## Abstract (as published)

Accurate land cover classification is crucial for the sustainability of urban planning, environment monitoring, and resources management within cities that are developing at a rapid pace. Traditional methods of land cover classification are known to fail in the representation of heterogeneous land covers and usually result in low interpretability of prediction results. The current paper introduces an explainable multi-source machine learning approach to land cover classification by means of GIS and remote sensing data, where the study site is Lugbe, Abuja, Nigeria. Multi-spectrum images from Sentinel-2 satellites were combined with topographical attributes (elevation, slope, and aspect), population density, and spectral indexes such as NDVI, NDBI, MNDWI, BSI, and SAVI. Four ensemble learners (Random Forest, XGBoost, LightGBM, and CatBoost) have been used together with GridSearchCV hyperparameter tuning. To facilitate the separation between the classes, spectrally similar vegetation classes were consolidated before training. In terms of validation accuracy, the optimized LightGBM model secured the highest value and is quite efficient when it comes to having the least training time. Using the SHapley Additive exPlanations (SHAP), the variables NDVI, population density, and NDBI were found to be highly correlated with the prediction results. With spatial prediction, the generated land cover map indicated vegetation to be the highest-class with 37.55%, built-up 25.52%, cropland 14.49%, trees 12.26%, and bare land 10.18%.

*Keywords (from the paper):* Explainable Artificial Intelligence, LightGBM, Land Cover Classification, Sentinel-2, SHAP, Urban Mapping, Lugbe.

## At a glance

| Item | Detail (from the paper) |
|---|---|
| Study area | Lugbe, Abuja Municipal Area Council (AMAC), FCT, on the Abuja–Airport Expressway; about 8.98°–9.05°N, 7.32°–7.41°E (p. 12) |
| Imagery | Sentinel-2 Level-2A, 2025, 10 m (pp. 12–13) |
| Terrain | SRTM DEM (30 m), giving elevation, slope and aspect |
| Population | WorldPop 2025 (100 m) |
| Other layers | JRC Global Surface Water (water occurrence); OpenStreetMap roads and buildings, used for visual interpretation and validation |
| Training labels | ESA WorldCover 2021 (10 m); vegetation-related classes grouped into five classes: Trees, Vegetation, Cropland, Built-up, Bare Land (p. 21) |
| Spectral indices | NDVI, NDBI, MNDWI, BSI, SAVI |
| Data split | 80% training, 20% testing (p. 16) |
| Models | Random Forest, XGBoost, LightGBM, CatBoost |
| Tuning | GridSearchCV with 5-fold cross-validation, applied to the selected LightGBM model |
| Explainability | Global SHAP feature importance |

## Model comparison (Table 7, p. 22)

| Model | Accuracy | Precision | Recall | F1 | Kappa | Time (s) |
|---|---|---|---|---|---|---|
| Random Forest | 0.593 | 0.605 | 0.593 | 0.591 | 0.467 | 9.11 |
| XGBoost | 0.596 | 0.604 | 0.596 | 0.594 | 0.471 | 8.91 |
| LightGBM | 0.595 | 0.603 | 0.595 | 0.594 | 0.471 | 2.36 |
| CatBoost | 0.594 | 0.605 | 0.594 | 0.592 | 0.469 | 5.10 |

The authors report no significant difference between the four learners (accuracy 59.3–59.6%). Untuned, **XGBoost** had the highest test accuracy (0.596). LightGBM had a similar score (F1 and kappa equal to XGBoost's in the table) and by far the shortest computation time (2.36 s), so the authors **selected LightGBM** for tuning and explanation. The abstract's statement that the optimised LightGBM "secured the highest value" of validation accuracy should be read in that context.

**Tuned LightGBM (Table 8, pp. 22–23):** learning rate 0.03, maximum depth 10, number of leaves 127, minimum child samples 20, number of trees 200. Cross-validation accuracy after tuning: **59.94%**.

## SHAP results (p. 24)

NDVI had the highest global SHAP importance (reported value 0.782), followed by **population density** and **NDBI**. The authors read this as vegetation properties and human activity being the main drivers of land cover differences in Lugbe.

## Land cover area statistics (Table 9, pp. 25–26)

| Class | Pixels | Area (ha) | Share |
|---|---|---|---|
| Vegetation | 562,715 | 5,627.15 | 37.55% |
| Built-up | 382,555 | 3,825.55 | 25.52% |
| Cropland | 217,097 | 2,170.97 | 14.49% |
| Trees | 183,806 | 1,838.06 | 12.26% |
| Bare land | 152,597 | 1,525.97 | 10.18% |
| **Total** | **1,498,770** | **14,987.70** | **100.00%** |

## Limitations stated by the authors (p. 29)

* The reference classes come from ESA WorldCover 2021, while the Sentinel-2 imagery is newer, so labels may be out of date in fast-changing areas.
* The open datasets have different spatial resolutions, which may affect some predictors.
* Merging similar classes into five broad classes improved classification accuracy but reduced thematic detail.

## My learning notes

These are my own observations, linked to the [methods framework article]({{ '/articles/explainable-multisource-lulc-framework/' | relative_url }}):

* The four models scored within 0.3 percentage points of each other. That makes training time a sensible tie-breaker, and it shows why it helps to report several metrics.
* The paper uses an 80/20 split and 5-fold cross-validation. The framework article discusses why spatially blocked splits give more conservative estimates for mapped data (Meyer et al., 2019).
* The labels come from an existing global product (WorldCover). This is a practical choice, but it means the model partly learns to reproduce that product.

Related reading: [explainable AI and SHAP]({{ '/reading-list/' | relative_url }}#xai), [Africa and Nigeria]({{ '/reading-list/' | relative_url }}#africa).
