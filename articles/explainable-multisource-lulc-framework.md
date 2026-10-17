---
title: "An Explainable Multi-Source Machine Learning Framework for Urban Land Cover Classification"
subtitle: "A methodological review and tutorial for Sentinel-2, GIS covariates, tree ensembles and SHAP, with reference to a case study in Lugbe, Abuja, Nigeria"
kind: Methods review · Tutorial
authors: Temitope Ogungbuji
status: Working draft
updated: 2026-10-07
math: true
description: A methods review and step-by-step framework for explainable multi-source land cover classification with Sentinel-2, terrain, population density, spectral indices, Random Forest, XGBoost, LightGBM, CatBoost and SHAP.
---

<div class="note"><strong>About this article.</strong> This is a methods review and tutorial. It reports <strong>no new experimental results</strong>. Every number about results comes from an external, published Lugbe, Abuja case study (Ekweozoh et al., 2026) and is labelled as such. Everything else is a placeholder in square brackets, for example <code>[OA = —]</code>, to be filled in by whoever applies the framework.</div>

<details class="toc" open markdown="1"><summary>Contents</summary>

* TOC
{:toc}

</details>

<div class="abstract" markdown="1">

## Abstract

Accurate land cover maps support urban planning, environmental monitoring and resource management, especially in cities that are growing quickly. Single-source, purely spectral classifiers often struggle with heterogeneous urban landscapes, and their predictions are hard to interpret. This article brings together recent literature and standard practice into a reproducible **framework for explainable, multi-source land cover classification** built on open data and open-source tools. The framework combines Sentinel-2 multispectral imagery with terrain attributes (elevation, slope, aspect), gridded population density, and five spectral indices (NDVI, NDBI, MNDWI, BSI and SAVI). It then compares four tree-ensemble learners (Random Forest, XGBoost, LightGBM and CatBoost) tuned with cross-validated grid search. We describe each step: preprocessing, index computation, feature stacking, consolidating spectrally similar classes, spatially aware validation, accuracy assessment (overall accuracy, kappa, F1 and the confusion matrix), area estimation, and model explanation with SHapley Additive exPlanations (SHAP). Throughout, we use a published external case study of Lugbe, Abuja, Nigeria (Ekweozoh et al., 2026) as a worked reference. In that study, the four ensembles performed almost identically (59.3–59.6% test accuracy), LightGBM was selected for its much shorter training time, SHAP ranked NDVI, population density and NDBI as the most influential features, and the predicted map was dominated by vegetation and built-up land. We close with common pitfalls (spatial autocorrelation, correlated features, resolution mismatch and kappa's limitations) and directions for future work.

**Keywords:** land use / land cover; Sentinel-2; spectral indices; multi-source data fusion; Random Forest; XGBoost; LightGBM; CatBoost; SHAP; explainable AI; urban remote sensing; Abuja; Nigeria

</div>

## 1. Introduction

Land use and land cover (LULC) maps describe what physically covers the Earth's surface (vegetation, water, built structures, bare soil) and how people use it. They are basic inputs for urban planning, flood and heat-risk assessment, agricultural monitoring and environmental reporting. In fast-growing cities, especially in Sub-Saharan Africa, land cover can change within a few years, faster than official maps are updated. The free, frequent, 10 m imagery from the Copernicus Sentinel-2 mission (Drusch et al., 2012) has made it practical to map these changes regularly.

Two developments have changed how such maps are made. The first is the shift from parametric classifiers to **machine learning**, in particular tree ensembles such as Random Forest (Breiman, 2001) and gradient-boosted decision trees: XGBoost (Chen & Guestrin, 2016), LightGBM (Ke et al., 2017) and CatBoost (Prokhorenkova et al., 2018). These models handle many correlated, non-linear and differently scaled predictors with little preprocessing. The second is the growing use of **multi-source inputs**: terrain, population and derived spectral indices sit alongside the raw reflectance bands, often improving accuracy when labels are scarce (Rao & Rolf, 2025; Gajardo et al., 2023).

These gains come at a cost to interpretability. A forest of hundreds of trees cannot be read like a decision rule, so users have little basis for trusting, debugging or learning from it. **Explainable AI (XAI)**, and SHAP in particular (Lundberg & Lee, 2017; Lundberg et al., 2019), addresses this by attributing each prediction to the input features. A recent systematic review shows that XAI is now spreading across remote sensing applications (Höhl et al., 2024).

This article sets out a complete, explainable, multi-source LULC workflow as a framework that can be reused. Its goals are to:

1. review the literature behind each design choice;
2. specify each processing step precisely, with equations;
3. provide reporting templates and a short Python implementation; and
4. use a published external Lugbe, Abuja case study (Ekweozoh et al., 2026) as a worked reference without inventing new results.

## 2. Background and related work

### 2.1 Supervised LULC classification from optical imagery

Supervised pixel-based classification of multispectral imagery remains the workhorse of LULC mapping. At continental scale, Venter and Sydenham (2021) produced a 10 m European land cover map (ELC10) with a Random Forest trained on LUCAS survey points in Google Earth Engine. They found that adding Sentinel-1 radar to Sentinel-2 improved accuracy, while atmospheric correction and speckle filtering changed little. At city scale, Tarek et al. (2026) mapped land use and vegetation change in Dhaka with Sentinel-2, Landsat 8 and spectral indices, and Random Forest outperformed Decision Tree and KNN. Deep learning approaches, reviewed by Zhu et al. (2017), can learn spatial and temporal features directly; Mazzia et al. (2020), for example, used multi-temporal Sentinel-2 data with a recurrent-convolutional network and compared it against SVM and Random Forest baselines. For tabular, per-pixel feature stacks like the one in this framework, however, tree ensembles remain strong and cheap baselines.

Benchmark datasets make methods comparable: EuroSAT (Helber et al., 2017) offers 27,000 labelled Sentinel-2 patches across 10 classes, BigEarthNet (Sumbul et al., 2019) offers 590,326 multi-label patches labelled from CORINE 2018, and SEN12MS (Schmitt et al., 2019) pairs Sentinel-1, Sentinel-2 and MODIS land cover patches for fusion research. These datasets are mostly European or global in coverage, so local reference data are still needed for African cities.

### 2.2 Tree ensembles and gradient boosting

Random Forest averages many decorrelated decision trees, each grown on a bootstrap sample with a random subset of features considered at each split (Breiman, 2001). Gradient boosting instead builds trees sequentially, each one correcting the errors of the current ensemble. XGBoost added a regularised objective, sparsity-aware split finding and a weighted quantile sketch for scalability (Chen & Guestrin, 2016). LightGBM speeds up training with histogram-based, leaf-wise tree growth, Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB) (Ke et al., 2017). CatBoost introduced ordered boosting and ordered target statistics to remove the prediction shift caused by target leakage, and it handles categorical inputs natively (Prokhorenkova et al., 2018). In remote sensing, Pande (2023) built ensembles of Random Forests with random rotations on fused Sentinel-1/2 data. On the UCI urban land cover dataset, Tabasum et al. (2026) found that tree ensembles, including XGBoost and CatBoost, remain strong baselines against tabular deep learning.

### 2.3 Explainability in remote sensing

SHAP assigns each feature an additive contribution to a single prediction, based on Shapley values from cooperative game theory, and unifies several earlier attribution methods (Lundberg & Lee, 2017). TreeSHAP computes exact SHAP values for tree ensembles in polynomial time and adds interaction values and global summaries (Lundberg et al., 2019). This makes it the natural choice for the four models in this framework. Höhl et al. (2024) review XAI in remote sensing and highlight open problems in evaluating explanations. Alternatives include counterfactual explanations for land cover time series (Dantas et al., 2023). In urban analysis, feature-importance analysis of Random Forest and XGBoost models has been used to link urban form and land cover to environmental hazard exposure (Ho et al., 2023).

### 2.4 Multi-source inputs and spectral indices

Spectral indices condense physically meaningful contrasts into single features. NDVI (Tucker, 1979) and SAVI (Huete, 1988) capture vegetation vigour, the latter correcting for soil background. NDBI was designed to highlight built-up areas (Zha et al., 2003), MNDWI improves open-water detection in built-up surroundings (Xu, 2006), and the bare soil index (BSI) combines SWIR, red, NIR and blue to separate exposed soil (Rikimaru et al., 2002). Urban climate studies routinely use these indices as proxies for vegetation and urban density (Alavi Soltani et al., 2026). Beyond spectra, adding non-optical geographic layers such as elevation can improve data efficiency and out-of-region generalisation (Rao & Rolf, 2025). In Nigeria, Gajardo et al. (2023) found that Sentinel-1, climate and topographic inputs were critical for country-scale cropland mapping. Fusing radar texture with optical bands is another route (Pande, 2023).

### 2.5 Mapping African and fast-growing cities

Global products are not always reliable locally. Kerner et al. (2024) compared 11 public land cover maps against reference data from eight Sub-Saharan African countries and found large differences in how well they captured cropland. Kakooei et al. (2024) mapped urban and rural settlements across Africa at 10 m by fusing Landsat-8, night-time lights and existing LULC layers. For Abuja specifically, Nwilo et al. (2022) assessed the 20 m SPOT DEM against ground control points in the FCT and Lagos. They found the lowest accuracy over built-up areas, which matters when elevation and slope are used as features. Gajardo et al. (2023) also found that local Nigerian training data outperformed larger global datasets.

### 2.6 Validation pitfalls

Spatial data violate the independence assumption behind random cross-validation. Meyer et al. (2019) show that random k-fold validation gives over-optimistic accuracy, and that location-like predictors can cause models to reproduce the training data rather than generalise. Reference labels are also uncertain: Tsutsumida and Kato (2024) document substantial variation between different people labelling the same land cover. Accuracy reporting itself has known weaknesses. Pontius and Millones (2011) argue that kappa should be replaced by quantity and allocation disagreement, and Olofsson et al. (2014) set out good practice for unbiased area estimation from a sample-based error matrix.

## 3. Reference case study, study area and data sources

### 3.1 The Lugbe, Abuja reference case (external study)

The framework is illustrated with an external, published case study: Ekweozoh et al. (2026), *Explainable Ensemble Machine Learning Approach for Land Cover Classification in Lugbe, Abuja, Nigeria*, *Global Scientific Journal (GSJ)* 14(8) (see the [case-study summary]({{ '/articles/lugbe-case-study/' | relative_url }})). A version of its abstract also circulated under a slightly different title, *An Explainable Multi-Source Machine Learning Framework for Urban Land Cover Classification Using GIS and Remote Sensing Data: A Case Study of Lugbe, Abuja, Nigeria*. Lugbe is a fast-growing peri-urban district of Abuja in Nigeria's Federal Capital Territory (FCT). It is a good test of the framework because rapid residential expansion there mixes built-up land, cropland, grassland, trees and exposed soil at fine spatial scales. That study is not the work of this article's author. It used Sentinel-2 L2A imagery (2025), an SRTM DEM, WorldPop 2025 population and a JRC surface-water layer, with ESA WorldCover 2021 as training labels. The figures quoted from it in this article were checked against the published PDF.

### 3.2 Data sources

| Layer | Example source | Native resolution | Role |
|---|---|---|---|
| Multispectral reflectance (B2, B3, B4, B8, B11, B12) | Sentinel-2 MSI Level-2A ([ESA SentiWiki](https://sentiwiki.copernicus.eu/web/s2-mission)) | 10 m (B2–B4, B8); 20 m (B11, B12) | Primary spectral evidence |
| Spectral indices (NDVI, NDBI, MNDWI, BSI, SAVI) | Derived from Sentinel-2 | 10–20 m | Physically meaningful contrasts |
| Elevation, slope, aspect | [Copernicus DEM](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM) or national 20 m SPOT DEM (Nwilo et al., 2022) | 30 m / 20 m | Terrain context |
| Population density | [WorldPop](https://www.worldpop.org/) gridded population | ~100 m | Socio-demographic context |
| Reference labels | Field GPS points, visual interpretation of very-high-resolution imagery | Points / polygons | Training and validation |

Table 1. Typical data layers for the framework. Resolutions are nominal; check the product documentation.

## 4. Methodology

### 4.1 Workflow overview

1. Acquire Sentinel-2 L2A scenes for the season with the least cloud, plus the DEM, population and reference data.
2. Mask clouds and shadows, build a median composite, and resample everything onto one 10 m grid in a projected CRS (for Abuja, UTM zone 32N, EPSG:32632).
3. Compute spectral indices and terrain derivatives, then stack them into a feature cube.
4. Define the class legend and merge classes that cannot be separated spectrally.
5. Sample labelled pixels, assign spatial blocks, and split the data into training and test sets by block.
6. Train Random Forest, XGBoost, LightGBM and CatBoost, tuning each with grid search under spatial cross-validation.
7. Assess accuracy on the held-out test set and estimate class areas.
8. Explain the best model with TreeSHAP, then predict and map the full study area.

### 4.2 Preprocessing

Use Level-2A (bottom-of-atmosphere) products. For scenes from processing baseline 04.00 onward, convert digital numbers to reflectance with the product's additive offset, $$\rho = (DN + \text{offset})/10\,000$$, where the offset is given in the metadata. Mask clouds, cloud shadows and saturated pixels using the Scene Classification Layer (SCL) and take a per-pixel median over the season. In Abuja's climate, dry-season scenes are usually much less cloudy. Resample the 20 m SWIR bands to 10 m with bilinear interpolation. Resample continuous auxiliary layers (DEM, population) with bilinear interpolation and categorical layers with nearest neighbour. Keep all layers co-registered in a single projected CRS so that slope and area calculations use metres.

### 4.3 Spectral indices

Let $$B$$, $$G$$, $$R$$, $$N$$ and $$S_1$$ denote surface reflectance in Sentinel-2 bands B2 (blue), B3 (green), B4 (red), B8 (NIR) and B11 (SWIR-1).

**Normalised Difference Vegetation Index** (Tucker, 1979):

$$
\mathrm{NDVI} = \frac{N - R}{N + R}
$$

**Normalised Difference Built-up Index** (Zha et al., 2003):

$$
\mathrm{NDBI} = \frac{S_1 - N}{S_1 + N}
$$

**Modified Normalised Difference Water Index** (Xu, 2006):

$$
\mathrm{MNDWI} = \frac{G - S_1}{G + S_1}
$$

**Bare Soil Index** (after Rikimaru et al., 2002; normalised form):

$$
\mathrm{BSI} = \frac{(S_1 + R) - (N + B)}{(S_1 + R) + (N + B)}
$$

**Soil-Adjusted Vegetation Index** (Huete, 1988), with soil-brightness factor $$L$$ (commonly $$L = 0.5$$):

$$
\mathrm{SAVI} = \frac{(1 + L)\,(N - R)}{N + R + L}
$$

All indices except SAVI are bounded in $$[-1, 1]$$. SAVI assumes reflectance scaled to $$[0, 1]$$. NDVI and SAVI are strongly correlated, and so are NDBI and BSI. This hardly affects tree ensembles' accuracy, but it does affect how SHAP credit is shared out (Section 4.10).

### 4.4 Terrain and population covariates

Slope $$\beta$$ and aspect $$\alpha$$ come from the DEM gradient. With $$\partial z/\partial x$$ eastward and $$\partial z/\partial y$$ northward,

$$
\beta = \arctan\sqrt{\left(\frac{\partial z}{\partial x}\right)^2 + \left(\frac{\partial z}{\partial y}\right)^2},
\qquad
\alpha = \operatorname{atan2}\!\left(-\frac{\partial z}{\partial x},\; -\frac{\partial z}{\partial y}\right) \bmod 360^\circ .
$$

Aspect is circular (1° and 359° are neighbours), so encode it as *eastness* $$\sin\alpha$$ and *northness* $$\cos\alpha$$ rather than as raw degrees. Population density from a roughly 100 m grid is much coarser than the 10 m imagery. Treat it as neighbourhood context rather than a pixel attribute, and note that gridded population products are themselves modelled from covariates that can include settlement or land cover layers (check the product documentation). This creates a risk of circularity (Section 6).

### 4.5 Feature stacking

At each valid pixel $$i$$, the feature vector is

$$
\begin{aligned}
\mathbf{x}_i = \big[\,&\rho_{B2}, \rho_{B3}, \rho_{B4}, \rho_{B8}, \rho_{B11}, \rho_{B12},\ \mathrm{NDVI}, \mathrm{NDBI}, \mathrm{MNDWI}, \mathrm{BSI}, \mathrm{SAVI}, \\
&z, \beta, \sin\alpha, \cos\alpha,\ \mathrm{pop}\,\big]^\top \in \mathbb{R}^{16}.
\end{aligned}
$$

Tree ensembles do not need features to be scaled. Do record the stacking order, though, because SHAP outputs and saved models depend on it. Pixels with any missing value (cloud gaps, DEM voids) are left out of training and marked as no-data in the output map.

### 4.6 Class legend and consolidation

Start from a legend suited to the purpose, for example Anderson et al.'s (1976) Level I classes adapted to the local context, such as built-up, vegetation (grass and shrub), cropland, trees, bare land and water. Classes that are spectrally similar at 10 m, such as grassland, shrubland and fallow vegetation, often have overlapping feature distributions and are confused in a pilot model. Two common diagnostics are:

* **Pilot confusion:** train a quick model and merge pairs whose mutual confusion is a large share of their reference totals.
* **Separability:** compute a pairwise separability measure, such as the Jeffries–Matusita distance $$JM = 2\,(1 - e^{-D_B})$$, where $$D_B$$ is the Bhattacharyya distance between class distributions, and merge pairs that fall below a chosen threshold.

The Lugbe case study (Ekweozoh et al., 2026) grouped vegetation-related ESA WorldCover classes into five classes (Trees, Vegetation, Cropland, Built-up, Bare Land) before training. Record any merge in a mapping table (old class → new class) so that it can be reproduced.

### 4.7 Reference sampling and spatial splitting

Collect reference samples by stratified random sampling within a provisional map, with enough samples in rare classes, for example by field GPS or visual interpretation of very-high-resolution imagery. Several people interpreting the same imagery introduces label variance (Tsutsumida & Kato, 2024), so document the labelling protocol and, if possible, double-label a subset. Because neighbouring pixels are strongly autocorrelated, **split by spatial block rather than by random pixel**: assign each sample to a grid cell (for example 500 m × 500 m) and keep whole blocks in either training or test (Meyer et al., 2019). Use the same grouping inside cross-validation (`GroupKFold`).

### 4.8 Classifiers

All four models are ensembles of decision trees. For $$K$$ classes, the boosted models fit one additive score function per class and convert scores to probabilities with a softmax, $$p_k(\mathbf{x}) = e^{F_k(\mathbf{x})}/\sum_{j} e^{F_j(\mathbf{x})}$$.

**Random Forest** (Breiman, 2001). Grow $$T$$ trees on bootstrap samples, each split choosing from a random subset of $$m_{\text{try}}$$ features, and predict by majority vote, $$\hat{y} = \operatorname{mode}\{h_t(\mathbf{x})\}_{t=1}^{T}$$. Out-of-bag samples give a free (but spatially naive) error estimate.

**XGBoost** (Chen & Guestrin, 2016). Minimise a regularised objective over additive trees $$f_k$$,

$$
\mathcal{L} = \sum_{i} \ell\big(y_i, \hat{y}_i\big) + \sum_{k} \Omega(f_k),
\qquad
\Omega(f) = \gamma T_f + \tfrac{1}{2}\lambda \lVert \mathbf{w}_f \rVert^2 ,
$$

where $$T_f$$ is the number of leaves and $$\mathbf{w}_f$$ the leaf weights. Optimisation uses a second-order Taylor expansion of the loss.

**LightGBM** (Ke et al., 2017). This is gradient boosting with histogram-binned features and leaf-wise (best-first) tree growth, which is controlled mainly by `num_leaves`. GOSS keeps all large-gradient samples and subsamples small-gradient ones, and EFB bundles mutually exclusive sparse features. Together these usually make it the fastest of the four to train.

**CatBoost** (Prokhorenkova et al., 2018). This is gradient boosting with ordered boosting and ordered target statistics to avoid target leakage, using symmetric (oblivious) trees. It handles categorical covariates, such as a soil or zoning class, without one-hot encoding.

### 4.9 Hyperparameter tuning

Tune each model with exhaustive grid search under grouped (spatial) k-fold cross-validation, using scikit-learn's `GridSearchCV` (Pedregosa et al., 2011) with `GroupKFold`. Macro-averaged F1 is a sensible selection metric when classes are imbalanced. The grids below are **illustrative starting points, not recommended or tested values**.

| Model | Parameters searched (illustrative) |
|---|---|
| Random Forest | `n_estimators` ∈ {200, 500}; `max_features` ∈ {sqrt, 0.3}; `min_samples_leaf` ∈ {1, 5} |
| XGBoost | `n_estimators` ∈ {200, 400}; `max_depth` ∈ {4, 6, 8}; `learning_rate` ∈ {0.05, 0.1}; `subsample` ∈ {0.8, 1.0} |
| LightGBM | `num_leaves` ∈ {15, 31, 63}; `learning_rate` ∈ {0.05, 0.1}; `n_estimators` ∈ {200, 400}; `min_child_samples` ∈ {20, 50} |
| CatBoost | `depth` ∈ {4, 6, 8}; `learning_rate` ∈ {0.05, 0.1}; `iterations` ∈ {300, 600}; `l2_leaf_reg` ∈ {1, 3, 5} |

Table 2. Example search spaces. Record the wall-clock training time of each refit on the same hardware so that efficiency can be compared fairly.

### 4.10 Accuracy assessment

Let $$n_{ij}$$ be the number of test samples mapped as class $$i$$ (rows) whose reference class is $$j$$ (columns), with row totals $$n_{i+}$$, column totals $$n_{+j}$$ and grand total $$N$$. This table is the **confusion (error) matrix**.

$$
\mathrm{OA} = \frac{1}{N}\sum_{k=1}^{K} n_{kk},
\qquad
\mathrm{UA}_k = \frac{n_{kk}}{n_{k+}},
\qquad
\mathrm{PA}_k = \frac{n_{kk}}{n_{+k}}
$$

User's accuracy (UA, 1 − commission error) is the precision of class $$k$$, and producer's accuracy (PA, 1 − omission error) is its recall. Per-class F1 and macro-F1 are

$$
F1_k = \frac{2\,\mathrm{UA}_k\,\mathrm{PA}_k}{\mathrm{UA}_k + \mathrm{PA}_k},
\qquad
\text{macro-}F1 = \frac{1}{K}\sum_{k} F1_k .
$$

Cohen's kappa compares observed agreement $$p_o = \mathrm{OA}$$ with chance agreement $$p_e$$:

$$
\kappa = \frac{p_o - p_e}{1 - p_e},
\qquad
p_e = \sum_{k=1}^{K} \frac{n_{k+}\, n_{+k}}{N^2}.
$$

Kappa is still widely reported, but Pontius and Millones (2011) show that it is hard to interpret and recommend reporting quantity and allocation disagreement instead, so report both. Because pixel counting gives biased area estimates when the map has errors, follow Olofsson et al. (2014). Estimate the proportion of area in reference class $$j$$ from a stratified sample as

$$
\hat{p}_{\cdot j} = \sum_{i=1}^{K} W_i \,\frac{n_{ij}}{n_{i+}},
$$

where $$W_i$$ is the mapped area proportion of class $$i$$, and report confidence intervals.

### 4.11 SHAP explainability

For a model $$f$$, feature set $$F$$ and instance $$\mathbf{x}$$, the SHAP value of feature $$j$$ is its Shapley value (Lundberg & Lee, 2017):

$$
\phi_j(\mathbf{x}) = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!\,\big(|F| - |S| - 1\big)!}{|F|!}\Big[f_{S \cup \{j\}}\big(\mathbf{x}_{S \cup \{j\}}\big) - f_S\big(\mathbf{x}_S\big)\Big],
$$

and the explanation is additive: $$f(\mathbf{x}) = \phi_0 + \sum_{j} \phi_j(\mathbf{x})$$, where $$\phi_0$$ is the expected model output. For multiclass tree models, TreeSHAP (Lundberg et al., 2019) returns one set of values per class, usually in raw score (log-odds) space. Global importance is the mean absolute SHAP value,

$$
I_j = \frac{1}{N}\sum_{i=1}^{N} \big|\phi_j(\mathbf{x}_i)\big| ,
$$

which can be averaged over classes for an overall ranking or kept per class (as a beeswarm plot for each class) to see, for example, what drives *built-up* rather than *bare land*. Three cautions apply:

1. SHAP explains the **model**, not causal processes in the landscape.
2. Correlated features (NDVI and SAVI, NDBI and BSI) share or swap credit, so interpret groups of related features together.
3. Compute SHAP on a representative held-out sample, not only on training pixels.

### 4.12 Map production and area statistics

Apply the selected model to every valid pixel, write the class raster as a GeoTIFF, and report mapped area per class. For 10 m pixels, $$A_k = n_k \times 100\ \mathrm{m^2}$$, and the share is $$100\,n_k / \sum_j n_j$$ %. Where a stratified accuracy sample exists, also report the error-adjusted areas from Section 4.10.

## 5. Results template and reference values

### 5.1 Model comparison (template)

| Model | OA | Kappa | Macro-F1 | Training time |
|---|---|---|---|---|
| Random Forest | [—] | [—] | [—] | [—] |
| XGBoost | [—] | [—] | [—] | [—] |
| LightGBM | [—] | [—] | [—] | [—] |
| CatBoost | [—] | [—] | [—] | [—] |

Table 3. Placeholders to fill with held-out (spatial block) test results. Report the confusion matrix of the selected model alongside this table.

### 5.2 What the Lugbe reference study reports

The following come from the published Lugbe, Abuja case study (Ekweozoh et al., 2026; page numbers refer to its PDF). They are **not** results of this article:

* **Model comparison (Table 7, p. 22):** test accuracy was 0.593 for Random Forest, 0.596 for XGBoost, 0.595 for LightGBM and 0.594 for CatBoost, with kappa between 0.467 and 0.471. Training times were 9.11 s, 8.91 s, 2.36 s and 5.10 s respectively. XGBoost had the marginally highest untuned accuracy, and LightGBM was selected for tuning because it performed similarly at a fraction of the training time.
* **Tuned LightGBM (pp. 22–23):** GridSearchCV with 5-fold cross-validation reached 59.94% cross-validation accuracy.
* **Most influential features (SHAP, p. 24):** NDVI, then population density and NDBI.
* **Validation design:** a random 80/20 train/test split (p. 16), not a spatially blocked one (compare Section 4.7).
* **Mapped land cover shares:** vegetation 37.55%, built-up 25.52%, cropland 14.49%, trees 12.26%, bare land 10.18%.

| Class | Share of mapped area (Lugbe reference study) |
|---|---|
| Vegetation | 37.55% |
| Built-up | 25.52% |
| Cropland | 14.49% |
| Trees | 12.26% |
| Bare land | 10.18% |

Table 4. Land cover shares reported in the Lugbe, Abuja case study (Ekweozoh et al., 2026, Table 9, pp. 25–26). The total mapped area is 14,987.70 ha.

### 5.3 SHAP reporting template

* Global bar chart of mean |SHAP| over all classes: [figure].
* Beeswarm plot for each class (at least built-up, vegetation and bare land): [figures].
* Dependence plots for the top three features: [figures].

## 6. Discussion

**Why these features matter.** A SHAP ranking led by NDVI, population density and NDBI, as in the Lugbe study (Ekweozoh et al., 2026), makes physical sense for a peri-urban landscape. NDVI separates green cover from everything else, NDBI targets the SWIR-bright impervious surfaces typical of built-up land, and population density adds human-settlement context that spectra alone can miss where roofing materials resemble bare soil. Similar index-based reasoning about vegetation and built-up density appears in urban climate work (Alavi Soltani et al., 2026) and in other rapidly growing cities (Tarek et al., 2026).

**Why LightGBM is often fast.** Histogram binning, leaf-wise growth, GOSS and EFB reduce the cost of finding splits (Ke et al., 2017), so LightGBM usually trains fastest on medium-sized tabular stacks. Its accuracy advantage, where there is one, is usually small and depends on the data and the tuning. Comparisons are only fair if all models get similar tuning budgets and are evaluated on the same spatial test blocks (Meyer et al., 2019).

**Population density: helpful but risky.** Population is a powerful predictor of built-up land, but gridded population products may themselves be modelled from settlement or land cover covariates. Then the model partly learns one map from another rather than from independent evidence. Population data are also coarse (about 100 m), so they can blur class boundaries. Running the model with and without this layer is a simple robustness check.

**Explanations are not ground truth.** SHAP values describe how the fitted model uses features. With correlated indices, credit can move between NDVI and SAVI, or between NDBI and BSI, from one run to the next without any change in accuracy (Lundberg & Lee, 2017; Höhl et al., 2024). Report grouped importance where possible.

**Local data matter.** Evidence from Nigeria (Gajardo et al., 2023) and Sub-Saharan Africa more broadly (Kerner et al., 2024) shows that global products and global training sets can perform poorly locally. City-scale studies with carefully collected local reference data, like the Lugbe case, fill an important gap.

## 7. Limitations

* **No new experiments.** This article synthesises methods. It does not test the framework on new data, and the illustrative grids are not tuned recommendations.
* **Per-pixel classification.** Pixel-based models ignore texture and spatial context. Object-based or convolutional approaches (Zhu et al., 2017) may separate classes like informal settlements and bare land better.
* **Single-date composites.** A single seasonal composite misses phenology that separates cropland from natural vegetation. Multi-temporal features help (Mazzia et al., 2020).
* **Optical only.** Cloud cover limits optical acquisitions in the wet season. Adding Sentinel-1 SAR is a proven remedy (Venter & Sydenham, 2021; Pande, 2023).
* **Reference data.** Label noise and interpreter variance limit both training and validation (Tsutsumida & Kato, 2024).
* **DEM accuracy in built-up areas.** DEM errors are largest over buildings (Nwilo et al., 2022), which can create spurious slope signals in dense urban areas.

## 8. Future work

1. Add Sentinel-1 backscatter and texture to make the workflow robust to clouds and to separate built-up land from bare soil.
2. Use multi-temporal features (seasonal medians, NDVI amplitude) to separate cropland from natural vegetation.
3. Use spatial cross-validation with buffered blocks, and test transfer to a neighbouring district.
4. Explain grouped features (for example all vegetation indices together) and check SHAP rankings for stability across seeds and folds.
5. Compare against deep tabular or convolutional models (Tabasum et al., 2026; Zhu et al., 2017) and against counterfactual explanations (Dantas et al., 2023).
6. Estimate areas with uncertainty using a design-based approach (Olofsson et al., 2014) and track change over several years.
7. Pre-train or benchmark on EuroSAT, BigEarthNet or SEN12MS before fine-tuning on local labels (Helber et al., 2017; Sumbul et al., 2019; Schmitt et al., 2019).

## 9. Conclusion

Explainable, multi-source machine learning offers a practical and transparent way to map land cover in fast-growing cities. Pairing open Sentinel-2 imagery with terrain, population and well-understood spectral indices, comparing tuned tree ensembles under spatial validation, and explaining the chosen model with TreeSHAP yields a map along with an account of *why* the map looks as it does. The Lugbe, Abuja case study shows the approach in practice. The checklists, equations and code here are meant to make the next study easier to run, report and reproduce.

## References

**arXiv papers** (see the [reading list]({{ '/reading-list/' | relative_url }}))

* Alavi Soltani, S., Niazmardi, S., & Esmaeily, A. (2026). Causal evaluation of contributing factors to urban heat island. arXiv:2605.24459. <https://arxiv.org/abs/2605.24459>
* Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proc. KDD 2016*. arXiv:1603.02754. <https://arxiv.org/abs/1603.02754>
* Dantas, C. F., Marcos, D., & Ienco, D. (2023). Towards explainable land cover mapping: A counterfactual-based strategy. arXiv:2301.01520. <https://arxiv.org/abs/2301.01520>
* Gajardo, J., Volpi, M., Onwude, D., & Defraeye, T. (2023). Evaluating the role of training data origin for country-scale cropland mapping in data-scarce regions: A case study of Nigeria. arXiv:2312.10872 (published in *ISPRS Open J. Photogramm. Remote Sens.* 17, 2025). <https://arxiv.org/abs/2312.10872>
* Helber, P., Bischke, B., Dengel, A., & Borth, D. (2017). EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification. arXiv:1709.00029. <https://arxiv.org/abs/1709.00029>
* Ho, Y.-H., Liu, Z., Lee, C.-C., & Mostafavi, A. (2023). ML4EJ: Decoding the role of urban features in shaping environmental injustice using interpretable machine learning. arXiv:2310.02476. <https://arxiv.org/abs/2310.02476>
* Höhl, A., Obadic, I., Fernández Torres, M. Á., et al. (2024). Opening the black-box: A systematic review on explainable AI in remote sensing. *IEEE GRSM* 12(4). arXiv:2402.13791. <https://arxiv.org/abs/2402.13791>
* Kakooei, M., Bailie, J., Pettersson, M. B., et al. (2024). A high resolution urban and rural settlement map of Africa using deep learning and satellite imagery. arXiv:2411.02935. <https://arxiv.org/abs/2411.02935>
* Kerner, H., Nakalembe, C., Yang, A., et al. (2024). How accurate are existing land cover maps for agriculture in Sub-Saharan Africa? *Scientific Data* 11, 486. arXiv:2307.02575. <https://arxiv.org/abs/2307.02575>
* Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 2017*. arXiv:1705.07874. <https://arxiv.org/abs/1705.07874>
* Lundberg, S. M., Erion, G., Chen, H., et al. (2019). Explainable AI for trees: From local explanations to global understanding. arXiv:1905.04610. <https://arxiv.org/abs/1905.04610>
* Mazzia, V., Khaliq, A., & Chiaberge, M. (2020). Improvement in land cover and crop classification based on temporal features learning from Sentinel-2 data using recurrent-convolutional neural network (R-CNN). *Applied Sciences* 10(1), 238. arXiv:2004.12880. <https://arxiv.org/abs/2004.12880>
* Meyer, H., Reudenbach, C., Wöllauer, S., & Nauss, T. (2019). Importance of spatial predictor variable selection in machine learning applications. *Ecological Modelling* 411, 108815. arXiv:1908.07805. <https://arxiv.org/abs/1908.07805>
* Nwilo, P., Onyegbula, J., Okolie, C., et al. (2022). Quality assessment of the 20m SPOT DEM in Nigeria. *Applied Geomatics* 14. arXiv:2205.02946. <https://arxiv.org/abs/2205.02946>
* Pande, S. (2023). Land use/land cover classification of fused Sentinel-1 and Sentinel-2 imageries using ensembles of Random Forests. arXiv:2312.10798. <https://arxiv.org/abs/2312.10798>
* Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR* 12, 2825–2830. arXiv:1201.0490. <https://arxiv.org/abs/1201.0490>
* Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: Unbiased boosting with categorical features. *NeurIPS 2018*. arXiv:1706.09516. <https://arxiv.org/abs/1706.09516>
* Rao, A., & Rolf, E. (2025). Using multiple input modalities can improve data-efficiency and O.O.D. generalization for ML with satellite imagery. arXiv:2507.13385. <https://arxiv.org/abs/2507.13385>
* Schmitt, M., Hughes, L. H., Qiu, C., & Zhu, X. X. (2019). SEN12MS: A curated dataset of georeferenced multi-spectral Sentinel-1/2 imagery for deep learning and data fusion. arXiv:1906.07789. <https://arxiv.org/abs/1906.07789>
* Sumbul, G., Charfuelan, M., Demir, B., & Markl, V. (2019). BigEarthNet: A large-scale benchmark archive for remote sensing image understanding. *IGARSS 2019*. arXiv:1902.06148. <https://arxiv.org/abs/1902.06148>
* Tabasum, M., Tasnim, T., Islam, M. E., & Habib, A. Z. S. B. (2026). Tabular deep learning vs classical machine learning for urban land cover classification. arXiv:2609.19010. <https://arxiv.org/abs/2609.19010>
* Tarek, M. M., Hossain, M. A., Islam, M. S., & Kanchan, M. H. (2026). Remote sensing and machine learning-based analysis of land use and vegetation change in Dhaka District, Bangladesh. arXiv:2608.12001. <https://arxiv.org/abs/2608.12001>
* Tsutsumida, N., & Kato, A. (2024). A Bayesian inference approach for reducing inter-investigator variability in sampling-based land cover classification. arXiv:2403.15720. <https://arxiv.org/abs/2403.15720>
* Venter, Z. S., & Sydenham, M. A. K. (2021). Continental-scale land cover mapping at 10 m resolution over Europe (ELC10). *Remote Sensing* 13, 2301. arXiv:2104.10922. <https://arxiv.org/abs/2104.10922>
* Zhu, X. X., Tuia, D., Mou, L., et al. (2017). Deep learning in remote sensing: A review. *IEEE GRSM*. arXiv:1710.03959. <https://arxiv.org/abs/1710.03959>

**Other sources**

* Anderson, J. R., Hardy, E. E., Roach, J. T., & Witmer, R. E. (1976). *A land use and land cover classification system for use with remote sensor data*. USGS Professional Paper 964. <https://doi.org/10.3133/pp964>
* Breiman, L. (2001). Random forests. *Machine Learning* 45(1), 5–32. <https://doi.org/10.1023/A:1010933404324>
* Drusch, M., et al. (2012). Sentinel-2: ESA's optical high-resolution mission for GMES operational services. *Remote Sensing of Environment* 120, 25–36. <https://doi.org/10.1016/j.rse.2011.11.026>
* Ekweozoh, C. F., Musa, M. M., Adejoro, O. M., Nwokolo, C. P., Ahmed, A. F., & Salaudeen, I. I. (2026). Explainable ensemble machine learning approach for land cover classification in Lugbe, Abuja, Nigeria. *Global Scientific Journal (GSJ)*, 14(8). ISSN 2320-9186. <https://www.globalscientificjournal.com/researchpaper/EXPLAINABLE_ENSEMBLE_MACHINE_LEARNING_APPROACH_FOR_LAND_COVER_CLASSIFICATION_IN_LUGBE_ABUJA_NIGERIA.pdf> (An abstract of this study also circulated under the title *An Explainable Multi-Source Machine Learning Framework for Urban Land Cover Classification Using GIS and Remote Sensing Data: A Case Study of Lugbe, Abuja, Nigeria*.)
* Huete, A. R. (1988). A soil-adjusted vegetation index (SAVI). *Remote Sensing of Environment* 25(3), 295–309. <https://doi.org/10.1016/0034-4257(88)90106-X>
* Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems 30*. <https://papers.nips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html>
* Olofsson, P., Foody, G. M., Herold, M., Stehman, S. V., Woodcock, C. E., & Wulder, M. A. (2014). Good practices for estimating area and assessing accuracy of land change. *Remote Sensing of Environment* 148, 42–57. <https://doi.org/10.1016/j.rse.2014.02.015>
* Pontius, R. G., Jr., & Millones, M. (2011). Death to Kappa: Birth of quantity disagreement and allocation disagreement for accuracy assessment. *International Journal of Remote Sensing* 32(15), 4407–4429. <https://doi.org/10.1080/01431161.2011.552923>
* Rikimaru, A., Roy, P. S., & Miyatake, S. (2002). Tropical forest cover density mapping. *Tropical Ecology* 43(1), 39–47.
* Tucker, C. J. (1979). Red and photographic infrared linear combinations for monitoring vegetation. *Remote Sensing of Environment* 8(2), 127–150. <https://doi.org/10.1016/0034-4257(79)90013-0>
* Xu, H. (2006). Modification of normalised difference water index (NDWI) to enhance open water features in remotely sensed imagery. *International Journal of Remote Sensing* 27(14), 3025–3033. <https://doi.org/10.1080/01431160600589179>
* Zha, Y., Gao, J., & Ni, S. (2003). Use of normalized difference built-up index in automatically mapping urban areas from TM imagery. *International Journal of Remote Sensing* 24(3), 583–594. <https://doi.org/10.1080/01431160304987>
* ESA. Sentinel-2 mission documentation (SentiWiki). <https://sentiwiki.copernicus.eu/web/s2-mission>

## Appendix A. Python code (optional)

The excerpt below shows the core steps with `numpy`, `rasterio`, `scikit-learn`, `lightgbm` and `shap`. The full script, including GeoTIFF I/O, map prediction, area statistics and a runnable demo on **synthetic** data, is at [`code/lulc_pipeline.py`]({{ '/code/lulc_pipeline.py' | relative_url }}).

```python
import numpy as np
import lightgbm as lgb
import shap
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold

EPS = 1e-6
FEATURES = ["B2", "B3", "B4", "B8", "B11", "B12",
            "NDVI", "NDBI", "MNDWI", "BSI", "SAVI",
            "elevation", "slope", "aspect_sin", "aspect_cos", "pop_density"]



def spectral_indices(blue, green, red, nir, swir1, L: float = 0.5) -> dict[str, np.ndarray]:
    """Indices from reflectance (0-1). Sentinel-2: blue=B2, green=B3, red=B4, nir=B8, swir1=B11."""
    return {
        "NDVI": (nir - red) / (nir + red + EPS),
        "NDBI": (swir1 - nir) / (swir1 + nir + EPS),
        "MNDWI": (green - swir1) / (green + swir1 + EPS),
        "BSI": ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + EPS),
        "SAVI": (1.0 + L) * (nir - red) / (nir + red + L),
    }


def slope_aspect(dem: np.ndarray, res: float) -> tuple[np.ndarray, np.ndarray]:
    """Slope (degrees) and aspect (degrees clockwise from north) for a north-up DEM."""
    dz_drow, dz_dx = np.gradient(dem.astype("float64"), res)
    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_drow)))
    aspect = np.degrees(np.arctan2(-dz_dx, dz_drow)) % 360.0  # rows increase southwards
    return slope.astype("float32"), aspect.astype("float32")


def tune_lightgbm(X, y, groups, n_splits: int = 5, seed: int = 42, n_jobs: int = -1) -> GridSearchCV:
    """Grid search with spatially grouped folds. Parallelism is at the grid level, so each
    LightGBM fit uses one thread (avoids CPU oversubscription)."""
    grid = {
        "num_leaves": [15, 31, 63],
        "learning_rate": [0.05, 0.1],
        "n_estimators": [200, 400],
        "min_child_samples": [20, 50],
    }
    model = lgb.LGBMClassifier(objective="multiclass", class_weight="balanced",
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                               random_state=seed, n_jobs=1, verbose=-1)
    search = GridSearchCV(model, grid, scoring="f1_macro", cv=GroupKFold(n_splits=n_splits),
                          n_jobs=n_jobs, refit=True)
    search.fit(X, y, groups=groups)
    return search


def assess(y_true, y_pred) -> dict:
    return {
        "overall_accuracy": accuracy_score(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "per_class_f1": f1_score(y_true, y_pred, average=None),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def shap_importance(model, X_sample: np.ndarray, feature_names=FEATURES):
    """Mean |SHAP| per feature (averaged over samples and classes) plus the raw values.

    Recent shap versions return an (n, features, classes) array for multiclass tree models;
    older versions return a list with one (n, features) array per class. Both are handled.
    """
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_sample)
    if isinstance(sv, list):
        sv = np.stack(sv, axis=-1)
    mean_abs = np.abs(sv).mean(axis=(0, 2)) if sv.ndim == 3 else np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    return [(feature_names[i], float(mean_abs[i])) for i in order], sv


def area_shares(class_map: np.ndarray, pixel_area_m2: float, names: dict[int, str]) -> list[tuple]:
    """(class, hectares, percent) for each mapped class; 0 is treated as no-data."""
    vals, counts = np.unique(class_map[class_map > 0], return_counts=True)
    total = counts.sum()
    return [(names.get(int(v), str(v)), c * pixel_area_m2 / 1e4, 100.0 * c / total)
            for v, c in zip(vals, counts)]
```

Example usage with real rasters (paths are placeholders):

```python
import numpy as np
import rasterio
from rasterio.enums import Resampling
from lulc_pipeline import (read_to_grid, s2_reflectance, stack_features, sample_pixels,
                           consolidate, tune_lightgbm, assess, shap_importance,
                           predict_map, area_shares, write_geotiff)

with rasterio.open("data/S2_B04_10m.tif") as ref:          # 10 m reference grid
    profile = ref.profile
bands = {b: s2_reflectance(read_to_grid(f"data/S2_{b}.tif", profile))
         for b in ["B2", "B3", "B4", "B8", "B11", "B12"]}
dem = read_to_grid("data/dem.tif", profile)
pop = read_to_grid("data/population.tif", profile)
labels = read_to_grid("data/reference_labels.tif", profile, Resampling.nearest)
labels = np.nan_to_num(labels).astype("int32")                # 0 = unlabelled

cube = stack_features(bands, dem, pop, res=10.0)
X, y, groups = sample_pixels(cube, labels, block=50)          # 50 px = 500 m blocks
y = consolidate(y, {7: 1, 8: 1})                              # e.g. grass, shrub -> vegetation
search = tune_lightgbm(X, y, groups)                          # use a block hold-out for the final test
model = search.best_estimator_
ranking, shap_values = shap_importance(model, X[:2000])
class_map = predict_map(model, cube)
write_geotiff("outputs/landcover.tif", class_map, profile)
print(area_shares(class_map, 100.0, {1: "vegetation", 2: "built-up", 3: "cropland", 4: "trees", 5: "bare land"}))
```
