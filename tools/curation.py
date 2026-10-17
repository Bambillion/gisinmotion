"""Curation layer: topic grouping + plain-language summaries written from each paper's real arXiv abstract.
Metadata (title, authors, dates, DOI) is taken verbatim from the arXiv API (see tools/arxiv_fetch.py)."""
TOPICS = [
 ("lulc-ml", "LULC classification with Sentinel-2 / Landsat and machine learning",
  "Core papers on supervised land use / land cover mapping from optical satellite imagery."),
 ("ensembles", "Ensemble learners and gradient boosting",
  "The algorithms behind XGBoost and CatBoost, and a recent benchmark of tree ensembles for urban land cover."),
 ("xai", "Explainable AI and SHAP",
  "Shapley-value explanations, tree-specific SHAP, and explainability reviews for remote sensing."),
 ("fusion", "Spectral indices, multi-source data and fusion",
  "Combining optical bands, radar, indices (NDVI, NDBI, ...) and auxiliary geographic layers."),
 ("validation", "Sampling, validation and spatial pitfalls",
  "Reference data quality, spatial cross-validation and over-optimistic accuracy."),
 ("africa", "Africa, Nigeria and fast-growing cities",
  "Land cover and settlement mapping in African and other rapidly urbanising contexts."),
 ("benchmarks", "Benchmark datasets",
  "Large labelled Sentinel datasets for training and comparing LULC models."),
]
PAPERS = {
 # lulc-ml
 "2104.10922": ("lulc-ml", "Builds a 10 m land cover map of Europe (ELC10) with a Random Forest trained on LUCAS survey points inside Google Earth Engine. It reports that combining Sentinel-1 radar with Sentinel-2 optical data improved accuracy, while atmospheric correction and speckle filtering added little."),
 "2608.12001": ("lulc-ml", "Uses Sentinel-2 and Landsat 8 with NDVI, NDBI and NDWI in Google Earth Engine to track land use and vegetation change in Dhaka (2019-2024), comparing Decision Tree, KNN and Random Forest. Random Forest performed best, and the study documents rapid built-up expansion at the expense of vegetation and water."),
 "2004.12880": ("lulc-ml", "Proposes a recurrent-convolutional neural network that learns temporal features from multi-temporal Sentinel-2 imagery for pixel-based land cover and crop classification in northern Italy. It compares the deep model with traditional learners such as SVM and Random Forest."),
 "1710.03959": ("lulc-ml", "A widely cited review of deep learning for remote sensing that discusses its challenges, recent advances and practical resources. Useful context for when tree ensembles are (and are not) the right tool."),
 # ensembles
 "1603.02754": ("ensembles", "The original XGBoost paper: a scalable tree-boosting system with a sparsity-aware split algorithm and weighted quantile sketch. It explains why XGBoost became a default strong baseline for tabular data."),
 "1706.09516": ("ensembles", "Describes CatBoost's two key ideas, ordered boosting and a new way of handling categorical features, both designed to stop the 'prediction shift' caused by target leakage in standard gradient boosting."),
 "2609.19010": ("ensembles", "Benchmarks classical models (including Random Forest, XGBoost and CatBoost) against tabular deep learning models on the UCI Urban Land Cover dataset, using macro-F1 and confusion matrices. Tree ensembles remain strong baselines, while tabular deep models can match them when imbalance is handled well."),
 # xai
 "1705.07874": ("xai", "Introduces SHAP (SHapley Additive exPlanations), a unified framework that assigns each feature an importance value for a single prediction, grounded in game-theoretic Shapley values. This is the method behind most SHAP plots in LULC papers."),
 "1905.04610": ("xai", "Presents TreeSHAP, a polynomial-time algorithm for exact Shapley-value explanations of tree models, plus interaction values and tools for turning many local explanations into global understanding. Directly relevant to explaining Random Forest, XGBoost, LightGBM and CatBoost."),
 "2402.13791": ("xai", "A systematic review of explainable AI in remote sensing: which methods are used, what they are used for, how explanations are interpreted and evaluated, and open challenges. A good map of the field for newcomers."),
 "2301.01520": ("xai", "Proposes a GAN-based counterfactual explanation method for land cover classification from satellite image time series, showing what minimal, time-localised changes would flip a prediction to another class."),
 "2310.02476": ("xai", "Uses Random Forest and XGBoost with feature-importance analysis to study how urban features, including land cover and infrastructure, shape exposure to air pollution, heat and flooding across six US counties. An example of interpretable ML applied to urban geography."),
 # fusion
 "2312.10798": ("fusion", "Fuses Sentinel-1 SAR texture with Sentinel-2 VNIR-SWIR bands (Bayesian fusion) and classifies LULC with ensembles of Random Forests built with random rotations. It evaluates how fusion and rotation strategy affect accuracy."),
 "2507.13385": ("fusion", "Shows that appending extra geographic layers (e.g. elevation and other non-optical inputs) to satellite imagery improves ML performance, especially with little labelled data and in out-of-region tests. Simple hard-coded fusion often beat learned fusion."),
 "2605.24459": ("fusion", "Analyses urban heat island drivers across Tehran's districts (2003-2021) using NDVI, EVI, NDBI, NDWI and NDSI time series with a Hotelling's T-square based causal test. Useful for seeing how spectral indices are interpreted as urban and vegetation proxies."),
 # validation
 "1908.07805": ("validation", "Demonstrates with Random Forest land cover and LAI case studies that random cross-validation gives over-optimistic accuracy for spatial data, and that spatial cross-validation plus spatial variable selection are needed. Location-like predictors can cause severe overfitting."),
 "2403.15720": ("validation", "Studies how different people labelling reference samples changes land cover results (44 investigators, Saitama, Japan) and proposes a Bayesian (Dirichlet) fusion of classification maps to reduce that variability and salt-and-pepper noise."),
 # africa
 "2307.02575": ("africa", "Compares 11 public land cover maps against rigorous reference data from 8 Sub-Saharan African countries to see which identify cropland best. A reminder that global products can disagree strongly in African landscapes."),
 "2411.02935": ("africa", "Produces a 10 m urban-rural settlement map for Africa (2016-2022) with a DeepLabV3 model fusing Landsat-8, VIIRS night lights, ESRI LULC and GHS-SMOD, validated against DHS survey labels."),
 "2312.10872": ("africa", "Country-scale cropland mapping in Nigeria with Sentinel-1, Sentinel-2, ERA5 climate and a DEM in Google Earth Engine, comparing Random Forest and LSTMs. Local (Nigeria-only) training data beat larger global datasets, and radar, climate and topography were important inputs."),
 "2205.02946": ("africa", "Assesses the vertical accuracy of the 20 m SPOT DEM against 780 ground control points in Lagos State and the FCT (Abuja), across land cover types. Relevant when using elevation, slope and aspect as features for Abuja-area studies."),
 # benchmarks
 "1709.00029": ("benchmarks", "Introduces EuroSAT: 27,000 labelled, geo-referenced Sentinel-2 patches across 10 LULC classes and all 13 bands, with CNN baselines. A standard starter dataset for LULC classification."),
 "1902.06148": ("benchmarks", "Introduces BigEarthNet, a large multi-label archive of 590,326 Sentinel-2 patches labelled from CORINE Land Cover 2018. Shows that models trained on it outperform ImageNet-pretrained CNNs for RS scene classification."),
 "1906.07789": ("benchmarks", "SEN12MS: 180,662 georeferenced triplets of Sentinel-1 SAR, Sentinel-2 multispectral patches and MODIS land cover covering all inhabited continents and seasons, built for deep learning and data fusion."),
}
