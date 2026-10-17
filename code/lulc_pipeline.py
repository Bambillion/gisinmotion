"""
Explainable multi-source LULC classification: a minimal tutorial pipeline.

Companion code for the GIS in Motion article
"An Explainable Multi-Source Machine Learning Framework for Urban Land Cover Classification".

What it shows:
  1. Reading Sentinel-2 bands, a DEM and a population raster onto one 10 m grid (rasterio)
  2. Computing NDVI, NDBI, MNDWI, BSI, SAVI, slope and aspect (numpy)
  3. Stacking features and sampling labelled pixels
  4. Tuning LightGBM with GridSearchCV and spatially blocked folds
  5. Accuracy assessment (OA, kappa, F1, confusion matrix)
  6. SHAP explanations with TreeExplainer
  7. Predicting a map and reporting class area shares

Run `python lulc_pipeline.py` to execute an end-to-end demo on SYNTHETIC data.
The demo's numbers mean nothing about any real place; they only prove the code runs.

Requirements: numpy, rasterio, scikit-learn, lightgbm, shap (and matplotlib for plots).
"""
from __future__ import annotations

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score)
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
import lightgbm as lgb
import shap

EPS = 1e-6
FEATURES = ["B2", "B3", "B4", "B8", "B11", "B12",
            "NDVI", "NDBI", "MNDWI", "BSI", "SAVI",
            "elevation", "slope", "aspect_sin", "aspect_cos", "pop_density"]


# ---------------------------------------------------------------------------
# 1. I/O helpers
# ---------------------------------------------------------------------------
def read_to_grid(path: str, ref_profile: dict, resampling=Resampling.bilinear) -> np.ndarray:
    """Read band 1 of `path` and reproject/resample it onto the reference grid."""
    dst = np.full((ref_profile["height"], ref_profile["width"]), np.nan, dtype="float32")
    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
            dst_transform=ref_profile["transform"], dst_crs=ref_profile["crs"], dst_nodata=np.nan,
            resampling=resampling,
        )
    return dst


def s2_reflectance(dn: np.ndarray, offset: float = -1000.0, scale: float = 10000.0) -> np.ndarray:
    """Convert Sentinel-2 L2A digital numbers to surface reflectance.

    For products from processing baseline 04.00 onwards, L2A carries BOA_ADD_OFFSET = -1000;
    check the product metadata and use offset=0 for older products.
    """
    return (dn.astype("float32") + offset) / scale


# ---------------------------------------------------------------------------
# 2. Spectral indices and terrain
# ---------------------------------------------------------------------------
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


def stack_features(bands: dict[str, np.ndarray], dem: np.ndarray, pop: np.ndarray, res: float) -> np.ndarray:
    """Return an (H, W, F) feature cube in the order given by FEATURES."""
    idx = spectral_indices(bands["B2"], bands["B3"], bands["B4"], bands["B8"], bands["B11"])
    slope, aspect = slope_aspect(dem, res)
    layers = {**bands, **idx, "elevation": dem, "slope": slope,
              "aspect_sin": np.sin(np.radians(aspect)),   # eastness
              "aspect_cos": np.cos(np.radians(aspect)),   # northness
              "pop_density": pop}
    return np.dstack([layers[f].astype("float32") for f in FEATURES])


# ---------------------------------------------------------------------------
# 3. Training samples, class consolidation, spatial blocks
# ---------------------------------------------------------------------------
def sample_pixels(cube: np.ndarray, labels: np.ndarray, block: int = 50):
    """Labelled pixels (label > 0) -> X, y, and a spatial block id for grouped CV."""
    rows, cols = np.nonzero(labels > 0)
    X = cube[rows, cols, :]
    y = labels[rows, cols]
    n_block_cols = int(np.ceil(labels.shape[1] / block))
    groups = (rows // block) * n_block_cols + (cols // block)
    ok = np.all(np.isfinite(X), axis=1)
    return X[ok], y[ok], groups[ok]


def consolidate(y: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    """Merge spectrally similar classes, e.g. {grassland: vegetation, shrubland: vegetation}."""
    out = y.copy()
    for old, new in mapping.items():
        out[y == old] = new
    return out


# ---------------------------------------------------------------------------
# 4. Model tuning
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 5. Accuracy assessment
# ---------------------------------------------------------------------------
def assess(y_true, y_pred) -> dict:
    return {
        "overall_accuracy": accuracy_score(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "per_class_f1": f1_score(y_true, y_pred, average=None),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# 6. SHAP explanations
# ---------------------------------------------------------------------------
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


def shap_summary_plot(sv: np.ndarray, X_sample: np.ndarray, class_index: int, path: str):
    """Beeswarm summary plot for one class (requires matplotlib)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    shap.summary_plot(sv[:, :, class_index], X_sample, feature_names=FEATURES, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# 7. Map prediction and area statistics
# ---------------------------------------------------------------------------
def predict_map(model, cube: np.ndarray, chunk: int = 500_000) -> np.ndarray:
    h, w, f = cube.shape
    flat = cube.reshape(-1, f)
    out = np.zeros(h * w, dtype="uint8")
    valid = np.all(np.isfinite(flat), axis=1)
    idx = np.nonzero(valid)[0]
    for start in range(0, idx.size, chunk):
        sel = idx[start:start + chunk]
        out[sel] = model.predict(flat[sel])
    return out.reshape(h, w)


def area_shares(class_map: np.ndarray, pixel_area_m2: float, names: dict[int, str]) -> list[tuple]:
    """(class, hectares, percent) for each mapped class; 0 is treated as no-data."""
    vals, counts = np.unique(class_map[class_map > 0], return_counts=True)
    total = counts.sum()
    return [(names.get(int(v), str(v)), c * pixel_area_m2 / 1e4, 100.0 * c / total)
            for v, c in zip(vals, counts)]


def write_geotiff(path: str, array: np.ndarray, profile: dict):
    prof = {**profile, "count": 1, "dtype": "uint8", "nodata": 0, "compress": "deflate"}
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(array.astype("uint8"), 1)


# ---------------------------------------------------------------------------
# Demo on synthetic data (proves the pipeline runs; results are meaningless)
# ---------------------------------------------------------------------------
def synthetic_scene(h: int = 200, w: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(1, 6, size=(h // 20, w // 20)).repeat(20, 0).repeat(20, 1)
    base = {  # rough, made-up reflectance levels per class 1..5
        "B2": [0.04, 0.10, 0.06, 0.03, 0.12], "B3": [0.07, 0.12, 0.09, 0.05, 0.14],
        "B4": [0.05, 0.14, 0.08, 0.03, 0.18], "B8": [0.35, 0.20, 0.30, 0.40, 0.24],
        "B11": [0.18, 0.26, 0.22, 0.15, 0.32], "B12": [0.09, 0.22, 0.15, 0.07, 0.28],
    }
    bands = {b: (np.take(v, labels - 1) + rng.normal(0, 0.02, (h, w))).astype("float32")
             for b, v in base.items()}
    yy, xx = np.mgrid[0:h, 0:w]
    dem = (450 + 0.3 * yy + 10 * np.sin(xx / 15)).astype("float32")
    pop = (np.where(labels == 2, 80.0, 10.0) + rng.gamma(2.0, 5.0, (h, w))).astype("float32")
    return bands, dem, pop, labels


def run_demo():
    names = {1: "vegetation", 2: "built-up", 3: "cropland", 4: "trees", 5: "bare land"}
    bands, dem, pop, labels = synthetic_scene()
    cube = stack_features(bands, dem, pop, res=10.0)
    # keep a sparse set of "reference" pixels, as with real field/visual samples
    rng = np.random.default_rng(1)
    sparse = np.where(rng.random(labels.shape) < 0.05, labels, 0)
    X, y, groups = sample_pixels(cube, sparse, block=40)
    # hold out whole spatial blocks (not random pixels) to limit spatial leakage
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42).split(X, y, groups))
    Xtr, Xte, ytr, yte, gtr = X[tr], X[te], y[tr], y[te], groups[tr]
    search = tune_lightgbm(Xtr, ytr, gtr, n_splits=3)
    print("best params:", search.best_params_)
    metrics = assess(yte, search.predict(Xte))
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics.items() if k != "confusion_matrix"})
    print("confusion matrix:\n", metrics["confusion_matrix"])
    ranking, _ = shap_importance(search.best_estimator_, Xte[:300])
    print("top SHAP features:", ranking[:5])
    class_map = predict_map(search.best_estimator_, cube)
    for name, ha, pct in area_shares(class_map, 100.0, names):
        print(f"{name:>10s}: {ha:8.1f} ha  {pct:5.1f} %")


if __name__ == "__main__":
    run_demo()
