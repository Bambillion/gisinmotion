"""Insert a syntax-checked excerpt of code/lulc_pipeline.py into the framework article."""
import ast, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(ROOT, "code", "lulc_pipeline.py")).read()
tree = ast.parse(src)
lines = src.splitlines()
WANT = ["spectral_indices", "slope_aspect", "tune_lightgbm", "assess", "shap_importance", "area_shares"]
header = '''import numpy as np
import lightgbm as lgb
import shap
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold

EPS = 1e-6
FEATURES = ["B2", "B3", "B4", "B8", "B11", "B12",
            "NDVI", "NDBI", "MNDWI", "BSI", "SAVI",
            "elevation", "slope", "aspect_sin", "aspect_cos", "pop_density"]
'''
parts = [header]
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in WANT:
        parts.append("\n".join(lines[node.lineno - 1:node.end_lineno]))
excerpt = "\n\n\n".join(parts)
usage = '''import numpy as np
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
print(area_shares(class_map, 100.0, {1: "vegetation", 2: "built-up", 3: "cropland", 4: "trees", 5: "bare land"}))'''
compile(excerpt, "excerpt", "exec"); compile(usage, "usage", "exec")
art = open(os.path.join(ROOT, "tools", "article_framework.src.md")).read()
art = art.replace("%%CODE_EXCERPT%%", excerpt).replace("%%CODE_USAGE%%", usage)
open(os.path.join(ROOT, "articles", "explainable-multisource-lulc-framework.md"), "w").write(art)
print("article written; excerpt lines:", excerpt.count("\n") + 1)
