# GOES-MODIS Albedo Postprocessing

This directory contains the notebook used after the GOES-to-MODIS U-Net has produced prediction arrays or GeoTIFFs. It georeferences predictions, prepares comparison data, calculates masked metrics, samples field observations, and creates diagnostic and publication-oriented figures.

## Contents

| File | Description |
|---|---|
| `Final-Visualizations.ipynb` | Main postprocessing, evaluation, field-comparison, and visualization notebook. |

The notebook imports reusable code from:

```text
../goes_modis_downscaling/plot_fxns.py
../goes_modis_downscaling/albedo_unet_fxns.py
../goes_modis_downscaling/data_preprocessing.py
../goes_modis_downscaling/modis_bluesky_albedo.py
```

## Prerequisites

Before using the notebook, complete the following:

- preprocess and align GOES and MODIS rasters;
- calculate daily MODIS blue-sky albedo;
- train or load the GOES-to-MODIS model;
- save the prediction array or prediction rasters; and
- preserve the exact ordered test-date list used by the model.

Activate the environment and open JupyterLab:

```bash
conda activate sail_env
jupyter lab
```

## Configuration

Update the helper import path at the start of the notebook:

```python
from pathlib import Path
import sys

repo_root = Path("/path/to/downscaling_geostationary_land_surface_albedo")
sys.path.append(str(repo_root / "goes_modis_downscaling"))

from plot_fxns import *
from albedo_unet_fxns import *
from data_preprocessing import *
```

Also review the hard-coded data, model, field-observation, AOI, and output paths in the imported modules. Some modules open configured files during import.

## Main tasks

`Final-Visualizations.ipynb` includes workflows for:

1. reading invalid and excluded dates;
2. reconstructing the ordered model test dates;
3. converting NumPy prediction stacks to daily GeoTIFFs;
4. creating interpolated MODIS rasters for display;
5. calculating masked R-squared and RMSE;
6. plotting missing-pixel percentages through time;
7. comparing GOES predictions with MODIS targets;
8. inspecting raw GOES, MODIS BSA, WSA, and blue-sky products;
9. sampling rasters at the SAIL field location;
10. comparing Sentinel-2, MODIS, GOES, and field values for selected dates; and
11. producing multi-panel maps and time-series figures.

## Required inputs

Typical inputs include:

- first-stage model prediction arrays or GeoTIFFs;
- corresponding test-date JSON or ordered date list;
- GOES 500 m-aligned rasters;
- MODIS blue-sky albedo rasters;
- invalid-date metadata;
- East River geometry;
- selected raw GOES and MODIS examples;
- Sentinel-2 products for cross-scale comparison; and
- optional SAIL field-albedo observations.

## Prediction georeferencing

Prediction arrays must be paired with reference MODIS rasters in the same date order. The output should inherit the target raster's:

- CRS;
- affine transform;
- width and height;
- nodata convention; and
- acquisition date.

Use the exact test dates saved by `goes_modis_downscaling/goes_modis_unet.py` or the notebook driver whenever available.

## Metrics

Metric helpers calculate R-squared and RMSE only over finite target pixels. Interpolated MODIS values may be useful for visualization, but they should not be treated as observed target pixels when calculating model skill.

Review active thresholds for:

- minimum valid-pixel fraction;
- minimum valid-pixel count; and
- nearly constant targets that make R-squared unstable.

## Outputs

The notebook may create or display:

- daily predicted GOES-to-MODIS GeoTIFFs;
- interpolated MODIS comparison rasters;
- masked per-date and aggregate metrics;
- missing-data diagnostics;
- prediction-versus-target maps;
- field-location comparisons;
- product-comparison figures; and
- final multi-source maps and time series.

Generated rasters and figures should be stored outside the source-code directories.

## File-matching conventions

Current workflows use several date formats:

- raw GOES files contain `_sYYYYDDDHHMM...`;
- MODIS products commonly use a leading `YYYYDDD` token;
- first-stage prediction rasters may contain calendar dates; and
- Sentinel-2 products use `YYYY-MM-DD`.

Update the relevant parsing functions in `../goes_modis_downscaling/` whenever naming conventions change.

## Validation checklist

Confirm that:

```python
assert prediction.shape == target.shape
assert prediction_crs == target_crs
assert prediction_transform == target_transform
```

Also verify:

- the prediction count equals the retained test-date count;
- skipped dates do not consume an array index;
- all products use consistent albedo scaling;
- masks are derived from original target validity;
- field coordinates are expressed in the raster CRS; and
- axis labels describe the actual coordinate system.

## Common problems

### Prediction dates are shifted

The date exclusions or test interval differ from model training. Use the saved test-date metadata from the same run.

### No overlapping files are found

Inspect filename patterns and parsed dates for each product.

### Shape mismatch when writing rasters

The prediction was created for a different AOI or target grid. Recreate the model inputs or use the correct spatial reference.

### Imports fail immediately

Update all module-level file paths before importing the helpers.

### Metrics look unrealistically good

Ensure that filled or interpolated target pixels are excluded from the metric mask.

## Related directories

- Inputs are prepared in `../GOES-Modis-Data-Preprocessing-main/`.
- The first-stage model is in `../goes_modis_downscaling/`.
- Sentinel-2 target production is in `../sentinel_2/`.
- Finer-resolution models are in `../modis_s2_downscaling/` and `../goes_s2_downscaling/`.
