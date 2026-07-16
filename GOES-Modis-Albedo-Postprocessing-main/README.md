# GOES-MODIS Albedo Postprocessing

This directory contains the notebook used after the first U-Net has produced GOES-to-MODIS albedo predictions. It converts prediction arrays to georeferenced rasters, prepares comparison data, calculates evaluation metrics, and creates diagnostic and publication-oriented figures.

## Contents

| File | Description |
|---|---|
| `Final-Visualizations.ipynb` | Main postprocessing and visualization notebook. |

The notebook imports shared code from `../functions/plot_fxns.py`, `../functions/albedo_unet1_fxns.py`, `../functions/data_preprocessing.py`, and `../functions/modis_bluesky_albedo.py`.

## Main tasks

`Final-Visualizations.ipynb` includes cells for:

1. reading the invalid-date JSON used during model preparation;
2. adding known GOES dates that must be skipped;
3. converting a saved U-Net `.npy` prediction stack into daily 500 m GeoTIFFs;
4. filling missing MODIS pixels to produce an interpolated comparison stack;
5. calculating masked R-squared and RMSE statistics;
6. plotting missing-pixel percentage through time;
7. displaying predicted GOES and MODIS rasters for matching dates;
8. comparing MODIS-validity masks with GOES inputs;
9. examining selected predictions at the SAIL field location;
10. displaying raw GOES, MODIS black-sky, white-sky, blue-sky, and modeled products; and
11. preparing multi-source maps and time-series figures.

## Prerequisites

Complete the following before using this notebook:

- preprocess GOES and MODIS data;
- calculate daily MODIS blue-sky albedo;
- train or load the GOES-to-MODIS U-Net;
- save the model predictions as a NumPy array; and
- save or reconstruct the exact ordered list of test dates.

Activate the project environment and start JupyterLab from the repository root:

```bash
conda activate sail_env
jupyter lab
```

Open `Final-Visualizations.ipynb` and select the `sail_env` kernel.

## Required inputs

The notebook expects paths to the following data:

- U-Net prediction array with shape `(number_of_dates, height, width)`;
- raw or MODIS-masked 500 m GOES rasters;
- daily MODIS blue-sky albedo GeoTIFFs;
- optional interpolated MODIS rasters;
- invalid MODIS date JSON;
- East River AOI shapefile;
- selected raw GOES and MODIS example files; and
- optional SAIL field-albedo CSV.

The default paths are defined in the shared helper modules and point to Boise State filesystem locations. Update them before importing the modules.

## Configuration

At the start of the notebook, replace the helper path and Matplotlib style path:

```python
from pathlib import Path
import sys

repo_root = Path("/path/to/downscaling_geostationary_land_surface_albedo")
sys.path.append(str(repo_root / "functions"))

import matplotlib.pyplot as plt
plt.style.use(repo_root / "MNRAS.mplstyle")
```

Also review the path constants in `functions/plot_fxns.py` and `functions/albedo_unet1_fxns.py`, especially:

- `GOES_500m_masked_data_path`;
- `GOES_500m_masked_output_dir`;
- `MODIS_bsa_dir`;
- `MODIS_interpolated_data_dir`;
- `GOES_NaN_Data_dir`;
- `INVALID_DATES_PATH`;
- `SAIL_field_data_file`; and
- the selected individual GOES and MODIS example files.

## Recommended execution order

### 1. Load helper modules

Restart the kernel and run the import cells only after every required path exists. Some imported modules access shapefiles and example rasters during import.

### 2. Reconstruct the prediction dates

Use the same test start date, test end date, invalid-date JSON, and hard-coded exclusions used during training. Array index `i` must represent the same date when the prediction is written to a raster.

The model pipeline writes `test_dates.json`; use that file when available rather than rebuilding the date list manually.

### 3. Convert predictions to GeoTIFF

`write_goes_to_raster(...)` uses the corresponding MODIS file as the spatial reference and writes daily files such as:

```text
05-06-2023-GOES-500m_new.tif
```

The prediction shape must match the selected MODIS raster's height and width.

### 4. Prepare comparison data

The notebook can write an interpolated MODIS version for visual comparison. Metric functions, however, use finite pixels from the non-interpolated MODIS raster as the validity mask so that filled values do not become artificial observations.

### 5. Calculate metrics

The helper functions match files by date and calculate:

- per-date masked R-squared;
- aggregate masked R-squared;
- per-date masked RMSE; and
- aggregate masked RMSE.

By default, dates are skipped when fewer than 60 percent of target pixels are valid, fewer than 238 pixels are valid, or the valid MODIS target is nearly constant. Review these thresholds in `functions/plot_fxns.py` for another AOI.

### 6. Create visual diagnostics

Run the comparison sections only after confirming that GOES and MODIS files overlap by date and have identical grids. Large date ranges can produce many figures, so filter the file lists before plotting when working interactively.

### 7. Compare with SAIL field data

The field-data section selects a timestamp and samples the nearest raster pixel at configured UTM coordinates. Verify the CSV column name, timezone, raster CRS, and SAIL coordinate before interpreting the comparison.

## Outputs

The notebook may create or display:

- daily predicted GOES 500 m GeoTIFFs;
- daily interpolated MODIS GeoTIFFs;
- masked R-squared and RMSE summaries;
- missing-data diagnostics;
- daily prediction-versus-target figures;
- raw-product comparison figures;
- SAIL point comparisons; and
- combined maps and time-series graphics.

Output rasters and images are ignored by the repository's `.gitignore`. Store finalized results in a backed-up project-results directory.

## File matching conventions

The postprocessing functions assume:

- predicted GOES rasters contain `MM-DD-YYYY` in the filename;
- MODIS rasters contain a seven-digit `YYYYDDD` acquisition date; and
- raw GOES files contain an `_sYYYYDDDHHMM...` start-time token.

Update the parsers in `functions/plot_fxns.py` and `functions/data_preprocessing.py` if these conventions change.

## Validation checks

Before accepting the results, verify:

```python
assert predicted.shape == target.shape
assert predicted_crs == target_crs
assert predicted_transform == target_transform
```

Also confirm that:

- the number of prediction arrays equals the number of retained test dates;
- skipped dates do not consume a prediction index;
- albedo values use consistent scaling and physical units;
- metrics are calculated only over valid target pixels; and
- map labels describe the actual CRS rather than generic longitude and latitude.

## Common problems

### Prediction dates appear shifted

The invalid-date list or test interval differs from the training run. Use `test_dates.json` from the same model output.

### No overlapping dates are found

Check filename conventions and verify that both file lists cover the same period.

### Shape mismatch when writing rasters

A prediction was produced for a different AOI or preprocessing grid. Recreate the test data on the intended MODIS grid or update the full training and cropping workflow.

### Helper import fails

Update every absolute path read at module import time, including the AOI shapefile and example GOES/MODIS files.

### Metrics look unrealistically good

Make sure interpolated target values are not being treated as observations. Build the metric mask from the original MODIS-valid pixels.

## Relationship to other directories

- Input rasters are prepared in `../GOES-Modis-Data-Preprocessing-main/`.
- The prediction array is produced in `../GOES-Modis-U-Net-Albedo-Code-main/` or by `../functions/unet1_main.py`.
- Metric and plotting implementations are in `../functions/plot_fxns.py`.
- Sentinel-2 refinement is handled separately in `../sentinel_2/` and `../s2_modis_downscaling/`.