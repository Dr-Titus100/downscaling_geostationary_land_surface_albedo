# Shared Project Functions

This directory contains the reusable Python code for the GOES-to-MODIS workflow. The modules cover raster preprocessing, MODIS blue-sky albedo calculation, U-Net data preparation and training, prediction export, evaluation metrics, and visualization.

## Contents

| File | Main responsibility |
|---|---|
| `data_preprocessing.py` | GOES/MODIS clipping, reprojection, resampling, quality filtering, date traversal, missing-data matching, and raster utilities. |
| `modis_bluesky_albedo.py` | Reads MODIS black-sky/white-sky albedo, obtains aerosol information, calculates diffuse fraction and blue-sky albedo, and writes GeoTIFFs. |
| `albedo_unet1_fxns.py` | First-stage U-Net data loading, padding, model architecture, training, metrics, prediction export, and date conversion. |
| `unet1_main.py` | Script entry point that assembles train/validation/test datasets and runs the first-stage U-Net. |
| `plot_fxns.py` | Prediction rasterization, date matching, masked R-squared/RMSE, missing-data diagnostics, and comparison plots. |

## Important design note

These modules were developed as research workflow helpers rather than as an installed Python package. They use module-level configuration and, in some cases, open files while being imported. You must update the absolute paths near the top of each module before importing it.

A safer long-term refactor would move paths and thresholds into a configuration file and defer all file access until a function is called. That refactor is not part of the current repository state.

## Import setup

From a notebook or script, add this directory to `sys.path`:

```python
from pathlib import Path
import sys

repo_root = Path("/path/to/downscaling_geostationary_land_surface_albedo")
sys.path.append(str(repo_root / "functions"))
```

Then import only what is needed:

```python
from data_preprocessing import reproject_raster, mask_goes_to_match_modis
from modis_bluesky_albedo import calculate_blue_sky_albedo
from albedo_unet1_fxns import get_UNet_model, get_data_and_mask
from plot_fxns import calculate_r2_scores, calculate_RMSE_scores
```

Explicit imports are easier to trace than wildcard imports and reduce accidental name collisions.

## `data_preprocessing.py`

### Configuration

The module defines paths for:

- raw MODIS albedo;
- clipped MODIS output;
- calculated blue-sky albedo;
- MODIS AOD;
- raw GOES ABI LSAC;
- clipped GOES output;
- GOES data masked with MODIS missing pixels;
- East River and Colorado boundaries; and
- example GOES and MODIS files.

It also defines:

```text
UPSCALE_FACTOR = 2
GOES_OFFSET = 10000
NAN_PIXEL_THRESHOLD = 0.6
NODATA = -9999.0
```

The module opens the AOI shapefile, Colorado boundary, and an example GOES file at import time to initialize projection information.

### Principal functions

| Function | Purpose |
|---|---|
| `resample_raster_bilinear` | Increases raster dimensions by an integer factor with bilinear resampling. |
| `DQF_analysis` | Tests whether enough GOES DQF pixels are in accepted classes. |
| `DQF_analysis2` | Applies bit/fill-value logic to reject scenes with excessive invalid retrievals. |
| `reproject_clip_and_upsample_goes_raster` | Reprojects GOES LSA/DQF to a clipped MODIS reference grid and writes accepted LSA rasters. |
| `is_valid_year_dir` | Validates a year directory against the processing interval. |
| `is_valid_day_dir` | Validates a Julian-day directory against the processing interval. |
| `is_valid_hour_dir` | Restricts processing to selected UTC-hour folders. |
| `clip_reproject_goes_data_loop_through_directories` | Traverses a `YYYY/DDD/HH` GOES archive and processes one selected file per hour. |
| `reproject_raster` | Reprojects all bands of a raster with Rasterio. |
| `print_rxr_metadata` | Prints Xarray/Rioxarray dimensions, variables, coordinates, and attributes. |
| `plot_band_data` | Plots the first band of a raster with spatial metadata. |
| `extract_date_from_modis_filename` | Parses a leading MODIS `YYYYDDD`. |
| `extract_datetime_from_goes_filename` | Parses the GOES `_sYYYYDDDHHMMSS_` token. |
| `visualize_all_tif_data` | Matches and plots GOES/MODIS files by day and reports gaps. |
| `mask_goes_to_match_modis` | Applies each MODIS finite-data footprint to corresponding GOES rasters. |
| `_open_raster_strict` | Normalizes GeoTIFF, NetCDF, HDF, subdataset, Dataset, or DataArray inputs. |
| `_write_geotiff_with_nodata` | Writes floating-point GeoTIFFs with an explicit no-data value. |
| `clip_modis_data` | Selects bands, optionally handles AOD orbit dimensions, clips, and writes MODIS rasters. |

### Resampling rules

- GOES LSA: bilinear when matching the MODIS grid.
- GOES DQF: nearest-neighbor to preserve categorical values.
- Generic `reproject_raster`: nearest-neighbor in the current implementation.
- AOD upsampling: bilinear in `clip_modis_data` when `aod_format=True`.

Change a resampling method only after considering whether the variable is continuous or categorical.

## `modis_bluesky_albedo.py`

### Configuration

The module expects paths to:

- reprojected MODIS albedo rasters;
- reprojected AOD rasters;
- a daily CERES AOD NetCDF;
- `sw_lut.csv`; and
- a reference GeoTIFF.

Current constants include:

```text
AOD_OFFSET = 0.001
ALBEDO_OFFSET = 0.001
SAIL_LOCATION = (39, -106)
```

### Principal functions

| Function | Purpose |
|---|---|
| `get_albedo_values` | Reads black-sky band 1 or white-sky band 2 into a date-keyed dictionary. |
| `get_aod_values` | Reads band-1 AOD rasters into a date-keyed dictionary. |
| `get_aod_static_data` | Selects nearest daily CERES AOD for a configured latitude/longitude and date. |
| `get_solar_zenith` | Approximates solar zenith angle at local solar noon. |
| `convert_last_digit_of_float` | Converts AOD to the even-hundredth string convention used by the lookup table. |
| `calculate_blue_sky_albedo` | Combines black-sky and white-sky albedo using date/pixel diffuse ratios. |
| `write_to_raster` | Writes output using a reference raster's profile, CRS, and transform. |

The calculation assumes compatible black-sky and white-sky dates and arrays. Verify lookup coverage before processing a full archive.

## `albedo_unet1_fxns.py`

### Configuration

This module defines all first-stage model data and output paths, known invalid GOES dates, the convolution kernel size, and model artifacts.

### Data functions

| Function | Purpose |
|---|---|
| `load_raster_da` | Opens a raster as `float32` while retaining spatial metadata. |
| `pad_da_2d` | Reflect-pads a `21 x 19` raster to `24 x 24`. |
| `pad_mask_2d` | Edge-pads a validity mask using the same fixed geometry. |
| `remove_padding` | Crops the fixed padding from predicted arrays. |
| `fill_inputs_interpolate` | Fills missing GOES pixels by interpolation and edge filling. |
| `prepare_target_and_mask` | Converts MODIS missing values to zero for computation and creates a validity mask. |
| `extract_goes_datetime` | Parses GOES start time. |
| `extract_modis_datetime` | Parses MODIS acquisition date. |
| `get_data_and_mask` | Filters dates, selects GOES hours, pairs targets, preprocesses rasters, and returns source paths. |
| `stack_array_4d` | Stacks date-keyed images as `(N, H, W, 1)`. |
| `stack_masks_3d` | Stacks date-keyed masks as `(N, H, W)`. |

### Model functions

| Function | Purpose |
|---|---|
| `EncoderMiniBlock` | Two convolutions plus optional max pooling and skip output. |
| `DecoderMiniBlock` | Transposed convolution, skip concatenation, and two convolutions. |
| `get_UNet_model` | Builds and compiles the U-Net regression model. |
| `run_unet` | Trains or reloads a model, evaluates it, predicts, calculates masked R-squared, and saves artifacts. |
| `masked_r2_numpy` | Calculates R-squared over mask-valid pixels. |
| `save_preds_as_geotiff` | Copies MODIS reference georeferencing to each prediction. |
| `convert_dates` | Converts invalid-date strings from JSON into Python datetimes. |

`run_unet` uses target masks as pixel-level sample weights. The saved Keras metric is RMSE, while the optimized loss is Huber loss.

## `unet1_main.py`

This file is the non-notebook entry point for the first U-Net.

Run it from the repository root:

```bash
python functions/unet1_main.py
```

The file contains Python despite its current `#!/bin/bash` first line.

The script:

1. adds the helper directory to `sys.path`;
2. fixes random seeds;
3. loads invalid dates;
4. creates train, validation, and test pairs;
5. stacks images and masks;
6. verifies finite predictors;
7. writes `test_dates.json`;
8. trains or loads the U-Net; and
9. saves predictions and timing information.

Review its date ranges, `goes_masked` setting, output folder, and model loading flags before execution.

## `plot_fxns.py`

### Configuration

This module defines paths to raw GOES, raw MODIS, MODIS blue-sky albedo, model arrays, rasterized predictions, SAIL field data, invalid-date JSON, and the East River shapefile.

It currently reads the East River shapefile at import time.

### Principal functions

| Function | Purpose |
|---|---|
| `write_goes_to_raster` | Converts ordered `.npy` predictions into daily GeoTIFFs using MODIS references. |
| `parse_goes_date` | Parses `MM-DD-YYYY` from prediction rasters. |
| `parse_modis_date` | Parses `YYYYDDD` from MODIS filenames. |
| `index_by_date` | Builds deterministic date-to-path mappings. |
| `ensure_2d_dataarray` | Selects band 1 when a band dimension is present. |
| `np2d` | Enforces a two-dimensional NumPy array. |
| `calculate_r2_scores` | Calculates filtered per-date and aggregate masked R-squared. |
| `calculate_RMSE_scores` | Calculates filtered per-date and aggregate masked RMSE and plots diagnostics. |
| `plot_pixels_per_date` | Plots valid/missing MODIS support over time. |
| GOES/MODIS date and comparison helpers | Support visual comparisons of model, mask, and target products. |

Current metric support thresholds are:

```text
MIN_VALID_PIX_FRAC = 0.60
MIN_VALID_PIX_ABS = 238
MIN_VAR = 1e-8
```

Recalculate the absolute threshold for a different AOI or raster size.

## Recommended call sequence

```text
1. data_preprocessing.py
   - clip and align GOES/MODIS
   - write masked GOES if required

2. modis_bluesky_albedo.py
   - calculate and write MODIS blue-sky targets

3. albedo_unet1_fxns.py through run_unet.ipynb or unet1_main.py
   - assemble data and train/evaluate model

4. plot_fxns.py through Final-Visualizations.ipynb
   - rasterize legacy arrays, calculate metrics, and plot results
```

## Validation and testing

There is no automated test suite in this directory. Before a full run, perform a small date-range smoke test and verify:

- path constants resolve;
- imports succeed;
- file date parsers return expected dates;
- AOI and raster bounds overlap;
- output CRS, resolution, transform, width, and height are consistent;
- scale factors are applied exactly once;
- categorical flags use nearest-neighbor resampling;
- predictor arrays contain no `NaN` or infinity;
- masks align with targets; and
- saved prediction order matches saved date order.

## Common issues

### Absolute-path failure during import

Edit the module-level paths before import. A later variable assignment in a notebook will not help if the import has already failed.

### Duplicate function names

`remove_padding` appears more than once in `albedo_unet1_fxns.py`. Python uses the last definition encountered. Keep this in mind when modifying or debugging the file.

### Date mismatch

The project uses both calendar dates and Julian dates. Normalize to a Python `datetime` or `date` before comparing files.

### Unexpected metric filtering

A scene may be skipped because of valid-pixel count, valid fraction, or low target variance even when the files exist.

### Raster output has incorrect location

Copy both CRS and affine transform from the intended reference, and verify that the array's row/column shape matches the reference raster.

## Attribution

The U-Net source includes an embedded MIT license and attribution for the encoder, decoder, and model-building code adapted from Vidushi Bhatia. Retain that notice in derivative copies.