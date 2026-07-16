# GOES-to-MODIS U-Net and Shared Modules

This directory contains the first-stage GOES-to-MODIS U-Net workflow and the shared modules used for GOES/MODIS preprocessing, MODIS blue-sky albedo, model training, evaluation, and plotting.

## Contents

| File | Responsibility |
|---|---|
| `run_goes_modis_unet.ipynb` | Interactive driver for assembling datasets, training or loading the first-stage model, evaluating it, and reviewing outputs. |
| `goes_modis_unet.py` | Script driver for the complete GOES-to-MODIS training and prediction workflow. |
| `albedo_unet_fxns.py` | Date matching, raster preparation, target masks, padding, U-Net architecture, training, metrics, and prediction output. |
| `data_preprocessing.py` | GOES and MODIS clipping, reprojection, quality screening, date checks, masking, and grid-alignment helpers. |
| `modis_bluesky_albedo.py` | MODIS black-sky/white-sky combination and diffuse-fraction helpers. |
| `plot_fxns.py` | Prediction rasterization, masked evaluation, date matching, field sampling, and visualization helpers. |
| `packages.py` | Shared imports used by the Python modules in this directory. |

These files were previously divided between `functions/` and `GOES-Modis-U-Net-Albedo-Code-main/`. Use the paths in this directory for current imports.

## Model role

The first downscaling stage uses:

- **predictor:** GOES ABI Level-2 LSAC albedo aligned to the MODIS grid;
- **target:** daily MODIS blue-sky albedo;
- **sample weight:** a per-pixel mask of finite MODIS target values; and
- **output:** predicted MODIS-scale albedo for available GOES dates.

This is continuous image-to-image regression.

## Prerequisites

Complete the workflows in `../GOES-Modis-Data-Preprocessing-main/` first. Required inputs include:

- aligned GOES GeoTIFFs;
- MODIS blue-sky albedo GeoTIFFs;
- optional GOES files masked to the MODIS footprint;
- invalid-date metadata;
- writable model and prediction directories; and
- filenames that preserve acquisition dates.

Activate the project environment:

```bash
conda activate sail_env
```

A GPU is recommended for training.

## Importing the modules

Run notebooks from the repository root or add this directory explicitly:

```python
from pathlib import Path
import sys

repo_root = Path("/path/to/downscaling_geostationary_land_surface_albedo")
sys.path.append(str(repo_root / "goes_modis_downscaling"))

from albedo_unet_fxns import *
from data_preprocessing import *
from modis_bluesky_albedo import *
from plot_fxns import *
```

The modules use `from packages import *`, so this directory must precede unrelated directories containing another `packages.py` on `sys.path`.

## Configuration

Review the module-level paths before import. Important settings include:

```text
MODIS_BLUE_SKY_ALBEDO_DIR
GOES_ALBEDO_DIR
MASKED_GOES_ALBEDO_DIR
UNET_RESULTS
TENSORFLOW_CHECKPOINT_PATH
TENSORFLOW_TRAINING_DIR
TF_HISTORY_PATH
INVALID_DATES_PATH
TF_FELIX_MODEL_UNMASKED_PATH
```

`data_preprocessing.py` also configures raw and processed satellite directories, AOI and Colorado boundaries, example files, scaling constants, and quality thresholds. It opens some of these resources during import.

## Date matching

The model helpers:

1. parse GOES timestamps from `_sYYYYDDDHHMM...`;
2. parse MODIS dates from a leading `YYYYDDD` token;
3. filter the requested date interval;
4. remove invalid dates;
5. prefer the selected 18 UTC GOES observation;
6. use configured 19 UTC replacements for selected dates;
7. retain common GOES and MODIS dates; and
8. preserve sorted source paths for prediction georeferencing.

Review all hard-coded invalid-date lists when the source data or preprocessing changes.

## Array preparation

### GOES input

GOES arrays are converted to `float32`. Missing values are interpolated and edge-filled before padding.

### MODIS target

Finite MODIS pixels form the target mask. Missing target values are replaced by zero only for tensor computation; their sample weights remain zero.

### Current East River grid

The first-stage native grid is approximately `21 x 19` pixels. It is padded to `24 x 24` with:

```text
1 row above, 2 rows below
2 columns left, 3 columns right
```

Predictions are cropped with the inverse operation. Update padding, cropping, and georeferencing together for another grid.

## U-Net configuration

The current workflow uses:

- one input channel;
- three pooling/upsampling levels with skip connections;
- convolutional encoder and decoder blocks;
- Adam optimization at approximately `1e-4`;
- Huber loss with `delta=0.05`;
- RMSE as a Keras metric;
- early stopping;
- learning-rate reduction; and
- checkpoint and full-model saving.

Check the active source rather than assuming notebook experiments use identical settings.

## Run the workflow

Interactive route:

```text
goes_modis_downscaling/run_goes_modis_unet.ipynb
```

Script route, from the repository root:

```bash
python goes_modis_downscaling/goes_modis_unet.py
```

Before running, confirm:

- train, validation, and test date ranges;
- masked or unmasked GOES selection;
- invalid dates;
- loading versus new-training flags;
- output directories; and
- exact tensor shapes and date counts.

## Outputs

The workflow may write:

- checkpoint weights;
- a saved Keras model;
- training-history JSON;
- ordered test-date metadata;
- NumPy prediction arrays;
- predicted GeoTIFFs using MODIS references;
- printed loss and RMSE; and
- masked R-squared diagnostics.

## Shared preprocessing modules

### `data_preprocessing.py`

Supports GOES DQF analysis, scaling, clipping, reprojection, MODIS-grid matching, archive traversal, invalid-date analysis, and GOES-to-MODIS missing-pixel masking.

### `modis_bluesky_albedo.py`

Calculates blue-sky albedo as a mixture of MODIS BSA and WSA using a diffuse skylight fraction derived from aerosol and solar geometry.

### `plot_fxns.py`

Supports date-aware raster writing, masked R-squared and RMSE, interpolation for display, product comparison, field sampling, and publication plots. It is used heavily by `../GOES-Modis-Albedo-Postprocessing-main/Final-Visualizations.ipynb`.

## Validation checklist

Verify:

```python
assert goes_data.shape == modis_data.shape
assert target_mask.shape == modis_data.shape[:3]
assert np.isfinite(goes_data).all()
```

Also confirm:

- all splits are non-empty and non-overlapping;
- predictor and target dates match;
- scaling was applied once;
- masks contain valid weights;
- source paths and arrays use the same sorted date order;
- predictions inherit the correct reference grid; and
- outputs are physically plausible.

## Common problems

### No arrays can be stacked

No files passed the path, date, filename, time, or invalid-date filters. Print parsed dates and active globs.

### Input and target counts differ

A date is missing from one source or exclusions were applied inconsistently.

### Inputs contain NaN or infinity

Interpolation failed or a scene has no usable data. Reject or explicitly repair that scene.

### U-Net concatenation fails

The padded dimensions are incompatible with pooling depth or data and mask padding differ.

### R-squared is poor or negative

Check temporal pairing, spatial alignment, scaling, masks, target variance, and distribution shift before changing the model.

## Next steps

- Evaluate and visualize the first-stage results in `../GOES-Modis-Albedo-Postprocessing-main/`.
- Use first-stage prediction rasters in `../modis_s2_downscaling/`.
- Compare with the direct route in `../goes_s2_downscaling/`.

## Credits

The source retains acknowledgments and attribution from the original workflow. Portions of the U-Net implementation are identified in the code as modifications of an MIT-licensed implementation by Vidushi Bhatia. Preserve those notices.
