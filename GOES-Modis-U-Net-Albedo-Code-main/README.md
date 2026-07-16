# GOES-to-MODIS U-Net Albedo Downscaling

This directory contains the interactive notebook for training, evaluating, or reloading the first-stage U-Net. The model learns a pixel-to-pixel mapping from preprocessed GOES-R ABI land-surface albedo to MODIS blue-sky albedo on the common 500 m East River grid.

## Contents

| File | Description |
|---|---|
| `run_unet.ipynb` | Notebook entry point for assembling date-matched datasets, training the model, evaluating predictions, and reviewing training history. |
| `.gitignore` | Directory-specific ignore rule. |

The notebook relies on `../functions/albedo_unet1_fxns.py`. A script-based equivalent is available at `../functions/unet1_main.py`.

## Model role

The first downscaling stage uses:

- **predictor:** GOES ABI Level-2 LSAC land-surface albedo, aligned to the MODIS grid;
- **target:** daily MODIS blue-sky albedo;
- **target weight:** a per-pixel validity mask derived from finite MODIS values; and
- **output:** predicted MODIS-scale albedo rasters and arrays for dates with GOES observations.

The U-Net performs image-to-image regression, not semantic segmentation.

## Prerequisites

Before running this notebook, complete the workflows in `../GOES-Modis-Data-Preprocessing-main/` and verify that the following are available:

- daily GOES 500 m-aligned GeoTIFFs;
- daily MODIS blue-sky albedo GeoTIFFs;
- optional GOES files masked to match the MODIS missing-pixel footprint;
- invalid-date JSON;
- output directories for checkpoints, models, history, predictions, and test-date metadata; and
- consistent filenames that preserve each acquisition date.

Create the environment from the repository root:

```bash
mamba env create -f environment.yml
conda activate sail_env
jupyter lab
```

A GPU is recommended for training. CPU execution is supported but can be considerably slower.

## Configure the helper module

Update the directory and output constants near the top of:

```text
../functions/albedo_unet1_fxns.py
```

The key settings are:

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

Update the notebook import path so it points to this repository:

```python
from pathlib import Path
import sys

repo_root = Path("/path/to/downscaling_geostationary_land_surface_albedo")
sys.path.append(str(repo_root / "functions"))

from albedo_unet1_fxns import *
```

## Data selection and date matching

The helper function `get_data_and_mask(...)`:

1. reads all GeoTIFFs from the configured input directory;
2. extracts dates from GOES or MODIS filenames;
3. restricts data to the requested interval;
4. removes dates listed in the invalid-date JSON and hard-coded invalid-date lists;
5. selects the preferred GOES observation time;
6. pairs MODIS dates with available GOES dates;
7. prepares arrays and target masks; and
8. returns the source paths used for each date.

### GOES time selection

The current logic prefers an observation in the 18 UTC hour. For dates listed in `INVALID_GOES_SOLAR_NOON_DATES`, a 19 UTC scene is accepted as a fallback. Dates listed in `INVALID_GOES_DATES_BOTH` are removed.

Review these lists for every new dataset or processing version.

### Filename assumptions

- GOES filenames must contain a start timestamp in the standard `_sYYYYDDDHHMM...` token.
- MODIS blue-sky filenames must begin with `YYYYDDD` before the first underscore.

Update `extract_goes_datetime(...)` or `extract_modis_datetime(...)` if the naming convention changes.

## Array preparation

### GOES predictors

GOES input pixels are converted to `float32`, missing values are filled by linear interpolation followed by forward/backward filling, and the two-dimensional image is padded.

### MODIS targets

The target workflow:

- records finite pixels in a mask;
- replaces target `NaN` values with zero only for tensor computation;
- pads the target and mask; and
- supplies the mask as per-pixel sample weights during training and evaluation.

Filled target pixels therefore do not contribute to the loss when their mask value is zero.

### Fixed East River dimensions

The current first-stage grid is approximately `21 x 19` pixels. The helper pads it to `24 x 24` using:

```text
rows:    1 top, 2 bottom
columns: 2 left, 3 right
```

Predictions are cropped back to `21 x 19` before saving.

For another AOI or grid, revise all of the following together:

- `pad_da_2d(...)`;
- `pad_mask_2d(...)`;
- `remove_padding(...)`;
- expected input dimensions; and
- georeferencing checks used when writing predictions.

The U-Net has three pooling levels, so padded height and width must be compatible with repeated factors of two.

## Model architecture

The current model uses:

- input shape `(24, 24, 1)`;
- four encoder blocks;
- three decoder blocks with skip connections;
- 64 base filters;
- `3 x 3` convolutions;
- ReLU activations;
- transposed-convolution upsampling;
- one output channel for continuous albedo;
- Adam optimizer with learning rate `1e-4`;
- Huber loss with `delta=0.05`; and
- root mean squared error as a Keras metric.

The encoder, decoder, and model-construction functions are identified in the source as modifications of an MIT-licensed U-Net implementation by Vidushi Bhatia. Preserve the embedded attribution and license notice.

## Training configuration

The scripted pipeline in `functions/unet1_main.py` currently uses:

| Split | Date interval |
|---|---|
| Training | 2021-09-01 through 2022-09-01 |
| Validation | 2022-09-02 through 2022-12-31 |
| Testing | 2023-01-01 through 2023-06-15 |

The notebook may contain the same or experimentally modified ranges. Confirm the active cells before running.

The training callback configuration includes:

- maximum of 500 epochs;
- early stopping on validation loss with patience 15;
- learning-rate reduction by a factor of 0.5 with patience 10;
- restoration of the best weights; and
- checkpoint saving to the configured `.weights.h5` path.

Random seeds are fixed in the scripted workflow for Python, NumPy, and TensorFlow. Exact reproducibility can still depend on hardware, TensorFlow version, and nondeterministic accelerator operations.

## Running the notebook

Recommended sequence:

1. restart the kernel;
2. import packages and helper functions;
3. verify all configured directories;
4. load and inspect invalid dates;
5. define non-overlapping train, validation, and test intervals;
6. choose masked or unmasked GOES input;
7. create predictor, target, and target-mask dictionaries;
8. stack them into model tensors;
9. inspect shapes and date counts;
10. assert that every predictor value is finite;
11. train from scratch, load checkpoint weights, or load a saved model;
12. evaluate on the test set;
13. save prediction arrays and GeoTIFFs; and
14. save the exact ordered test-date list.

## Running the script instead

From the repository root:

```bash
python functions/unet1_main.py
```

Although that file currently begins with a Bash-style shebang, its contents are Python and it should be invoked with Python.

Before running it, edit:

- `function_path`;
- `dest_folder`;
- date ranges;
- `goes_masked`;
- `load_weights_bool`;
- `load_model`; and
- model/result paths in `albedo_unet1_fxns.py`.

## Loading behavior

`run_unet(...)` supports three modes:

| `load_weights_bool` | `load_model` | Behavior |
|---|---|---|
| `False` | `False` | Build and train a new model. |
| `True` | `False` | Build the architecture and initialize it from checkpoint weights before continued fitting. |
| `False` | `True` | Load the saved Keras model before continued fitting. |

Do not set both loading options to `True`. Confirm that a checkpoint or model was produced with a compatible architecture and TensorFlow version.

## Outputs

The workflow writes some or all of the following:

- Keras model file;
- checkpoint weights;
- training-history JSON;
- test-date JSON;
- unpadded prediction array with shape `(N, 21, 19)` for the current AOI;
- predicted GeoTIFFs using MODIS reference grids;
- printed test loss and RMSE; and
- printed masked R-squared.

Prediction filenames are based on the reference MODIS filename and commonly begin with:

```text
predicted_
```

The NumPy output filename records train and test periods and whether masked GOES data was used.

## Validation checklist

Before accepting a training run, verify:

```python
assert goes_training_data_4d.shape == modis_training_data_4d.shape
assert modis_training_mask_3d.shape == modis_training_data_4d.shape[:3]
assert np.isfinite(goes_training_data_4d).all()
```

Also confirm:

- all three data splits are non-empty;
- predictor and target dates are identical within each split;
- date splits do not overlap;
- masks contain only valid weights, normally 0 and 1;
- test reference paths are sorted in the same date order as the test arrays;
- prediction values are checked for physical plausibility; and
- saved GeoTIFFs inherit the correct CRS and affine transform.

## Common problems

### `ValueError: need at least one array to stack`

No files passed the date, filename, invalid-date, or time filters. Print the parsed dates and inspect the configured directory.

### Predictor and target sample counts differ

One satellite is missing a date, filenames were parsed incorrectly, or invalid dates were not applied consistently. Build targets using the GOES date dictionary as the date gate.

### `NaN` or infinity in inputs

The interpolation step did not fill an edge or an entire raster is missing. Reject the date or provide a documented filling method rather than silently training with invalid values.

### Model shape error during concatenation

The input height or width is not compatible with the pooling depth. Pad both spatial dimensions to a suitable multiple and crop the prediction using the exact inverse operation.

### Poor or negative R-squared

Check spatial alignment, scaling, target masks, temporal matching, train/test distribution, and missing-data thresholds before changing architecture. A low RMSE can coexist with poor R-squared when the target's spatial variance is small.

### Saved predictions do not match their dates

The path list was not sorted using the same dates as the stacked dictionary. Always derive reference paths from the sorted target dictionary and save `test_dates.json`.

## Next step

Use `../GOES-Modis-Albedo-Postprocessing-main/Final-Visualizations.ipynb` to rasterize legacy NumPy outputs, calculate date-matched metrics, and generate figures.

## Credits

Thanks are retained from the original project documentation to Dr. Utkarsh Mital for U-Net support, Dr. Daniel Feldman for Earth-science workflow references, Dr. William Rudisill for coding and Earth-science expertise, and Lawrence Berkeley National Laboratory for computational infrastructure.