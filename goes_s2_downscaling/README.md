# Direct GOES-to-Sentinel-2 U-Net Downscaling

This directory contains a direct U-Net route from GOES albedo predictors to Sentinel-2 20 m shortwave blue-sky albedo targets. It provides an alternative to the two-stage GOES-to-MODIS-to-Sentinel-2 workflow.

## Contents

| File | Responsibility |
|---|---|
| `run_goes_s2_unet.ipynb` | Interactive driver for matching files, preparing arrays, training or loading the direct model, evaluating it, and reviewing outputs. |
| `goes_s2_unet.py` | Primary direct GOES-to-Sentinel-2 implementation. |
| `goes_s2_unet-Copy1.py` | Legacy or experimental copy with alternate GOES paths and filename assumptions. Use it as a reference rather than the default entry point. |
| `packages.py` | Shared imports used by the Python workflows in this directory. |

## Model role

- **Predictor:** GOES land-surface albedo reprojected to the exact Sentinel-2 grid.
- **Additional predictor channel:** GOES finite-pixel validity mask in the current primary implementation.
- **Target:** Sentinel-2 20 m shortwave blue-sky albedo.
- **Sample weight:** finite-pixel mask from the Sentinel-2 target.
- **Output:** direct Sentinel-2-grid albedo predictions.

The model performs continuous image-to-image regression.

## Prerequisites

Before running this workflow:

1. preprocess GOES rasters using the project GOES/MODIS preprocessing tools;
2. produce Sentinel-2 target rasters in `../sentinel_2/`;
3. create `tsi_cloud_fractions.csv`; and
4. verify that GOES and Sentinel-2 filenames can be matched by calendar date.

Activate the environment:

```bash
conda activate sail_env
```

A GPU is recommended for training.

## Configuration

`goes_s2_unet.py` contains absolute paths for:

```text
S2_BLUE_SKY_ALBEDO_DIR
GOES_ALBEDO_DIR
MASKED_GOES_ALBEDO_DIR
UNET_RESULTS
TENSORFLOW_CHECKPOINT_PATH
TENSORFLOW_TRAINING_DIR
TF_HISTORY_PATH
TF_FELIX_MODEL_UNMASKED_PATH
cf_file
goes_train_path
goes_test_path
s2_train_path
s2_test_path
```

Also review:

- invalid GOES and Sentinel-2 date lists;
- preferred and fallback GOES observation times;
- train/test slicing;
- output destination;
- loading flags;
- epochs and callback settings; and
- filename templates used to construct matched paths.

The module imports the local `packages.py`. Run from this directory or place it first on `sys.path`.

## Cloud and date screening

The primary script reads the TSI-derived cloud table, retains:

```text
cf_interp <= 0.4
```

and removes duplicate dates. It then excludes hard-coded invalid dates and retains dates with both GOES and Sentinel-2 files.

The current top-level matching logic uses the final 12 retained pairs as the test set and preceding pairs as training candidates. Review this split explicitly whenever the file inventory changes.

## Filename conventions

The primary script currently expects Sentinel-2 targets such as:

```text
YYYY-MM-DD_S2_BLUE20m_SW_hard.tif
```

and constructs GOES paths such as:

```text
YYYY-MM-DD-GOES-500m.tif
```

Other helper functions also contain parsers for original GOES `_sYYYYDDDHHMM...` names. Ensure the active matching block and the input directory use the same convention.

`goes_s2_unet-Copy1.py` contains alternate masked-GOES paths and naming assumptions. Do not mix its conventions with the primary script without updating the full pairing logic.

## Spatial alignment

For every pair, the GOES raster is reprojected to match the Sentinel-2 target's:

- CRS;
- transform;
- width;
- height; and
- pixel coordinates.

Bilinear resampling is used for continuous GOES albedo. The Sentinel-2 target is the reference for saved outputs.

## Input channels and masks

The current primary loader builds a two-channel predictor:

1. GOES albedo with missing values filled for computation; and
2. a GOES validity channel containing 1 for finite source pixels and 0 for missing or padded pixels.

The Sentinel-2 target mask is separate and controls the loss. Finite target pixels receive weight 1; invalid target and padded pixels receive weight 0.

Predictor, target, and mask arrays are padded to dimensions compatible with the U-Net pooling depth and cropped before output.

## U-Net configuration

The active implementation includes:

- two predictor channels in the aligned-stack route;
- convolutional encoder and decoder blocks with skip connections;
- dropout regularization in encoder blocks;
- Adam optimizer at approximately `1e-4`;
- Huber loss with `delta=0.05`;
- RMSE metric;
- early stopping with restored best weights;
- learning-rate reduction; and
- checkpoint and full-model saving.

The training configuration currently permits up to 150 epochs and repeats a new-model run until a minimum training-history length is reached. Inspect this experimental control flow before production runs.

## Run the workflow

Notebook route:

```text
goes_s2_downscaling/run_goes_s2_unet.ipynb
```

Script route:

```bash
python goes_s2_downscaling/goes_s2_unet.py
```

Before training, print:

- paired filenames and dates;
- train, validation, and test counts;
- predictor channel count;
- tensor and mask shapes;
- finite-pixel fractions; and
- source and target grids for representative dates.

## Evaluation

The script calculates masked RMSE and R-squared for padded and cropped arrays. The target validity mask must be used consistently for training, validation, and evaluation.

Loading behavior supports a new model, checkpoint weights, or a saved Keras model. Verify architecture compatibility, especially the expected input-channel count.

## Outputs

The workflow may write:

- checkpoint weights;
- a saved Keras model;
- training-history JSON;
- cropped NumPy prediction arrays;
- masked RMSE and R-squared diagnostics; and
- GeoTIFFs georeferenced from the corresponding Sentinel-2 targets.

Prediction filenames use a prefix similar to:

```text
predicted_s2_
```

## Validation checklist

Confirm:

```python
assert len(goes_paths) == len(s2_paths)
assert X.shape[0] == Y.shape[0] == mask.shape[0]
assert X.shape[-1] == 2
```

Also verify:

- cloud and invalid-date rules are applied once;
- date ordering is preserved across paths and tensors;
- GOES is reprojected to the exact Sentinel-2 grid;
- the validity channel correctly marks missing and padded GOES pixels;
- target sample weights are zero for invalid and padded pixels;
- predictions are cropped to the reference shape; and
- outputs inherit the Sentinel-2 CRS and transform.

## Common problems

### No matched pairs

Check the cloud-table dates, GOES filename template, Sentinel-2 pattern, and active input directories.

### Model expects the wrong number of channels

The aligned primary loader creates two channels. Ensure `get_UNet_model(...)`, saved weights, and input arrays all use the same channel count.

### Predictor and target shapes differ

Reproject GOES with the Sentinel-2 raster as the exact reference before padding.

### Validation weights fail

Inspect the expected rank of sample weights and ensure padded borders remain zero.

### Poor direct-model performance

Check GOES temporal selection, source resolution, alignment, missing-data channel, cloud screening, target variance, date split, and seasonal distribution.

## Related workflows

- GOES preprocessing and first-stage modules: `../goes_modis_downscaling/`
- Sentinel-2 target production: `../sentinel_2/`
- Two-stage 20 m model: `../modis_s2_downscaling/`
