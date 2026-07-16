# MODIS-to-Sentinel-2 U-Net Downscaling

This directory contains the second-stage U-Net that uses first-stage MODIS-scale albedo predictions to estimate Sentinel-2 shortwave blue-sky albedo on the exact 20 m Sentinel-2 grid.

## Contents

| File | Responsibility |
|---|---|
| `run_modis_s2_unet2.ipynb` | Interactive driver for matching files, preparing tensors, training or loading the model, evaluating predictions, and reviewing outputs. |
| `modis_s2_unet2.py` | Main implementation of file matching, reprojection, masks, U-Net training, metrics, and output writing. |
| `packages.py` | Shared imports used by the script and notebook. |

This directory replaces the former `s2_modis_downscaling/` path.

## Model role

- **Predictor:** GOES-to-MODIS model prediction raster at MODIS scale.
- **Target:** Sentinel-2 20 m shortwave blue-sky albedo.
- **Sample weight:** finite-pixel mask from the Sentinel-2 target.
- **Output:** predicted Sentinel-2-grid albedo GeoTIFFs and NumPy arrays.

The current target filename pattern is:

```text
YYYY-MM-DD_S2_BLUE20m_SW_hard.tif
```

## Prerequisites

Complete these stages first:

1. produce first-stage MODIS-scale prediction rasters in `../goes_modis_downscaling/`;
2. produce Sentinel-2 targets in `../sentinel_2/`; and
3. create the TSI-derived cloud-fraction table used to screen scenes.

Activate the environment:

```bash
conda activate sail_env
```

A GPU is recommended.

## Configuration

`modis_s2_unet2.py` contains absolute Boise State paths. Review:

```text
UNET_RESULTS
TENSORFLOW_CHECKPOINT_PATH
TENSORFLOW_TRAINING_DIR
TF_HISTORY_PATH
TF_FELIX_MODEL_UNMASKED_PATH
cf_file
modis_train_path
modis_test_path
s2_train_path
s2_test_path
```

Also review:

- hard-coded invalid train and test dates;
- output destination;
- loading flags;
- training epochs and callbacks; and
- active train, validation, and test selection logic.

The module uses `from packages import *`. Run it from this directory or place this directory first on `sys.path` so the correct `packages.py` is imported.

## Cloud screening

The script reads `tsi_cloud_fractions.csv`, retains rows where:

```text
cf_interp <= 0.4
```

and removes duplicate dates. Confirm that `cf_interp` is present, uses the intended cloud-fraction definition, and corresponds to the same Sentinel-2 acquisitions.

## File matching

The script constructs expected predictor and target paths from each retained calendar date. Current predictor names follow a pattern similar to:

```text
predicted_YYYY-MM-DD_modis_blue_sky_albedo_.tif
```

Dates are excluded when they appear in the invalid-date lists or the predictor file is absent. Preserve the same ordered date pairing for tensors and reference paths.

## Spatial alignment

Each MODIS predictor is opened with Rioxarray and reprojected to match the exact Sentinel-2 target:

- CRS;
- affine transform;
- width;
- height; and
- pixel locations.

Continuous albedo uses bilinear resampling. The Sentinel-2 raster remains the georeferencing reference for saved predictions.

## Target mask and padding

Finite Sentinel-2 pixels receive weight 1; invalid pixels receive weight 0. Missing target values are filled with zero only for tensor computation.

Predictor, target, and mask arrays are padded so height and width are compatible with three U-Net pooling levels, normally to a multiple of eight. Padded mask pixels remain zero and predictions are cropped back before output.

## U-Net configuration

The current implementation uses:

- one predictor channel;
- three encoder pooling levels and corresponding decoder levels;
- 64 base filters;
- ReLU activations and skip connections;
- Adam optimizer with learning rate `1e-4`;
- Huber loss with `delta=0.05`;
- RMSE metric;
- early stopping;
- learning-rate reduction; and
- checkpoint saving.

The model accepts variable spatial dimensions provided they are padded compatibly with the pooling depth.

## Run the workflow

Notebook route:

```text
modis_s2_downscaling/run_modis_s2_unet2.ipynb
```

Script route from the repository root:

```bash
python modis_s2_downscaling/modis_s2_unet2.py
```

Before training, print and compare:

- matched predictor and target counts;
- ordered dates;
- tensor shapes;
- finite-pixel fractions; and
- predictor and target grids for several samples.

## Training and evaluation

The model uses the Sentinel-2 mask as per-pixel sample weights during training and validation. It reports masked RMSE and R-squared for padded and cropped predictions.

Loading behavior supports either constructing a new model, loading checkpoint weights, or loading a saved Keras model. Do not enable incompatible loading modes simultaneously.

## Outputs

The workflow may write:

- checkpoint weights;
- a saved Keras model;
- training-history JSON;
- cropped NumPy prediction arrays;
- masked RMSE and R-squared diagnostics; and
- GeoTIFFs named like:

```text
predicted_s2_<original Sentinel-2 filename>
```

The output GeoTIFFs use the corresponding Sentinel-2 CRS and transform.

## Validation checklist

Confirm:

```python
assert len(modis_paths) == len(s2_paths)
assert X.shape[0] == Y.shape[0] == mask.shape[0]
assert X.shape[1:3] == Y.shape[1:3] == mask.shape[1:3]
```

Also verify:

- cloud filtering and invalid dates are applied once;
- predictor and target dates match;
- MODIS is reprojected, not merely relabeled;
- target and padded-border masks are zero where invalid;
- output predictions are cropped to the reference dimensions;
- georeferencing matches the Sentinel-2 target; and
- albedo values are physically plausible.

## Common problems

### No matched files

Check cloud-table dates, filename templates, predictor output locations, and target naming.

### Predictor and target shapes differ

Use the exact Sentinel-2 raster as the `reproject_match` reference before stacking.

### Validation or sample weights have the wrong rank

Keras per-pixel weights should align with the target spatial dimensions. Inspect whether the active code expects `(N,H,W)` or `(N,H,W,1)`.

### U-Net concatenation fails

Pad both spatial dimensions to a multiple compatible with the pooling depth.

### Metrics are poor

Check first-stage prediction quality, temporal pairing, Sentinel-2 cloud and snow masks, spatial alignment, target variance, and train/test distribution.

## Related workflows

- First-stage predictions: `../goes_modis_downscaling/`
- Sentinel-2 targets: `../sentinel_2/`
- Direct GOES comparison model: `../goes_s2_downscaling/`
