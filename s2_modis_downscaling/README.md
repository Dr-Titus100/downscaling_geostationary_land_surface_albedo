# MODIS-to-Sentinel-2 U-Net Downscaling

This directory contains the second neural-network stage in the project. It learns a mapping from MODIS-scale albedo predictions to Sentinel-2-derived 20 m albedo targets.

The workflow is intended to add Sentinel-2 spatial structure after the first U-Net has transformed GOES observations to a MODIS-like 500 m product.

## Contents

| File | Description |
|---|---|
| `modis_s2_unet2.py` | Script implementation for date matching, cloud filtering, grid alignment, U-Net training, evaluation, and GeoTIFF export. |
| `modis_s2_unet2.ipynb` | Interactive notebook version used for development, inspection, and experiments. |

The Python script is the clearest reproducible entry point. Review the notebook for experimental cells before relying on its active state.

## Inputs

The script expects:

1. predicted MODIS blue-sky albedo GeoTIFFs from the first U-Net;
2. Sentinel-2 shortwave albedo GeoTIFFs with filenames ending in `_S2_BLUE20m_SW_hard.tif`;
3. a cloud-fraction CSV containing at least `date` and `cf_interp` columns;
4. lists of dates to exclude from training and testing; and
5. output directories for checkpoints, models, histories, NumPy arrays, and predicted Sentinel-2-grid GeoTIFFs.

The current file globs point to Boise State scratch directories and must be replaced.

## Current scene filtering

The script reads:

```text
tsi_cloud_fractions.csv
```

and retains rows where:

```python
cf_interp <= 0.4
```

It then drops duplicate dates and matches the remaining calendar dates to both MODIS-prediction and Sentinel-2 target filenames.

Review the threshold, interpolation method used to create `cf_interp`, and duplicate-date rule before each experiment. A cloud fraction of 0.4 means that scenes with up to 40 percent interpolated cloud fraction are retained under the current rule.

## Configure paths

Edit the constants and file patterns near the top of `modis_s2_unet2.py`:

```text
UNET_RESULTS
TENSORFLOW_CHECKPOINT_PATH
TENSORFLOW_TRAINING_DIR
TF_HISTORY_PATH
TF_FELIX_MODEL_UNMASKED_PATH
cf_file
modis_train_path
s2_train_path
modis_test_path
s2_test_path
```

Also update every fully constructed filename inside the matching loops. The current code assumes exact patterns such as:

```text
predicted_YYYY-MM-DD_modis_blue_sky_albedo_.tif
YYYY-MM-DD_S2_BLUE20m_SW_hard.tif
```

A mismatch in underscores, date format, or suffix prevents a pair from being found.

## Data pairing

For every retained Sentinel-2 date, the script constructs the corresponding predicted MODIS path and Sentinel-2 target path. A date is used only when the required MODIS prediction exists and the date is not in the exclusion list.

The script prints the number of matched files for both datasets. These counts must be equal before training.

Recommended additional check:

```python
assert len(modis_unet_train_files_list) == len(s2_unet_train_files_list)
for modis_path, s2_path in zip(modis_unet_train_files_list, s2_unet_train_files_list):
    print(Path(modis_path).name, Path(s2_path).name)
```

## Grid alignment

`load_stacks_from_lists(...)` performs the spatial preparation for each pair:

1. open MODIS and Sentinel-2 rasters as `float32`;
2. select band 1;
3. reproject and resample MODIS to the exact Sentinel-2 CRS, transform, width, and height using bilinear interpolation;
4. create a validity mask from finite Sentinel-2 pixels;
5. replace `NaN` with zero for tensor computation;
6. pad both input and target arrays; and
7. preserve the Sentinel-2 path for output georeferencing.

The target mask is padded with zeros so synthetic border pixels do not affect loss or metrics.

## Padding behavior

The script contains two padding layers:

- a fixed `pad_da_2d(...)` operation inherited from the first U-Net workflow; and
- `pad_batch_to_multiple(...)`, which symmetrically pads a batch to a multiple of eight.

The current comments still describe an original `21 x 19` grid in places, but Sentinel-2 target dimensions depend on the actual AOI and resolution. Inspect shapes printed by the script and verify that fixed padding is appropriate for the current data.

`crop_pred(...)` removes the batch-level multiple-of-eight padding before metrics and GeoTIFF export.

## Model architecture

The second-stage model uses a fully convolutional U-Net with:

- input shape `(None, None, 1)`;
- four encoder blocks and three decoder blocks;
- 64 base filters;
- ReLU activations;
- skip connections;
- Adam optimizer at learning rate `1e-4`;
- Huber loss with `delta=0.05`; and
- RMSE as a Keras metric.

The model can accept varying spatial dimensions as long as each batch has a consistent padded shape and the dimensions are compatible with the pooling depth.

The final model currently returns `conv9`, a single-channel linear output. A subsequent `conv10` layer is defined but not used as the model output.

## Train, validation, and test split

`main(valid_test_dates=False)` currently supports two paths.

### `valid_test_dates=False`

- takes the last 30 percent of the matched training list as the test set;
- uses the remaining 70 percent for training; and
- currently assigns the same remaining samples to validation and training.

Using the training data as validation data does not provide an independent estimate of generalization. For a formal experiment, create non-overlapping train, validation, and test date groups.

### `valid_test_dates=True`

- uses the separately constructed test lists;
- retains the full matched training lists for training.

Review variable scope in this branch before use. The current code later references `s2_train2`, which is only created in the `False` branch. Test this mode on a small run and correct the split logic if needed before production use.

## Training behavior

`run_unet(...)` supports training from scratch, loading weights, or loading a complete Keras model.

Current callback settings:

```text
maximum epochs: 100
early stopping patience: 15
learning-rate reduction patience: 10
learning-rate reduction factor: 0.5
```

Sentinel-2 validity masks are passed as per-pixel sample weights. After prediction, the script calculates masked RMSE and masked R-squared on both padded and cropped arrays.

## Run the script

From the repository root:

```bash
conda activate sail_env
python s2_modis_downscaling/modis_s2_unet2.py
```

Before the full run:

- confirm all matched file counts;
- print the first and last paired dates;
- use a very small subset for a smoke test;
- inspect input, target, and mask plots;
- confirm enough independent scenes remain after cloud and exclusion filters; and
- verify GPU availability if training on an accelerator node.

## Outputs

The workflow writes:

- checkpoint weights;
- a saved Keras model;
- training-history JSON;
- predicted arrays in `.npy` format;
- masked RMSE and R-squared summaries; and
- one GeoTIFF per test scene.

GeoTIFF filenames begin with:

```text
predicted_s2_
```

Each output copies the CRS and transform from the corresponding Sentinel-2 target raster.

## Validation checklist

Before interpreting results, verify:

```python
assert X_train.shape == Y_train.shape
assert M_train.shape[:3] == Y_train.shape[:3]
assert np.isfinite(X_train).all()
```

Also verify:

- each pair represents the same date;
- MODIS and Sentinel-2 albedo use the same scale and units;
- MODIS has been reprojected to the exact target grid;
- target masks exclude `NaN` and padded pixels;
- train, validation, and test periods are independent;
- file order is deterministic;
- output rasters match the target width, height, transform, and CRS; and
- evaluation metrics use only mask-valid target pixels.

## Common problems

### Zero matched scenes

Check the cloud CSV dates, filename patterns, exclusion lists, directory globs, and whether the first-stage predictions were generated for the same Sentinel-2 acquisition dates.

### Concatenation shape error

The spatial dimensions are not compatible after pooling and upsampling. Use `pad_batch_to_multiple(..., mult=8)` consistently for input, target, and mask.

### GeoTIFF shape does not match the reference

Crop only the padding added during tensor preparation. Compare the predicted array with the Sentinel-2 reference before writing.

### Validation score is misleading

The current default uses the training scenes as validation scenes. Create a date-based held-out validation set.

### Metrics are `NaN`

The target mask may contain no valid pixels, or the target may have no variance. Inspect the scene and cloud filter rather than forcing it into the evaluation.

### Script fails when `valid_test_dates=True`

Review the train-split variables in `main()`. The current branch structure requires testing and may need a consistent definition of `s2_train2` and `modis_train2`.

## Relationship to other directories

- First-stage MODIS predictions come from `../GOES-Modis-U-Net-Albedo-Code-main/` and `../functions/`.
- Sentinel-2 targets and cloud information come from `../sentinel_2/`.
- The resulting 20 m predictions can be evaluated and visualized with the raster and plotting conventions used elsewhere in the repository.

## Reproducibility recommendations

For each run, save:

- matched date table;
- cloud-fraction threshold and source;
- excluded-date lists;
- exact train/validation/test dates;
- input and target filename patterns;
- padding details;
- repository commit;
- TensorFlow version and hardware; and
- checkpoint and output paths.