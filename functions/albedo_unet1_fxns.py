#!/usr/bin/env python3
## Objective
## Assemble predictors for various data combinations and train U-Net models

# Note some hard code things
import os
import re
import sys
import time
import json
import random
import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
import rioxarray as rxr
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
from tensorflow.keras import Input
from datetime import datetime, timedelta
from tensorflow.keras.models import Model
from rasterio.plot import plotting_extent
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Input, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D, ReLU, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Directories
MODIS_BLUE_SKY_ALBEDO_DIR = "/bsuhome/tnde/scratch/felix/modis/blue_sky_albedo_sail_new/"
GOES_ALBEDO_DIR = "/bsuhome/tnde/scratch/felix/GOES/data/goes_output_data_new/"
MASKED_GOES_ALBEDO_DIR = "/bsuhome/tnde/scratch/felix/GOES/data/nan_data_new/"

UNET_RESULTS = "/bsuhome/tnde/scratch/felix/UNet/Results_new/"
TENSORFLOW_CHECKPOINT_PATH = "/bsuhome/tnde/scratch/felix/UNet/Results_new/training/cp.weights.h5"
TENSORFLOW_TRAINING_DIR = "/bsuhome/tnde/scratch/felix/UNet/Results_new/training/"
TF_HISTORY_PATH = "/bsuhome/tnde/scratch/felix/UNet/Results_new/training/history_new.json"
INVALID_DATES_PATH = "/bsuhome/tnde/scratch/felix/modis/invalid_modis_dates_new.json"
TF_FELIX_MODEL_UNMASKED_PATH = "/bsuhome/tnde/scratch/felix/UNet/Results_new/training/felix-model-aug-8-2024-unmasked_new.keras"

# Data
INVALID_GOES_SOLAR_NOON_DATES = [datetime(2022, 1, 5), datetime(2022, 1, 25), datetime(2022, 2, 8), datetime(2022, 2, 13), 
                                 datetime(2022, 2, 21), datetime(2022, 12, 14), datetime(2022, 12, 23)]
INVALID_GOES_AQUA_DATES = [datetime(2022, 1, 2), datetime(2022, 1, 3), datetime(2022, 1, 7), datetime(2022, 12, 17), datetime(2023, 1, 5), datetime(2023, 1, 31)]
# Investigate why MODIS has no data on 04/08/2022, 05/11/2023, 03/14/2023
INVALID_GOES_DATES_BOTH = [datetime(2022, 12, 16), datetime(2022, 12, 18), datetime(2022, 12, 19), datetime(2022, 4, 8), datetime(2023, 1, 7), datetime(2023, 1, 8), datetime(2023, 5, 11), datetime(2023, 3, 14)]

kernel = 3

############################################################################
############################################################################
# NEW: robust raster loader (uses rioxarray)
def load_raster_da(path: str | Path) -> xr.DataArray:
    """
    Load GeoTIFF as a DataArray with dims ('band','y','x').
    Returns band-1 if single band; keeps CRS/transform via rioxarray.
    """
    da = rxr.open_rasterio(path)  # DataArray, dims: (band, y, x)
    # Ensure float32 for learning stability
    da = da.astype("float32")
    return da

def pad_da_2d(da2d: xr.DataArray) -> np.ndarray:
    """
    Accept a 2D (y,x) DataArray and pad to (24,24) using reflection.
    Original target ~ (21,19).
    """
    arr = da2d.values  # (y,x)
    padded = np.pad(arr, pad_width=((1, 2), (2, 3)), mode="reflect")
    return padded  # (24,24)

def pad_mask_2d(mask2d: np.ndarray) -> np.ndarray:
    """
    Pad a boolean/float mask the same way as the data.
    """
    return np.pad(mask2d, pad_width=((1, 2), (2, 3)), mode="edge")

def remove_padding(np_data_in: np.ndarray) -> np.ndarray:
    """
    Keep original logic but clarified; works for (N,H,W) or (N,H,W,1).
    """
    top, bottom = 1, np_data_in.shape[1] - 2
    left, right = 2, np_data_in.shape[2] - 3
    if np_data_in.ndim == 3:
        return np_data_in[:, top:bottom, left:right]
    elif np_data_in.ndim == 4:
        return np_data_in[:, top:bottom, left:right, :]
    raise ValueError("Expected 3D or 4D array.")

############################################################################
############################################################################
def fill_inputs_interpolate(da2d: xr.DataArray) -> xr.DataArray:
    """
    For INPUTS (GOES) only: bilinear interp over x/y, then ffill/bfill.
    """
    # Reindex to itself to ensure coordinates are monotonic
    da = da2d.copy()
    da = da.interp(x=da.x, y=da.y, method="linear")
    da = da.ffill("x").ffill("y").bfill("x").bfill("y")
    return da

def prepare_target_and_mask(da2d: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    """
    For TARGETS (MODIS): build mask = valid (non-NaN), fill NaNs with 0 for compute,
    pad both data and mask to (24,24).
    """
    arr = da2d.values  # (y,x)
    mask = np.isfinite(arr).astype("float32")  # 1=valid, 0=invalid
    arr_filled = np.nan_to_num(arr, nan=0.0)

    arr_filled = pad_da_2d(xr.DataArray(arr_filled, dims=("y","x")))
    mask = pad_mask_2d(mask)
    return arr_filled, mask

############################################################################
############################################################################
def extract_goes_datetime(filename: str) -> datetime:
    """
    Extract the GOES observation datetime from the filename.

    The function parses filenames following the GOES naming convention, such as:
    'OR_ABI-L2-LSAC-M6_G16_s20231631826173_e20231631828546_c20231631830241_clipped_reprojected.tif'
    
    It extracts the start time segment (e.g., 's20231631826173') and converts it to a datetime object.

    Parameters
    ----------
    filename : str
        The full name of the GOES file.

    Returns
    -------
    datetime
        Python datetime object representing the start time of the observation.

    Notes
    -----
    - The timestamp format extracted is 'YYYYJJJHHMM' (Year, Julian day, Hour, Minute).
    - This implementation uses only the start timestamp (denoted by 's') in the filename.
    """
    # Extract the start datetime portion from the filename (e.g., 's20231631826173' -> '20231631826')
    start_timestamp_str = filename.split("_")[3][1:-3]  # Remove 's' and last 3 digits (seconds)

    # Convert to datetime object: format is 'YYYYJJJHHMM'
    return datetime.strptime(start_timestamp_str, '%Y%j%H%M')

#######################################################
def extract_modis_datetime(filename: str) -> datetime:
    """
    Extract the MODIS observation date from the filename.

    The function expects the filename to begin with a date string in 'YYYYJJJ' format,
    where 'YYYY' is the 4-digit year and 'JJJ' is the Julian day of year.

    Example filename:
    '2022272_modis_blue_sky_albedo_.tif' --> Extracts '2022272' and converts it to datetime.

    Parameters
    ----------
    filename : str
        The MODIS filename containing a leading date in 'YYYYJJJ' format.

    Returns
    -------
    datetime
        Python datetime object corresponding to the MODIS acquisition date.
    """
    # Extract the date component from the filename (first element before underscore)
    date_str = filename.split("_")[0]

    # Convert to datetime object using format 'YYYYJJJ'
    return datetime.strptime(date_str, '%Y%j')

############################################################################
############################################################################
def get_data_and_mask(
    date_start: datetime,
    date_finish: datetime,
    goes_date_gate: set[datetime] | dict,
    invalid_dates: list[datetime],
    is_goes: bool,
    use_masked_goes_dir: bool
) -> tuple[Dict[datetime, np.ndarray], Dict[datetime, np.ndarray] | None, Dict[datetime, Path]]:
    """
    Extract and preprocess training or test data from GOES or MODIS albedo files.

    This function:
    - Filters files by date range
    - Skips files with excessive invalid data
    - Interpolates NaNs and pads the input for U-Net compatibility
    - Handles date-specific logic for GOES (18:30 vs. 19:30 fallback)
    - Reports missing dates in the expected date range

    Parameters
    ----------
    date_start : datetime
        Start date (inclusive) of the desired date range.
    date_finish : datetime
        End date (inclusive) of the desired date range.
    goes_dataset : List[datetime]
        List of valid MODIS dates that match with GOES dates (used when `goes=False`).
    invalid_dates : List[datetime]
        Dates that should be excluded due to excessive NaNs.
    is_goes : bool
        If True, processes GOES data. If False, processes MODIS data.
    use_masked_goes_dir : bool
        If True, uses masked GOES data directory. Ignored for MODIS.

    Returns
    -------
    Dict[datetime, np.ndarray, paths_out]
        Dictionary mapping each valid truncated date to its padded, preprocessed 2D image array.
        Dictionary with key as date and values as the image numpy values
        Path to input files

    Notes
    -----
    - GOES files are filtered based on observation hour:
        - 18:00 (18:30 UTC): preferred
        - 19:00 (19:30 UTC): used as fallback if 18:30 is invalid
    - MODIS files are assumed to have a single image per date.
    - NaNs are interpolated using bilinear interpolation and padded to match model input shape.
    
    Thus:
    - List of dates with invalid data at 18:30 (use 19:30 instead).
    - List of dates with invalid data at both times (skip)
    - If training: Goes bool is true. If test: modis: Goes bool is false

    Returns: (data_dict, mask_dict, paths_out_dict). For GOES, mask_dict=None.
    For MODIS, mask_dict is per-pixel 0/1 weights (same padding as data).
    """
    data_out: Dict[datetime, np.ndarray] = {}
    mask_out: Dict[datetime, np.ndarray] | None = ({} if not is_goes else None)
    paths_out: Dict[datetime, Path] = {}

    dir_path = (MASKED_GOES_ALBEDO_DIR if (is_goes and use_masked_goes_dir) else
                GOES_ALBEDO_DIR if is_goes else
                MODIS_BLUE_SKY_ALBEDO_DIR)
    files = list(Path(dir_path).glob("*.tif"))

    for fp in files:
        if is_goes:
            dt = extract_goes_datetime(fp.name)
        else:
            dt = extract_modis_datetime(fp.name)
        d = datetime(dt.year, dt.month, dt.day)

        if not (date_start <= d <= date_finish):
            continue
        if d in invalid_dates:
            continue

        # GOES-specific hour logic
        if is_goes:
            # prefer 18:xx unless flagged; otherwise accept 19:xx replacement
            if dt.hour == 18 and d in INVALID_GOES_SOLAR_NOON_DATES:
                continue
            if dt.hour == 19 and d not in INVALID_GOES_SOLAR_NOON_DATES:
                continue

        # Load raster
        da = load_raster_da(fp)
        # Use band 1 (convert to 2D)
        if "band" in da.dims:
            da2d = da.sel(band=1, drop=True)
        else:
            # Some rasters load without 'band', ensure 2D
            da2d = da

        if is_goes:
            # Inputs: interpolate NaNs, then pad
            da2d_filled = fill_inputs_interpolate(da2d)
            data_out[d] = pad_da_2d(da2d_filled)
            paths_out[d] = fp
        else:
            # Targets: only keep days that appear in GOES set
            if d not in goes_date_gate:
                continue
            arr_filled, mask = prepare_target_and_mask(da2d)
            data_out[d] = arr_filled
            mask_out[d] = mask
            paths_out[d] = fp   

    # Missing-date report (unchanged idea)
    expected = set(date_start + timedelta(days=i) for i in range((date_finish - date_start).days + 1))
    present = set(data_out.keys())
    missing = sorted(expected - present)
    # print("All invalid or missing dates:")
    # for dd in missing:
    #     print(dd.strftime("%Y-%m-%d"))

    return data_out, mask_out, paths_out

############################################################################
############################################################################
def stack_array_4d(data_in: Dict[datetime, np.ndarray]) -> np.ndarray:
    """
    Stack a dictionary of 2D arrays into a 4D NumPy array suitable for U-Net input.
    Turn data into 4D array stacked like (num_samples, height, width, channels) for U-Net.

    This function:
    - Sorts the input dictionary by date keys
    - Adds a channel dimension to each 2D array (i.e., expands to shape HxWx1)
    - Stacks all arrays along a new sample axis to form a 4D array with shape:
      (num_samples, height, width, channels)

    Parameters
    ----------
    data_in : Dict[datetime, np.ndarray]
        Dictionary where each key is a date and each value is a 2D NumPy array
        (e.g., padded blue sky albedo data).

    Returns
    -------
    np.ndarray
        A 4D NumPy array of shape (num_samples, height, width, channels), suitable for
        input into a convolutional neural network like U-Net.

    Notes
    -----
    - Assumes all 2D arrays in the dictionary have the same shape.
    - The dictionary is sorted by date to ensure temporal consistency in stacking.
    """
    # Sort dictionary by date keys
    sorted_data = {k: data_in[k] for k in sorted(data_in)}
    # Add a channel dimension to each 2D array (HxW -> HxWx1)
    vals = [np.expand_dims(np.array(v), axis=-1) for v in sorted_data.values()]  # (H,W,1)
    return np.stack(vals, axis=0)  # (N,H,W,1) # Stack into a 4D array: (N, H, W, 1)

#######################################################
def stack_masks_3d(mask_in: Dict[datetime, np.ndarray]) -> np.ndarray:
    # (N,H,W) float32 mask
    sorted_mask = {k: mask_in[k] for k in sorted(mask_in)}
    vals = [np.array(v, dtype="float32") for v in sorted_mask.values()]
    return np.stack(vals, axis=0) 

############################################################################
############################################################################
def remove_padding(np_data_in: np.ndarray) -> np.ndarray:
    """
    Remove fixed padding from a 4D NumPy array along the height and width dimensions.

    This function removes:
    - 1 row from the top
    - 2 rows from the bottom
    - 2 columns from the left
    - 3 columns from the right

    Assumes input array is of shape (N, H, W) or (N, H, W, C), where:
    - N is the number of samples
    - H is the height (with padding)
    - W is the width (with padding)
    - C is the optional channel dimension

    Parameters
    ----------
    np_data_in : np.ndarray
        A padded NumPy array of shape (N, H, W) or (N, H, W, C)

    Returns
    -------
    np.ndarray
        Array with padding removed. 
        Shape becomes (N, H-3, W-5) for 3D input, or (N, H-3, W-5, C) for 4D input.

    Example
    -------
    Input shape : (356, 24, 24) -> Output shape: (356, 21, 19)
    """
    # Define trimming indices
    top = 1
    bottom = np_data_in.shape[1] - 2
    left = 2
    right = np_data_in.shape[2] - 3

    if np_data_in.ndim == 3:
        trimmed = np_data_in[:, top:bottom, left:right]
    elif np_data_in.ndim == 4:
        trimmed = np_data_in[:, top:bottom, left:right, :]
    else:
        raise ValueError("Input must be a 3D or 4D NumPy array.")

    return trimmed


#####################################################################################################################################
### The following three functions: EncoderMiniBlock, DecoderMiniBlock, and get_UNet_model are a modification of the code published here:
### https://github.com/VidushiBhatia/U-Net-Implementation/blob/main/U_Net_for_Image_Segmentation_From_Scratch_Using_TensorFlow_v4.ipynb
### These functions are subject to the following license:

# MIT License

# Copyright (c) 2021 Vidushi Bhatia

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

############################################################################
############################################################################
def EncoderMiniBlock(inputs, n_filters=64, max_pooling=True):
    """
    Builds an encoder mini-block consisting of two convolutional layers followed optionally by max pooling.
    This block uses multiple convolution layers, max pool, relu activation to create an architecture for learning. 
    The block returns the activation values for next layer along with a skip connection which will be used in the decoder.

    This block is commonly used in U-Net-like architectures. It performs two convolution operations 
    with ReLU activation and same padding to preserve spatial dimensions. If `max_pooling` is True,
    a max pooling operation is applied to reduce the spatial dimensions by a factor of 2.

    Parameters
    ----------
    inputs : tf.Tensor
        Input tensor to the encoder block.
    n_filters : int, optional
        Number of filters for the convolutional layers (default is 64).
    max_pooling : bool, optional
        Whether to apply max pooling after convolutions (default is True).

    Returns
    -------
    next_layer : tf.Tensor
        Output tensor passed to the next encoder block (pooled or convolved).
    skip_connection : tf.Tensor
        Tensor used as a skip connection for the decoder path in a U-Net architecture.
    
    Example
    -------
        inputs = tf.keras.Input(shape=(128, 128, 3))
        next_layer, skip = encoder_mini_block(inputs, n_filters=32, max_pooling=True)
    """
    
    # First convolution
    conv = Conv2D(n_filters, 
                  kernel,   # Kernel size   
                  activation='relu',
                  padding='same')(inputs)
    # Second convolution
    conv = Conv2D(n_filters, 
                  kernel,   # Kernel size
                  activation='relu',
                  padding='same')(conv)
    
    # Optional max pooling
    if max_pooling:
        next_layer = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(conv)    
    else:
        next_layer = conv

    # Store intermediate output for skip connection
    skip_connection = conv

    return next_layer, skip_connection

############################################################################
############################################################################
def DecoderMiniBlock(prev_layer_input, skip_layer_input, n_filters=64,
                       padding='same', strides=(2, 2), kernel_size=(3, 3)):
    """
    Builds a decoder mini-block typically used in U-Net architectures.
    
    Decoder Block first uses transpose convolution to upscale the image to a bigger size and then,
    merges the result with skip layer results from encoder block
    Adding 2 convolutions with 'same' padding helps further increase the depth of the network for better predictions
    The function returns the decoded layer output

    This block performs the following operations:
    1. Upsamples the previous layer using a transposed convolution (deconvolution).
    2. Concatenates the upsampled feature map with the corresponding skip connection from the encoder.
    3. Applies two convolutional layers with ReLU activation to refine the features.

    Parameters
    ----------
    prev_layer_input : tf.Tensor
        The input tensor from the previous decoder block (or the bottleneck layer).
    skip_layer_input : tf.Tensor
        The skip connection tensor from the corresponding encoder block.
    n_filters : int, optional
        Number of filters for the convolutional layers (default is 64).
    padding : str, optional
        Padding method for convolutions and transposed convolutions ('same' or 'valid', default is 'same').
    strides : tuple, optional
        Stride size for the transposed convolution (default is (2, 2)).
    kernel_size : tuple, optional
        Kernel size for the transposed and regular convolutions (default is (3, 3)).

    Returns
    -------
    tf.Tensor
        The output tensor after upsampling, concatenation, and convolution operations.

    Example
    -------
        x = tf.keras.Input(shape=(32, 32, 128))
        skip = tf.keras.Input(shape=(64, 64, 64))
        decoded = decoder_mini_block(x, skip, n_filters=64)
    """

    # Upsample using transposed convolution
    up = Conv2DTranspose(
                 n_filters,
                 kernel_size=kernel,    # Kernel size
                 strides=strides,
                 padding=padding)(prev_layer_input)

    # Concatenate with the skip connection from the encoder
    merge = concatenate([up, skip_layer_input], axis=3)
    
    # Apply two convolutional layers to refine features
    conv = Conv2D(n_filters, 
                 kernel,     # Kernel size
                 activation='relu',
                 padding='same')(merge)
    conv = Conv2D(n_filters,
                 kernel,   # Kernel size
                 activation='relu',
                 padding='same')(conv)
    return conv

############################################################################
############################################################################
def get_UNet_model(input_size):
    """
    Constructs and compiles a U-Net model for image-to-image regression tasks (e.g., albedo estimation).

    The U-Net architecture consists of:
    - An encoder path (downsampling) using `EncoderMiniBlock` with optional max pooling
    - A decoder path (upsampling) using `DecoderMiniBlock` with skip connections
    - A final convolutional layer to produce the regression output

    Parameters
    ----------
    input_size : tuple
        Shape of the input images in the format (height, width, channels), e.g., (24, 24, 1).

    Returns
    -------
    tf.keras.Model
        A compiled U-Net model with mean squared error loss and root mean squared error as a metric.

    Example
    -------
        model = get_unet_model((24, 24, 1))
        model.summary()
    """
    # Reset any previous model sessions to avoid cluttered graphs or memory leaks
    tf.keras.backend.clear_session()
    n_classes = 1 # Single-channel regression output
    n_filters = 64
    inputs = Input(input_size)

    # Encoder path
    cblock1 = EncoderMiniBlock(inputs, n_filters,   max_pooling=True)
    cblock2 = EncoderMiniBlock(cblock1[0], n_filters*2, max_pooling=True)
    cblock3 = EncoderMiniBlock(cblock2[0], n_filters*4, max_pooling=True)
    cblock4 = EncoderMiniBlock(cblock3[0], n_filters*8, max_pooling=False)

    # Decoder path with skip connections
    ublock1 = DecoderMiniBlock(cblock4[0], cblock3[1], n_filters*4)
    ublock2 = DecoderMiniBlock(ublock1,    cblock2[1], n_filters*2)
    ublock3 = DecoderMiniBlock(ublock2,    cblock1[1], n_filters)

    # Final convolutional layers to reduce channels to output
    conv8  = Conv2D(n_filters, 3, activation='relu', padding='same')(ublock3)
    conv9  = Conv2D(n_classes, 1, padding='same')(conv8)
    conv10 = Conv2D(n_classes, 1, padding='same')(conv9)

    # Define and compile the model
    model = Model(inputs=inputs, outputs=conv10)

    opt = Adam(learning_rate=1e-4)
    loss = tf.keras.losses.Huber(delta=0.05)  # robust, reduces blur vs MSE
    model.compile(optimizer=opt,
                  loss=loss,
                  metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")])
    return model

# End of the code under the above MIT license.
#######################################################
#######################################################
def save_preds_as_geotiff(preds_unpadded: np.ndarray, ref_paths: List[Path], out_dir: str):
    """
    Save each prediction (H,W) as GeoTIFF using the corresponding MODIS file's
    CRS/transform and filename. Output name: predicted_<original_filename>.tif
    """
    os.makedirs(out_dir, exist_ok=True)
    for pred_arr, ref_path in zip(preds_unpadded, ref_paths):
        # Open reference (MODIS test file) to copy georeferencing
        ref_da = rxr.open_rasterio(ref_path)

        # Ensure 2D array (H,W)
        pred_2d = np.squeeze(np.asarray(pred_arr, dtype="float32"))
        if pred_2d.ndim != 2:
            raise ValueError(f"Expected 2D prediction, got shape: {pred_2d.shape}")

        # Wrap prediction as DataArray with one band
        out_da = xr.DataArray(pred_2d[np.newaxis, ...], dims=("band", "y", "x"))
        out_da = out_da.rio.write_crs(ref_da.rio.crs, inplace=True)
        out_da = out_da.rio.write_transform(ref_da.rio.transform(), inplace=True)

        out_name = f"predicted_{Path(ref_path).name}"
        out_path = Path(out_dir) / out_name
        # Write GeoTIFF
        out_da.rio.to_raster(out_path, dtype="float32", compress="deflate")

############################################################################
############################################################################
def masked_r2_numpy(y_true, y_pred, mask, eps=1e-12):
    y = y_true[mask > 0.5]
    p = y_pred[mask > 0.5]
    ss_res = np.sum((y - p)**2)
    ss_tot = np.sum((y - np.mean(y))**2) + eps
    return 1.0 - ss_res/ss_tot

def run_unet(
    unet_dimensions,
    load_weights_bool,
    load_model,
    goes_training_data_4d,
    modis_training_data_4d,
    modis_training_mask_3d,          # NEW
    combined_validation_data,        # (x_val, y_val)
    val_mask_3d,                     # NEW
    goes_test_data_final,
    modis_test_data_final,
    test_mask_3d,                    # NEW
    start_date_training_data_str,
    end_date_validation_data_str,
    start_date_test_data_str,
    end_date_test_data_str,
    run_mask = True,
    test_ref_paths_sorted: List[Path] = None,  # NEW
    dest_folder: str = None                    # NEW
):
    """
    Train, evaluate, and test a U-Net model for surface albedo prediction using GOES and MODIS data.

    This function supports training from scratch or loading pre-trained weights/models. It uses early stopping,
    learning rate reduction, and checkpointing for model optimization.

    Parameters
    ----------
    unet_dimensions : tuple
        Input shape of the data, e.g., (24, 24, 1).
    load_weights_bool : bool
        If True, loads model weights from checkpoint.
    load_model : bool
        If True, loads the entire saved model.
    goes_training_data_4d : np.ndarray
        4D training input data from GOES (samples, height, width, channels).
    modis_training_data_4d : np.ndarray
        4D training target data from MODIS (samples, height, width, channels).
    combined_validation_data : tuple
        Tuple of validation inputs and labels (x_val, y_val).
    goes_test_data_final : np.ndarray
        4D test input data from GOES.
    modis_test_data_final : np.ndarray
        4D test target data from MODIS.
    start_date_training_data_str : str
        Start date of training data (YYYY-MM-DD).
    end_date_validation_data_str : str
        End date of validation data (YYYY-MM-DD).
    start_date_test_data_str : str
        Start date of test data (YYYY-MM-DD).
    end_date_test_data_str : str
        End date of test data (YYYY-MM-DD).

    Returns
    -------
    None
        Saves the trained model, prediction outputs, and training history to disk.
    """
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1, min_delta=1e-8, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, min_delta=1e-7, cooldown=1)
    cp_callback = ModelCheckpoint(filepath=TENSORFLOW_CHECKPOINT_PATH, save_weights_only=True, verbose=1)

    if not load_weights_bool and not load_model:
        while True:
            model = get_UNet_model(unet_dimensions)
            history = model.fit(
                goes_training_data_4d,                 # x
                modis_training_data_4d,                # y
                sample_weight=modis_training_mask_3d,  # per-pixel weights (N,H,W)
                epochs=500,
                validation_data=(
                    combined_validation_data[0],
                    combined_validation_data[1],
                    val_mask_3d
                ),
                callbacks=[early_stopping, reduce_lr, cp_callback],
                verbose=2
            )
            model.save(TF_FELIX_MODEL_UNMASKED_PATH)

            loss, rmse = model.evaluate(
                goes_test_data_final,
                modis_test_data_final,
                sample_weight=test_mask_3d,
                verbose="auto"
            )
            print(f"Test Loss: {loss}")
            print(f"Test RMSE: {rmse}")

            if len(history.epoch) >= 25:
                break

        preds = tf.squeeze(model(goes_test_data_final), axis=-1).numpy()  # (N,24,24)
        preds_sans_padding = remove_padding(preds[:, :, :, np.newaxis])[:, :, :, 0]  # back to (N,21,19)
        
        if test_ref_paths_sorted is not None and dest_folder is not None:
            # preds_sans_padding is shaped (N, H, W)
            save_preds_as_geotiff(preds_sans_padding, test_ref_paths_sorted, dest_folder)
            print(f"Saved {len(test_ref_paths_sorted)} GeoTIFFs to: {dest_folder}")

        # Compute masked R-squared on native padded grid (more fair), then also on unpadded
        r2_masked_padded = masked_r2_numpy(
            modis_test_data_final[..., 0], preds, test_mask_3d
        )
        print(f"Masked R-squared (padded 24×24): {r2_masked_padded:.3f}")

        save_file_name = (
            UNET_RESULTS +
            f"Train-Start={start_date_training_data_str}-Train-End={end_date_validation_data_str}"
            f"-Test-Start={start_date_test_data_str}-Test-End={end_date_test_data_str}"
            + ("_masked_new.npy" if run_mask else "_not_masked_new.npy")
        )
        # print(f"Predicted array shape: {preds.shape}")
        # print(f"Final Array shape (unpadded): {preds_sans_padding.shape}")
        np.save(save_file_name, preds_sans_padding)

    else:
        if load_weights_bool:
            model = get_UNet_model(unet_dimensions)
            model.load_weights(TENSORFLOW_CHECKPOINT_PATH)
        else:
            model = tf.keras.models.load_model(TF_FELIX_MODEL_UNMASKED_PATH)

        history = model.fit(
            goes_training_data_4d,
            modis_training_data_4d,
            sample_weight=modis_training_mask_3d,
            epochs=500,
            validation_data=(combined_validation_data[0], combined_validation_data[1], val_mask_3d),
            callbacks=[early_stopping, reduce_lr, cp_callback],
            verbose=2
        )
        model.save(TF_FELIX_MODEL_UNMASKED_PATH)

        loss, rmse = model.evaluate(
            goes_test_data_final,
            modis_test_data_final,
            sample_weight=test_mask_3d,
            verbose="auto"
        )
        print(f"Test Loss: {loss}")
        print(f"Test RMSE: {rmse}")

        preds = tf.squeeze(model(goes_test_data_final), axis=-1).numpy()
        preds_sans_padding = remove_padding(preds[:, :, :, np.newaxis])[:, :, :, 0]
        
        if test_ref_paths_sorted is not None and dest_folder is not None:
            # preds_sans_padding is shaped (N, H, W)
            save_preds_as_geotiff(preds_sans_padding, test_ref_paths_sorted, dest_folder)
            print(f"Saved {len(test_ref_paths_sorted)} GeoTIFFs to: {dest_folder}")
        
        r2_masked_padded = masked_r2_numpy(
            modis_test_data_final[..., 0], preds, test_mask_3d
        )
        print(f"Masked R-squared (padded 24×24): {r2_masked_padded:.3f}")

        save_file_name = (
            UNET_RESULTS +
            f"Train-Start={start_date_training_data_str}-Train-End={end_date_validation_data_str}"
            f"-Test-Start={start_date_test_data_str}-Test-End={end_date_test_data_str}"
            + ("_masked_new.npy" if run_mask else "_not_masked_new.npy")
        )
        # print(f"Predicted array shape: {preds.shape}")
        # print(f"Final Array shape (unpadded): {preds_sans_padding.shape}")
        np.save(save_file_name, preds_sans_padding)

    with open(TF_HISTORY_PATH, 'w') as f:
        json.dump(history.history, f)

############################################################################
############################################################################
def convert_dates(json_path):
    """
    Load a JSON file containing datetime strings and convert them into Python datetime objects.

    Parameters
    ----------
    json_path : str or Path
        Path to the JSON file. The file should contain a list of ISO-like datetime strings,
        e.g., ["2023-07-16T18:30:00", "2023-07-17T19:30:00", ...]

    Returns
    -------
    List[datetime.datetime]
        A list of datetime objects converted from the input strings.

    Notes
    -----
    - If the datetime strings use a different format, update the `date_format` accordingly.
    """
    # Load the JSON data containing datetime strings
    with open(json_path, "r") as file:
        date_strings = json.load(file)

    # Define the format of the datetime strings in the file
    date_format = "%Y-%m-%dT%H:%M:%S"

    # Convert each string to a datetime object
    datetime_objects = [datetime.strptime(date_str, date_format) for date_str in date_strings]

    return datetime_objects

#######################################################
#######################################################


#######################################################
#######################################################


#######################################################
#######################################################


#######################################################
#######################################################


#######################################################
#######################################################


#######################################################
#######################################################