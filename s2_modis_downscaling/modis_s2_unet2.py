# Note some hard code things
import os
import re
import sys
import time
import json
import glob
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
from rasterio.enums import Resampling
from datetime import datetime, timedelta
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Input, Conv2D, Conv2DTranspose
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D, ReLU, LeakyReLU, concatenate

# Directories
UNET_RESULTS = "/bsuhome/tnde/scratch/felix/UNet2/Results_new/"
TENSORFLOW_CHECKPOINT_PATH = "/bsuhome/tnde/scratch/felix/UNet2/Results_new/training/old_cp.weights.h5"
TENSORFLOW_TRAINING_DIR = "/bsuhome/tnde/scratch/felix/UNet2/Results_new/training/"
TF_HISTORY_PATH = "/bsuhome/tnde/scratch/felix/UNet2/Results_new/training/history_new_old.json"
TF_FELIX_MODEL_UNMASKED_PATH = "/bsuhome/tnde/scratch/felix/UNet2/Results_new/training/unet2_old.keras"

# keep seeds fixed
os.environ["PYTHONHASHSEED"]="0"
random.seed(0); np.random.seed(0); tf.random.set_seed(0)

kernel = 3

# Sentinel-2 cloud fractions
cf_file = "/bsuhome/tnde/scratch/felix/Sentinel-2/s2_albedo_outputs/tsi_cloud_fractions.csv"
cf_vals = pd.read_csv(cf_file)
cf_vals = cf_vals[cf_vals["cf_interp"]<=0.4]
cf_vals = cf_vals.drop_duplicates(subset=["date"])
# display(cf_vals.head())

#-------------------- Matching Sentinel-2 and MODIS files --------------------#
# Matching train files
modis_train_path = "/bsuhome/tnde/scratch/felix/UNet/Unet_train_preds_modis_new/predicted_*"
modis_train_files = os.path.abspath(modis_train_path)
modis_train_files_sorted = sorted(glob.glob(modis_train_files))
modis_unet_train_files_list = []

s2_train_path = "/bsuhome/tnde/scratch/felix/Sentinel-2/s2_albedo_outputs/*_S2_BLUE20m_SW_hard.tif"
s2_train_files = os.path.abspath(s2_train_path)
s2_train_files_sorted = sorted(glob.glob(s2_train_files))
s2_unet_train_files_list = []
# invalid_train_dates = ["2021-09-23", "2022-03-12", "2022-03-12", 
#                        "2022-04-06", "2022-05-06", "2022-05-26", 
#                        "2022-09-18", "2022-10-13", "2022-10-18"]

# invalid_train_dates = ["2022-07-10", "2022-05-06", "2022-04-21", 
#                       "2022-04-06", "2022-04-01", "2022-03-12", 
#                       "2021-09"]

invalid_train_dates = ["2022-09-23", "2021-10-28", "2022-03-12", "2022-04-01", 
                       "2022-04-06", "2022-04-21", "2022-04-26", "2022-05-06", 
                       "2022-05-26", "2022-05-31", "2022-07-10"]

for s2_date in list(cf_vals["date"]):
    modis_unet_train_file = f"/bsuhome/tnde/scratch/felix/UNet/Unet_train_preds_modis_new/predicted_{s2_date}_modis_blue_sky_albedo_.tif"
    s2_unet_train_file = f"/bsuhome/tnde/scratch/felix/Sentinel-2/s2_albedo_outputs/{s2_date}_S2_BLUE20m_SW_hard.tif"
    if s2_date in invalid_train_dates:
        pass
    elif modis_unet_train_file not in modis_train_files_sorted:
        # print(f"File: {modis_unet_train_file} not matched.")
        pass
    else:
        modis_unet_train_files_list.append(modis_unet_train_file)
        s2_unet_train_files_list.append(s2_unet_train_file)
print(f"Number of MODIS U-Net train files: {len(modis_unet_train_files_list)}")
print(f"Number of Sentinel-2 U-Net train files: {len(s2_unet_train_files_list)}")

# Matching test files
modis_test_path = "/bsuhome/tnde/scratch/felix/UNet/Unet_test_preds_modis_new/predicted_*"
modis_test_files = os.path.abspath(modis_test_path)
modis_test_files_sorted = sorted(glob.glob(modis_test_files))
modis_unet_test_files_list = []
# invalid_test_dates = ["2021-09-23", "2022-03-12", "2022-03-12", 
#                       "2022-04-06", "2022-05-06", "2022-05-26", 
#                       "2022-09-18", "2022-10-13", "2022-10-18"]

# invalid_test_dates = ["2022-07-10", "2022-05-06", "2022-04-21", 
#                       "2022-04-06", "2022-04-01", "2022-03-12", 
#                       "2021-09-23"]

invalid_test_dates = ["2022-09-23", "2021-10-28", "2022-03-12", "2022-04-01", 
                      "2022-04-06", "2022-04-21", "2022-04-26", "2022-05-06", 
                      "2022-05-26", "2022-05-31", "2022-07-10"]


s2_test_path = "/bsuhome/tnde/scratch/felix/Sentinel-2/s2_albedo_outputs/*_S2_BLUE20m_SW_hard.tif"
s2_test_files = os.path.abspath(s2_test_path)
s2_test_files_sorted = sorted(glob.glob(s2_test_files))
s2_unet_test_files_list = []

for s2_date in list(cf_vals["date"]):
    modis_unet_test_file = f"/bsuhome/tnde/scratch/felix/UNet/Unet_test_preds_modis_new/predicted_{s2_date}_modis_blue_sky_albedo_.tif"
    s2_unet_test_file = f"/bsuhome/tnde/scratch/felix/Sentinel-2/s2_albedo_outputs/{s2_date}_S2_BLUE20m_SW_hard.tif"
    if s2_date in invalid_test_dates:
        pass
    if modis_unet_test_file not in modis_test_files_sorted:
        # print(f"File: {modis_unet_test_file} not matched.")
        pass
    else:
        modis_unet_test_files_list.append(modis_unet_test_file)
        s2_unet_test_files_list.append(s2_unet_test_file)
print(f"Number of MODIS U-Net test files: {len(modis_unet_test_files_list)}")
print(f"Number of Sentinel-2 U-Net test files: {len(s2_unet_test_files_list)}")

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

# def pad_mask_2d(mask2d: np.ndarray) -> np.ndarray:
#     """Pad a boolean/float mask the same way as the data."""
#     return np.pad(
#         mask2d, pad_width=((1, 2), (2, 3)),
#         mode="constant", constant_values=0.0
#     )

############################################################################
############################################################################
def pad_batch_to_multiple(X: np.ndarray, Y: np.ndarray, M: np.ndarray, mult: int = 8):
    """
    X,Y: (N,H,W,1) or (N,H,W) or (N,H,W,C==1)
    M:   (N,H,W,1) or (N,H,W)
    Pads all three with zeros so H and W become multiples of `mult`.
    Returns (Xp, Yp, Mp, pad_info_dict). 
            Padded arrays and a dict with pad sizes.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    M = np.asarray(M)

    # Coerce to 4D with singleton channel
    if X.ndim == 3: X = X[..., np.newaxis]
    if Y.ndim == 3: Y = Y[..., np.newaxis]
    if M.ndim == 3: M = M[..., np.newaxis]
    if X.shape[-1] != 1: X = X[..., :1]
    if Y.shape[-1] != 1: Y = Y[..., :1]
    if M.shape[-1] != 1: M = M[..., :1]

    # Shape sanity
    if not (X.shape[0] == Y.shape[0] == M.shape[0]):
        raise ValueError(f"Batch mismatch: X{X.shape}, Y{Y.shape}, M{M.shape}")
    if not (X.shape[1] == Y.shape[1] == M.shape[1] and X.shape[2] == Y.shape[2] == M.shape[2]):
        raise ValueError(f"H/W mismatch: X{X.shape}, Y{Y.shape}, M{M.shape}")

    _, H, W, _ = X.shape

    def pad_to_multiple_hw(h: int, w: int, mult: int = 8):
        """Compute symmetric padding to reach next multiple of `mult`."""
        h_pad = (mult - (h % mult)) % mult
        w_pad = (mult - (w % mult)) % mult
        top  = h_pad // 2
        bot  = h_pad - top
        left = w_pad // 2
        right= w_pad - left
        return top, bot, left, right

    top, bot, left, right = pad_to_multiple_hw(H, W, mult=mult)
    if top==bot==left==right==0:
        return X, Y, M, {"top":0, "bot":0, "left":0, "right":0}

    pad_spec = ((0,0), (top,bot), (left,right), (0,0))
    Xp = np.pad(X, pad_spec, mode="constant", constant_values=0.0)
    Yp = np.pad(Y, pad_spec, mode="constant", constant_values=0.0)
    # keep padded mask = 0
    Mp = np.pad(M, pad_spec, mode="constant", constant_values=0.0)  # mask stays 0 on padded rim 
    return Xp, Yp, Mp, {"top":top, "bot":bot, "left":left, "right":right}


def crop_pred(pred: np.ndarray, pad_info: dict):
    """pred: (N,H,W,1). Remove the padding added by pad_batch_to_multiple."""
    t, b, l, r = pad_info["top"], pad_info["bot"], pad_info["left"], pad_info["right"]
    if t==b==l==r==0:
        return pred
    return pred[:, t:pred.shape[1]-b, l:pred.shape[2]-r, :]

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
    
    # # First convolution + LeakyReLU
    # conv = Conv2D(n_filters, kernel, padding='same')(inputs)
    # conv = ReLU()(conv)

    # # Second convolution + LeakyReLU
    # conv = Conv2D(n_filters, kernel, padding='same')(conv)
    # conv = ReLU()(conv)

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
    
    # # Apply two convolutional layers to refine features
    # conv = Conv2D(n_filters, kernel, padding='same')(merge)
    # conv = ReLU()(conv)

    # conv = Conv2D(n_filters, kernel, padding='same')(conv)
    # conv = ReLU()(conv)
    return conv

############################################################################
############################################################################
def get_UNet_model(input_size=(None, None, 1)):
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
    inputs = Input((None, None, 1))  # accept any H×W

    # Encoder path
    cblock1 = EncoderMiniBlock(inputs, n_filters,   max_pooling=True)
    cblock2 = EncoderMiniBlock(cblock1[0], n_filters*2, max_pooling=True)
    cblock3 = EncoderMiniBlock(cblock2[0], n_filters*4, max_pooling=True)
    cblock4 = EncoderMiniBlock(cblock3[0], n_filters*8, max_pooling=False)

    # Decoder path with skip connections
    ublock1 = DecoderMiniBlock(cblock4[0], cblock3[1], n_filters*4)
    ublock2 = DecoderMiniBlock(ublock1, cblock2[1], n_filters*2)
    ublock3 = DecoderMiniBlock(ublock2, cblock1[1], n_filters)

    # Final convolutional layers to reduce channels to output
    conv8  = Conv2D(n_filters, 3, activation='relu', padding='same')(ublock3)
    conv9  = Conv2D(n_classes, 1, padding='same')(conv8)
    conv10 = Conv2D(n_classes, 1, padding='same')(conv9)
    
    # # Final convolutional layers to reduce channels to output
    # conv8 = Conv2D(n_filters, 3, padding='same')(ublock3)
    # conv8 = ReLU()(conv8)
    # conv9  = Conv2D(n_classes, 1, padding='same')(conv8)   # linear
    # conv10 = Conv2D(n_classes, 1, padding='same')(conv9)   # linear

    # Define and compile the model
    model = Model(inputs=inputs, outputs=conv9)

    opt = Adam(learning_rate=1e-4) #Titus
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
    Save predictions with the S2 file's CRS/transform.
    Output name: predicted_<original S2 filename>.tif
    """
    os.makedirs(out_dir, exist_ok=True)
    for pred_arr, ref_path in zip(preds_unpadded, ref_paths):
        ref_da = rxr.open_rasterio(ref_path)
        pred_2d = np.asarray(pred_arr, dtype="float32")

        out_da = xr.DataArray(pred_2d[np.newaxis, ...], dims=("band", "y", "x"))
        out_da = out_da.rio.write_crs(ref_da.rio.crs, inplace=True)
        out_da = out_da.rio.write_transform(ref_da.rio.transform(), inplace=True)

        out_name = f"predicted_s2_{Path(ref_path).name}"
        out_da.rio.to_raster(Path(out_dir)/out_name, dtype="float32", compress="deflate")

############################################################################
############################################################################
# ----- masked metrics (computed manually) -----
def masked_rmse(y_true, y_pred, mask, eps=1e-12):
    yt = y_true[..., 0] if y_true.ndim == 4 and y_true.shape[-1] == 1 else y_true
    yp = y_pred[..., 0] if y_pred.ndim == 4 and y_pred.shape[-1] == 1 else y_pred
    m  = mask[..., 0]  if mask.ndim  == 4 and mask.shape[-1]  == 1 else mask
    v = m > 0.5
    if v.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((yt[v] - yp[v])**2)))

def masked_r2(y_true, y_pred, mask, eps=1e-12):
    yt = y_true[..., 0] if y_true.ndim == 4 and y_true.shape[-1] == 1 else y_true
    yp = y_pred[..., 0] if y_pred.ndim == 4 and y_pred.shape[-1] == 1 else y_pred
    m  = mask[..., 0]  if mask.ndim  == 4 and mask.shape[-1]  == 1 else mask
    v = m > 0.5
    if v.sum() == 0:
        return np.nan
    ss_res = np.sum((yt[v] - yp[v])**2)
    ss_tot = np.sum((yt[v] - yt[v].mean())**2) + eps
    return float(1.0 - ss_res/ss_tot)

def run_unet(
    unet_dimensions,
    load_weights_bool,
    load_model,
    modis_training_data_4d,
    s2_training_data_4d,
    s2_training_mask_3d,             # NEW
    combined_validation_data,        # (x_val, y_val)
    val_mask_3d,                     # NEW
    modis_test_data_final,
    s2_test_data_final,
    test_mask_3d,                    # NEW
    start_date_training_data_str,
    end_date_validation_data_str,
    start_date_test_data_str,
    end_date_test_data_str,
    run_mask = True,
    test_ref_paths_sorted: List[Path] = None,  # NEW
    dest_folder: str = None,                   # NEW
    pad_info_test: dict | None = None
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
    modis_training_data_4d : np.ndarray
        4D training input data from GOES (samples, height, width, channels).
    s2_training_data_4d : np.ndarray
        4D training target data from MODIS (samples, height, width, channels).
    combined_validation_data : tuple
        Tuple of validation inputs and labels (x_val, y_val).
    modis_test_data_final : np.ndarray
        4D test input data from GOES.
    s2_test_data_final : np.ndarray
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
                modis_training_data_4d,             # x
                s2_training_data_4d,                # y
                sample_weight=s2_training_mask_3d[..., 0],  # per-pixel weights (N,H,W)
                epochs=100,
                validation_data=(
                    combined_validation_data[0],
                    combined_validation_data[1],
                    val_mask_3d
                ),
                callbacks=[early_stopping, reduce_lr, cp_callback],
                verbose=2
            )
            model.save(TF_FELIX_MODEL_UNMASKED_PATH)

            # ensure concrete numpy float32 (avoids unknown-rank issues)
            x_eval = np.asarray(modis_test_data_final, dtype="float32")
            y_eval = np.asarray(s2_test_data_final, dtype="float32")
            w_eval = np.asarray(test_mask_3d, dtype="float32")

            # run prediction
            preds = model.predict(x_eval, verbose=1)              # (N,H_pad,W_pad,1)

            # crop back if we padded
            if pad_info_test is not None:
                preds_cropped = crop_pred(preds, pad_info_test)   # (N,H,W,1)
                y_eval_c = crop_pred(y_eval, pad_info_test)
                w_eval_c = crop_pred(w_eval, pad_info_test)
            else:
                preds_cropped = preds
                y_eval_c = y_eval
                w_eval_c = w_eval

            rmse_masked  = masked_rmse(y_eval, preds, w_eval)            # padded-safe
            r2_masked    = masked_r2 (y_eval, preds, w_eval)
            rmse_cropped = masked_rmse(y_eval_c, preds_cropped, w_eval_c) # original size
            r2_cropped   = masked_r2 (y_eval_c, preds_cropped, w_eval_c)

            print(f"Test RMSE (masked, padded): {rmse_masked:.4f}")
            print(f"Test R-squared (masked, padded): {r2_masked:.4f}")
            print(f"Test RMSE (masked, cropped): {rmse_cropped:.4f}")
            print(f"Test R-squared (masked, cropped): {r2_cropped:.4f}")

            if len(history.epoch) >= 25:
                break

        # --- save GeoTIFFs using S2 references (save uncropped size that matches refs)
        # If your refs correspond to the ORIGINAL (uncropped) rasters, save CROPPED arrays:
        save_preds_as_geotiff(preds_cropped[..., 0], test_ref_paths_sorted, dest_folder)
        print(f"Saved {len(test_ref_paths_sorted)} GeoTIFFs to: {dest_folder}")

        # .npy artifact, save CROPPED predictions (match S2 size)
        save_file_name = (
            UNET_RESULTS +
            f"Train-Start={start_date_training_data_str}-Train-End={end_date_validation_data_str}"
            f"-Test-Start={start_date_test_data_str}-Test-End={end_date_test_data_str}"
            + ("_masked_new.npy" if run_mask else "_not_masked_new.npy")
        )
        preds_cropped_2d = preds_cropped[..., 0]  # (N,H,W)
        print(f"Predicted array shape (padded): {preds.shape}")
        print(f"Predicted array shape (cropped): {preds_cropped_2d.shape}")
        np.save(save_file_name, preds_cropped_2d)
    else:
        if load_weights_bool:
            model = get_UNet_model(unet_dimensions)
            model.load_weights(TENSORFLOW_CHECKPOINT_PATH)
        else:
            model = tf.keras.models.load_model(TF_FELIX_MODEL_UNMASKED_PATH)

        history = model.fit(
            modis_training_data_4d,
            s2_training_data_4d,
            sample_weight=s2_training_mask_3d[..., 0],
            epochs=100,
            validation_data=(combined_validation_data[0], combined_validation_data[1], val_mask_3d),
            callbacks=[early_stopping, reduce_lr, cp_callback],
            verbose=2
        )
        model.save(TF_FELIX_MODEL_UNMASKED_PATH)
        
        # ensure concrete numpy float32 (avoids unknown-rank issues)
        x_eval = np.asarray(modis_test_data_final, dtype="float32")
        y_eval = np.asarray(s2_test_data_final,  dtype="float32")
        w_eval = np.asarray(test_mask_3d,        dtype="float32")

        # run prediction
        preds = model.predict(x_eval, verbose=1)              # (N,H_pad,W_pad,1)

        # crop back if we padded
        if pad_info_test is not None:
            preds_cropped = crop_pred(preds, pad_info_test)   # (N,H,W,1)
            y_eval_c = crop_pred(y_eval, pad_info_test)
            w_eval_c = crop_pred(w_eval, pad_info_test)
        else:
            preds_cropped = preds
            y_eval_c = y_eval
            w_eval_c = w_eval

        rmse_masked  = masked_rmse(y_eval, preds, w_eval)            # padded-safe
        r2_masked    = masked_r2 (y_eval, preds, w_eval)
        rmse_cropped = masked_rmse(y_eval_c, preds_cropped, w_eval_c) # original size
        r2_cropped   = masked_r2 (y_eval_c, preds_cropped, w_eval_c)

        print(f"Test RMSE (masked, padded): {rmse_masked:.4f}")
        print(f"Test R-squared (masked, padded): {r2_masked:.4f}")
        print(f"Test RMSE (masked, cropped): {rmse_cropped:.4f}")
        print(f"Test R-squared (masked, cropped): {r2_cropped:.4f}")

        # Save as GeoTIFFs using S2 refs
        save_preds_as_geotiff(preds_cropped[..., 0], test_ref_paths_sorted, dest_folder)
        print(f"Saved {len(test_ref_paths_sorted)} GeoTIFFs to: {dest_folder}")

        # .npy artifact, save CROPPED predictions (match S2 size)
        save_file_name = (
            UNET_RESULTS +
            f"Train-Start={start_date_training_data_str}-Train-End={end_date_validation_data_str}"
            f"-Test-Start={start_date_test_data_str}-Test-End={end_date_test_data_str}"
            + ("_masked_new.npy" if run_mask else "_not_masked_new.npy")
        )
        preds_cropped_2d = preds_cropped[..., 0]  # (N,H,W)
        print(f"Predicted array shape (padded): {preds.shape}")
        print(f"Predicted array shape (cropped): {preds_cropped_2d.shape}")
        np.save(save_file_name, preds_cropped_2d)

    with open(TF_HISTORY_PATH, 'w') as f:
        json.dump(history.history, f)

###############################
# --- NEW: reproject MODIS onto the exact S2 grid (CRS, transform, shape) ---
def reproject_to_match(src_da: xr.DataArray, ref_da: xr.DataArray, resampling: Resampling = Resampling.bilinear) -> xr.DataArray:
    if not src_da.rio.crs or not ref_da.rio.crs:
        raise ValueError("Source or reference raster has no CRS")
    return src_da.rio.reproject_match(ref_da, resampling=resampling)

# --- NEW: build stacks directly from aligned lists (same length/order) ---
def load_stacks_from_lists(
    modis_paths: List[str | Path],
    s2_paths: List[str | Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, List[Path]]:
    """
    Returns:
      X_4d : (N, H, W, 1)  # MODIS resampled to S2 grid, padded to 24x24
      Y_4d : (N, H, W, 1)  # S2 target on its native grid, padded to 24x24
      M_3d : (N, H, W)     # target validity mask (1 valid, 0 invalid), padded with zeros
      s2_refs : list[Path] # S2 file paths (for georeferencing when saving predictions)
    Assumes each raster is single-band albedo (use band=1 if multi-band).
    """
    assert len(modis_paths) == len(s2_paths), "MODIS/S2 lists must have equal length"
    X_list, Y_list, M_list, s2_refs = [], [], [], []

    for m_path, s2_path in zip(modis_paths, s2_paths):
        m_path, s2_path = Path(m_path), Path(s2_path)

        # load
        m_da  = load_raster_da(m_path)
        s2_da = load_raster_da(s2_path)

        m2d  = m_da.sel(band=1, drop=True) if "band" in m_da.dims else m_da
        s22d = s2_da.sel(band=1, drop=True) if "band" in s2_da.dims else s2_da

        # reproject MODIS -> S2
        m_on_s2 = reproject_to_match(m2d, s22d, resampling=Resampling.bilinear)

        # target mask from S2 validity
        y_arr = s22d.values.astype("float32")
        msk   = np.isfinite(y_arr).astype("float32")

        # fill NaNs for compute, then pad to 24×24
        y_filled = np.nan_to_num(y_arr, nan=0.0)
        y_pad    = pad_da_2d(xr.DataArray(y_filled, dims=("y","x")))
        # IMPORTANT: padding in mask must be zeros so it never contributes
        m_pad    = np.pad(msk, pad_width=((1, 2), (2, 3)), mode="constant", constant_values=0.0)

        # inputs: MODIS on S2 grid → fill & pad
        x_arr = m_on_s2.values.astype("float32")
        x_arr = np.nan_to_num(x_arr, nan=0.0)
        x_pad = pad_da_2d(xr.DataArray(x_arr, dims=("y","x")))

        X_list.append(x_pad[..., np.newaxis])  # (H,W,1)
        Y_list.append(y_pad[..., np.newaxis])  # (H,W,1)
        M_list.append(m_pad)                   # (H,W)
        s2_refs.append(s2_path)

    X_4d = np.stack(X_list, axis=0)
    Y_4d = np.stack(Y_list, axis=0)
    M_3d = np.stack(M_list, axis=0)
    return X_4d, Y_4d, M_3d, s2_refs

#######################################################
#######################################################
# def main(valid_test_dates = False):
#     dest_folder = "/bsuhome/tnde/scratch/felix/UNet2/Unet_test_preds_s2_new/"
#     os.makedirs(dest_folder, exist_ok=True)
#     start = time.time()
    
#     # local aliases
#     all_modis_train = modis_unet_train_files_list
#     all_s2_train    = s2_unet_train_files_list
    
#     # ----------------- matched lists -----------------
#     if valid_test_dates:
#         modis_test = all_modis_train
#         s2_test = all_s2_train
#     else:
#         test_size = int(0.25 * len(all_modis_train))
#         modis_test = all_modis_train[-test_size:]
#         s2_test = all_s2_train[-test_size:]
        
#         all_modis_train = all_modis_train[:-test_size]
#         all_s2_train = all_s2_train[:-test_size]
    
#     n = len(all_modis_train)
#     # n_train = int(0.85 * n)
#     n_train = int(1.0 * n)

#     # modis_train = all_modis_train[:n_train]
#     # s2_train = all_s2_train[:n_train]
#     # modis_val = all_modis_train[n_train:]
#     # s2_val = all_s2_train[n_train:]
    
#     modis_train = all_modis_train[:n_train]
#     s2_train = all_s2_train[:n_train]
#     modis_val = all_modis_train[:n_train]
#     s2_val = all_s2_train[:n_train]

#     # Build stacks (MODIS reprojected to S2 grid; both padded to 24x24)
#     X_train, Y_train, M_train, s2_train_refs = load_stacks_from_lists(modis_train, s2_train)
#     X_val, Y_val, M_val, s2_val_refs = load_stacks_from_lists(modis_val,   s2_val)
#     X_test, Y_test, M_test, s2_test_refs = load_stacks_from_lists(modis_test,  s2_test)
    
#     print("Train shapes:", X_train.shape, Y_train.shape, M_train.shape)
#     print("Val shapes:", X_val.shape, Y_val.shape, M_val.shape)
#     print("Test shapes:", X_test.shape, Y_test.shape, M_test.shape)

#     # Apply padding
#     X_train, Y_train, M_train, pad_train = pad_batch_to_multiple(X_train, Y_train, M_train, mult=8)
#     X_val, Y_val, M_val, pad_val = pad_batch_to_multiple(X_val, Y_val, M_val, mult=8)
#     X_test, Y_test, M_test, pad_test = pad_batch_to_multiple(X_test, Y_test, M_test, mult=8)

#     # Sanity checks
#     for X in (X_train, X_val, X_test):
#         assert np.isfinite(X).all(), "Inputs contain NaN/Inf"

#     unet_dimensions = (24, 24, 1)
#     os.makedirs(UNET_RESULTS, exist_ok=True)
#     os.makedirs(TENSORFLOW_TRAINING_DIR, exist_ok=True)

#     # ----------------- train/eval/save -----------------
#     run_unet(
#         unet_dimensions=(None, None, 1),
#         load_weights_bool=False,
#         load_model=False,
#         modis_training_data_4d=X_train,                  # inputs = MODIS on S2 grid
#         s2_training_data_4d=Y_train,                     # targets = S2
#         s2_training_mask_3d=M_train,                     # per-pixel weights
#         combined_validation_data=(X_val, Y_val),
#         val_mask_3d=M_val,
#         modis_test_data_final=X_test,
#         s2_test_data_final=Y_test,
#         test_mask_3d=M_test,
#         start_date_training_data_str="train",
#         end_date_validation_data_str="val",
#         start_date_test_data_str="test",
#         end_date_test_data_str="end",
#         run_mask=True,
#         test_ref_paths_sorted=s2_test_refs, # use S2 geometries for saving
#         dest_folder=dest_folder,
#         pad_info_test=pad_test,   # NEW padding test
#     )
#     print(f"Time for U-Net to run: {time.time() - start:.2f} seconds")

# --- FIX main() scoping and small edge cases ---
def main(valid_test_dates=False):
    dest_folder = "/bsuhome/tnde/scratch/felix/UNet2/Unet_test_preds_s2_new/"
    os.makedirs(dest_folder, exist_ok=True)
    start = time.time()

    # local aliases
    all_modis_train = modis_unet_train_files_list
    all_s2_train    = s2_unet_train_files_list

    if valid_test_dates:
        modis_test = modis_unet_test_files_list
        s2_test    = s2_unet_test_files_list
        modis_train = all_modis_train
        s2_train    = all_s2_train
    else:
        n_total   = len(all_modis_train)
        test_size = max(1, int(0.3 * n_total))   # avoid 0 for tiny datasets
        modis_test = all_modis_train[-test_size:]
        s2_test    = all_s2_train[-test_size:]
        modis_train2 = all_modis_train[:-test_size]
        s2_train2    = all_s2_train[:-test_size]
    
    n = len(s2_train2)
    
    # n_train = int(0.85 * n)
    # modis_val = modis_unet_train_files_list[n_train:]
    # s2_val = s2_unet_train_files_list[n_train:]
    
    n_train = int(1 * n)
    modis_val = modis_train2[:n_train]
    s2_val = s2_train2[:n_train]
    modis_train = modis_train2[:n_train]
    s2_train    = s2_train2[:n_train]
    
    # print(modis_val)
    
    # Build stacks (MODIS reprojected to S2 grid; both padded to 24x24)
    X_train, Y_train, M_train, s2_train_refs = load_stacks_from_lists(modis_train, s2_train)
    X_val, Y_val, M_val, s2_val_refs = load_stacks_from_lists(modis_val,   s2_val)
    X_test, Y_test, M_test, s2_test_refs = load_stacks_from_lists(modis_test,  s2_test)

    print("Train shapes:", X_train.shape, Y_train.shape, M_train.shape)
    print("Val shapes:", X_val.shape, Y_val.shape, M_val.shape)
    print("Test shapes:", X_test.shape, Y_test.shape, M_test.shape)

    # # Apply padding. Pad to multiples-of-8 for the model
    X_train, Y_train, M_train, pad_train = pad_batch_to_multiple(X_train, Y_train, M_train, mult=8)
    X_val, Y_val, M_val, pad_val = pad_batch_to_multiple(X_val, Y_val, M_val, mult=8)
    X_test, Y_test, M_test, pad_test = pad_batch_to_multiple(X_test, Y_test, M_test, mult=8)

    # Sanity checks
    for X in (X_train, X_val, X_test):
        if not np.isfinite(X).all():
            raise ValueError("Inputs contain NaN/Inf")

    # ----------------- train/eval/save -----------------
    run_unet(
        unet_dimensions=(None, None, 1),
        load_weights_bool=False,
        load_model=False,
        modis_training_data_4d=X_train,
        s2_training_data_4d=Y_train,
        s2_training_mask_3d=M_train,
        combined_validation_data=(X_val, Y_val),
        val_mask_3d=M_val,
        modis_test_data_final=X_test,
        s2_test_data_final=Y_test,
        test_mask_3d=M_test,
        start_date_training_data_str="train",
        end_date_validation_data_str="val",
        start_date_test_data_str="test",
        end_date_test_data_str="end",
        run_mask=True,
        test_ref_paths_sorted=s2_test_refs,
        dest_folder=dest_folder,
        pad_info_test=pad_test,
    )
    print(f"Time for U-Net to run: {time.time() - start:.2f} seconds")

    
if __name__ == "__main__":
    main(valid_test_dates = False)
