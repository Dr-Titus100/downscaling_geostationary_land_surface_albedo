# Import packages
from packages import *

# Directories
S2_BLUE_SKY_ALBEDO_DIR = "/bsuhome/tnde/scratch/felix/Sentinel-2/s2_albedo_outputs/"
GOES_ALBEDO_DIR = "/bsuhome/tnde/scratch/felix/GOES/data/goes_output_data_new/"
MASKED_GOES_ALBEDO_DIR = "/bsuhome/tnde/scratch/felix/GOES/data/nan_data_new/"

# Data
INVALID_GOES_SOLAR_NOON_DATES = [datetime(2022, 1, 5), datetime(2022, 1, 25), datetime(2022, 2, 8), datetime(2022, 2, 13), 
                                 datetime(2022, 2, 21), datetime(2022, 12, 14), datetime(2022, 12, 23)]
INVALID_GOES_AQUA_DATES = [datetime(2022, 1, 2), datetime(2022, 1, 3), datetime(2022, 1, 7), datetime(2022, 12, 17), datetime(2023, 1, 5), datetime(2023, 1, 31)]
INVALID_GOES_DATES_BOTH = [datetime(2022, 12, 16), datetime(2022, 12, 18), datetime(2022, 12, 19), datetime(2022, 4, 8), datetime(2023, 1, 7), datetime(2023, 1, 8), datetime(2023, 5, 11), datetime(2023, 3, 14)]

# Directories
UNET_RESULTS = "/bsuhome/tnde/scratch/felix/unet_goes_s2/"
TENSORFLOW_CHECKPOINT_PATH = "/bsuhome/tnde/scratch/felix/unet_goes_s2/training/old_cp.weights.h5"
TENSORFLOW_TRAINING_DIR = "/bsuhome/tnde/scratch/felix/unet_goes_s2/Results/training/"
TF_HISTORY_PATH = "/bsuhome/tnde/scratch/felix/unet_goes_s2/Results/training/history.json"
TF_FELIX_MODEL_UNMASKED_PATH = "/bsuhome/tnde/scratch/felix/unet_goes_s2/Results/training/goes_s2_unet.keras"

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

#-------------------- Matching Sentinel-2 and GOES files --------------------#
# NOTE: adjust these GOES paths/patterns to match your actual filenames
goes_train_path = "/bsuhome/tnde/scratch/felix/GOES/data/500m-raster/*.tif"
goes_train_files = os.path.abspath(goes_train_path)
goes_train_files_sorted = sorted(glob.glob(goes_train_files))
goes_unet_train_files_list = []

s2_train_path = "/bsuhome/tnde/scratch/felix/Sentinel-2/s2_albedo_outputs/*_S2_BLUE20m_SW_hard.tif"
s2_train_files = os.path.abspath(s2_train_path)
s2_train_files_sorted = sorted(glob.glob(s2_train_files))
s2_unet_train_files_list = []

invalid_train_dates = ["2022-09-23", "2021-10-28", "2022-03-12", "2022-04-01",
                       "2022-04-06", "2022-04-21", "2022-04-26", "2022-05-06",
                       "2022-05-26", "2022-05-31", "2022-07-10"]

# # NEW TITUS
# ############################################################################
# ############################################################################
# def convert_dates(json_path):
#     """
#     Load a JSON file containing datetime strings and convert them into Python datetime objects.

#     Parameters
#     ----------
#     json_path : str or Path
#         Path to the JSON file. The file should contain a list of ISO-like datetime strings,
#         e.g., ["2023-07-16T18:30:00", "2023-07-17T19:30:00", ...]

#     Returns
#     -------
#     List[datetime.datetime]
#         A list of datetime objects converted from the input strings.

#     Notes
#     -----
#     - If the datetime strings use a different format, update the `date_format` accordingly.
#     """
#     # Load the JSON data containing datetime strings
#     with open(json_path, "r") as file:
#         date_strings = json.load(file)

#     # Define the format of the datetime strings in the file
#     date_format = "%Y-%m-%dT%H:%M:%S"

#     # Convert each string to a datetime object
#     datetime_objects = [datetime.strptime(date_str, date_format) for date_str in date_strings]

#     return datetime_objects
# # END NEW TITUS

# # NEW TITUS
invalid_dates = INVALID_GOES_DATES_BOTH

# # Load invalid dates and append hard-coded invalid GOES dates
# invalid_dates = convert_dates(INVALID_DATES_PATH)
# for date in INVALID_GOES_DATES_BOTH:
#     if date not in invalid_dates:
#         invalid_dates.append(date)
# # END NEW TITUS       
        
for s2_date in list(cf_vals["date"]):
    # IMPORTANT: update the filename template to your GOES product naming
    goes_unet_train_file = (
        f"/bsuhome/tnde/scratch/felix/GOES/data/500m-raster/"
        f"{s2_date}-GOES-500m.tif"
    )
    s2_unet_train_file = f"/bsuhome/tnde/scratch/felix/Sentinel-2/s2_albedo_outputs/{s2_date}_S2_BLUE20m_SW_hard.tif"

    if s2_date in invalid_train_dates:
        continue
    if goes_unet_train_file not in goes_train_files_sorted:
        continue

    goes_unet_train_files_list.append(goes_unet_train_file)
    s2_unet_train_files_list.append(s2_unet_train_file)
goes_unet_train_files_list = goes_unet_train_files_list[:-12]
s2_unet_train_files_list = s2_unet_train_files_list[:-12]
print(f"Number of GOES U-Net train files: {len(goes_unet_train_files_list)}")
print(f"Number of Sentinel-2 U-Net train files: {len(s2_unet_train_files_list)}")
print(goes_unet_train_files_list)
print(s2_unet_train_files_list)

# ----------------- Matching test files -----------------
goes_test_path = "/bsuhome/tnde/scratch/felix/GOES/data/500m-raster/*.tif"
goes_test_files = os.path.abspath(goes_test_path)
goes_test_files_sorted = sorted(glob.glob(goes_test_files))
goes_unet_test_files_list = []

invalid_test_dates = ["2022-09-23", "2021-10-28", "2022-03-12", "2022-04-01",
                      "2022-04-06", "2022-04-21", "2022-04-26", "2022-05-06",
                      "2022-05-26", "2022-05-31", "2022-07-10"]

s2_test_path = "/bsuhome/tnde/scratch/felix/Sentinel-2/s2_albedo_outputs/*_S2_BLUE20m_SW_hard.tif"
s2_test_files = os.path.abspath(s2_test_path)
s2_test_files_sorted = sorted(glob.glob(s2_test_files))
s2_unet_test_files_list = []

for s2_date in list(cf_vals["date"]):
    goes_unet_test_file = (
        f"/bsuhome/tnde/scratch/felix/GOES/data/500m-raster/"
        f"{s2_date}-GOES-500m.tif"
    )
    s2_unet_test_file = f"/bsuhome/tnde/scratch/felix/Sentinel-2/s2_albedo_outputs/{s2_date}_S2_BLUE20m_SW_hard.tif"

    if s2_date in invalid_test_dates:
        continue
    if goes_unet_test_file not in goes_test_files_sorted:
        continue

    goes_unet_test_files_list.append(goes_unet_test_file)
    s2_unet_test_files_list.append(s2_unet_test_file)
goes_unet_test_files_list = goes_unet_test_files_list[-12:]
s2_unet_test_files_list = s2_unet_test_files_list[-12:]
print(f"Number of GOES U-Net test files: {len(goes_unet_test_files_list)}")
print(f"Number of Sentinel-2 U-Net test files: {len(s2_unet_test_files_list)}")
print(goes_unet_test_files_list)
print(s2_unet_test_files_list)

# NEW TITUS
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
    Extract and preprocess training or test data from GOES or S2 albedo files.

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
        List of valid S2 dates that match with GOES dates (used when `goes=False`).
    invalid_dates : List[datetime]
        Dates that should be excluded due to excessive NaNs.
    is_goes : bool
        If True, processes GOES data. If False, processes S2 data.
    use_masked_goes_dir : bool
        If True, uses masked GOES data directory. Ignored for S2.

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
    - S2 files are assumed to have a single image per date.
    - NaNs are interpolated using bilinear interpolation and padded to match model input shape.
    
    Thus:
    - List of dates with invalid data at 18:30 (use 19:30 instead).
    - List of dates with invalid data at both times (skip)
    - If training: Goes bool is true. If test: S2: Goes bool is false

    Returns: (data_dict, mask_dict, paths_out_dict). For GOES, mask_dict=None.
    For S2, mask_dict is per-pixel 0/1 weights (same padding as data).
    """
    data_out: Dict[datetime, np.ndarray] = {}
    mask_out: Dict[datetime, np.ndarray] | None = ({} if not is_goes else None)
    paths_out: Dict[datetime, Path] = {}

    dir_path = (MASKED_GOES_ALBEDO_DIR if (is_goes and use_masked_goes_dir) else
                GOES_ALBEDO_DIR if is_goes else
                S2_BLUE_SKY_ALBEDO_DIR)
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
# END NEW TITUS

# NEW TITUS
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
# END NEW TITUS



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
    # padded = np.pad(arr, pad_width=((1, 2), (2, 3)), mode="reflect")
    
    #######################################
    #######################################
    # NEW Titus
    padded = np.pad(arr, pad_width=((1, 2), (2, 3)), mode="constant", constant_values=0.0)
    #######################################
    #######################################
    
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

    # # Coerce to 4D with singleton channel
    # if X.ndim == 3: X = X[..., np.newaxis]
    # if Y.ndim == 3: Y = Y[..., np.newaxis]
    # if M.ndim == 3: M = M[..., np.newaxis]
    # if X.shape[-1] != 1: X = X[..., :1]
    # if Y.shape[-1] != 1: Y = Y[..., :1]
    # if M.shape[-1] != 1: M = M[..., :1]

    ######################################
    ######################################
    # NEW Titus
    if X.ndim == 3: X = X[..., np.newaxis]
    if Y.ndim == 3: Y = Y[..., np.newaxis]
    if M.ndim == 3: M = M[..., np.newaxis]

    # KEEP X channels (could be 1 or 2+). Do NOT slice it down.
    # Only enforce singleton channel for Y and M.
    if Y.shape[-1] != 1: Y = Y[..., :1]
    if M.shape[-1] != 1: M = M[..., :1]
    ######################################
    ######################################
    
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
def EncoderMiniBlock(inputs, n_filters=32, max_pooling=True):
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
    
    # # First convolution
    # conv = Conv2D(n_filters, 
    #               kernel,   # Kernel size   
    #               activation='relu',
    #               padding='same')(inputs)
    # # Second convolution
    # conv = Conv2D(n_filters, 
    #               kernel,   # Kernel size
    #               activation='relu',
    #               padding='same')(conv)
    
    ########################################
    ########################################
    # NEW Titus
    # First convolution
    conv = Conv2D(n_filters, 
                  kernel,     # Kernel size   
                  activation='relu',
                  padding='same')(inputs)
    # Second convolution
    conv = Conv2D(n_filters, 
                  kernel,     # Kernel size 
                  activation='relu',
                  padding='same')(conv)

    # NEW: dropout regularization
    conv = tf.keras.layers.Dropout(0.15)(conv)
    ########################################
    ########################################
    
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
def DecoderMiniBlock(prev_layer_input, skip_layer_input, n_filters=32,
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
    
    # # Apply two convolutional layers to refine features
    # conv = Conv2D(n_filters, 
    #              kernel,     # Kernel size
    #              activation='relu',
    #              padding='same')(merge)
    # conv = Conv2D(n_filters,
    #              kernel,   # Kernel size
    #              activation='relu',
    #              padding='same')(conv)
    
    ########################################
    ########################################
    # NEW Titus
    # Apply two convolutional layers to refine features
    conv = Conv2D(n_filters, 
                 kernel,     # Kernel size
                 activation='relu',
                 padding='same')(merge)
    conv = Conv2D(n_filters,
                 kernel,     # Kernel size
                 activation='relu',
                 padding='same')(conv)

    # NEW: dropout regularization
    conv = tf.keras.layers.Dropout(0.15)(conv)
    ########################################
    ########################################
    
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
    n_filters = 32
    inputs = Input((None, None, 2))  # accept any H×W

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
    goes_training_data_4d,
    s2_training_data_4d,
    s2_training_mask_3d,             # NEW
    combined_validation_data,        # (x_val, y_val)
    val_mask_3d,                     # NEW
    goes_test_data_final,
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
    Train, evaluate, and test a U-Net model for surface albedo prediction using GOES and S2 data.

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
    s2_training_data_4d : np.ndarray
        4D training target data from S2 (samples, height, width, channels).
    combined_validation_data : tuple
        Tuple of validation inputs and labels (x_val, y_val).
    goes_test_data_final : np.ndarray
        4D test input data from GOES.
    s2_test_data_final : np.ndarray
        4D test target data from S2.
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
                goes_training_data_4d,             # x
                s2_training_data_4d,                # y
                sample_weight=s2_training_mask_3d[..., 0],  # per-pixel weights (N,H,W)
                epochs=150,
                # epochs=4,
                # validation_data=(combined_validation_data[0], combined_validation_data[1], val_mask_3d),
                
                #######################################################
                #######################################################
                # NEW Titus
                validation_data=(combined_validation_data[0], combined_validation_data[1], val_mask_3d[..., 0]),
                #######################################################
                #######################################################
                
                callbacks=[early_stopping, reduce_lr, cp_callback],
                verbose=2
            )
            os.makedirs(os.path.dirname(TF_FELIX_MODEL_UNMASKED_PATH), exist_ok=True)
            model.save(TF_FELIX_MODEL_UNMASKED_PATH)

            # ensure concrete numpy float32 (avoids unknown-rank issues)
            x_eval = np.asarray(goes_test_data_final, dtype="float32")
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
            goes_training_data_4d,
            s2_training_data_4d,
            sample_weight=s2_training_mask_3d[..., 0],
            epochs=150,
            # epochs=4,
            # validation_data=(combined_validation_data[0], combined_validation_data[1], val_mask_3d),
            
            #######################################################
            #######################################################
            # NEW Titus
            validation_data=(combined_validation_data[0], combined_validation_data[1], val_mask_3d[..., 0]),
            #######################################################
            #######################################################
            
            callbacks=[early_stopping, reduce_lr, cp_callback],
            verbose=2
        )
        os.makedirs(os.path.dirname(TF_FELIX_MODEL_UNMASKED_PATH), exist_ok=True)
        model.save(TF_FELIX_MODEL_UNMASKED_PATH)
        
        # ensure concrete numpy float32 (avoids unknown-rank issues)
        x_eval = np.asarray(goes_test_data_final, dtype="float32")
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
# --- NEW: reproject GOES onto the exact S2 grid (CRS, transform, shape) ---
def reproject_to_match(src_da: xr.DataArray, ref_da: xr.DataArray,
                       resampling: Resampling = Resampling.bilinear) -> xr.DataArray:
    # If either is missing CRS, try to repair from the other
    if not ref_da.rio.crs:
        raise ValueError("Reference raster has no CRS (S2 file likely not georeferenced)")

    if not src_da.rio.crs:
        # Assume source is meant to be on the reference CRS (common for model outputs)
        src_da = src_da.rio.write_crs(ref_da.rio.crs, inplace=False)

    return src_da.rio.reproject_match(ref_da, resampling=resampling)

# --- NEW: build stacks directly from aligned lists (same length/order) ---
def load_stacks_from_lists(
    goes_paths: List[str | Path],
    s2_paths: List[str | Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, List[Path]]:
    """
    Returns:
      X_4d : (N, H, W, 1)  # GOES resampled to S2 grid, padded to 24x24
      Y_4d : (N, H, W, 1)  # S2 target on its native grid, padded to 24x24
      M_3d : (N, H, W)     # target validity mask (1 valid, 0 invalid), padded with zeros
      s2_refs : list[Path] # S2 file paths (for georeferencing when saving predictions)
    Assumes each raster is single-band albedo (use band=1 if multi-band).
    """
    assert len(goes_paths) == len(s2_paths), "GOES/S2 lists must have equal length"
    X_list, Y_list, M_list, s2_refs = [], [], [], []

    for g_path, s2_path in zip(goes_paths, s2_paths):
        g_path, s2_path = Path(g_path), Path(s2_path)

        # load
        g_da  = load_raster_da(g_path)
        s2_da = load_raster_da(s2_path)

        g2d  = g_da.sel(band=1, drop=True) if "band" in g_da.dims else g_da
        s22d = s2_da.sel(band=1, drop=True) if "band" in s2_da.dims else s2_da

        # reproject GOES -> S2
        g_on_s2 = reproject_to_match(g2d, s22d, resampling=Resampling.bilinear)

        # target mask from S2 validity
        y_arr = s22d.values.astype("float32")
        msk   = np.isfinite(y_arr).astype("float32")

        # fill NaNs for compute, then pad to 24×24
        y_filled = np.nan_to_num(y_arr, nan=0.0)
        y_pad    = pad_da_2d(xr.DataArray(y_filled, dims=("y","x")))
        # IMPORTANT: padding in mask must be zeros so it never contributes
        g_pad    = np.pad(msk, pad_width=((1, 2), (2, 3)), mode="constant", constant_values=0.0)
        
        ##################################################################################
        ##################################################################################
        # NEW Titus
        # inputs: GOES on S2 grid -> fill & pad
        x_arr = g_on_s2.values.astype("float32")
        x_valid = np.isfinite(x_arr).astype("float32")   # NEW: 1 valid, 0 missing
        x_arr = np.nan_to_num(x_arr, nan=0.0)
        x_pad = pad_da_2d(xr.DataArray(x_arr, dims=("y","x")))

        xv_pad = np.pad(x_valid, pad_width=((1, 2), (2, 3)), mode="constant", constant_values=0.0)

        # CHANGE: stack 2 channels instead of 1
        X_list.append(np.stack([x_pad, xv_pad], axis=-1))   # (H,W,2)
        ##################################################################################
        ##################################################################################

        Y_list.append(y_pad[..., np.newaxis])  # (H,W,1)
        # M_list.append(g_pad)                   # (H,W)
        mask_pad = np.pad(msk, pad_width=((1, 2), (2, 3)), mode="constant", constant_values=0.0)
        M_list.append(mask_pad)                # (H,W)
        s2_refs.append(s2_path)

    X_4d = np.stack(X_list, axis=0)
    Y_4d = np.stack(Y_list, axis=0)
    M_3d = np.stack(M_list, axis=0)
    return X_4d, Y_4d, M_3d, s2_refs

#######################################################
#######################################################
def main(valid_test_dates=False):
    dest_folder = "/bsuhome/tnde/scratch/felix/unet_goes_s2/Unet_test_preds_s2_new/"
    os.makedirs(dest_folder, exist_ok=True)
    start = time.time()

    # local aliases
    all_goes_train = goes_unet_train_files_list
    all_s2_train   = s2_unet_train_files_list

#     if valid_test_dates:
#         goes_test  = goes_unet_test_files_list
#         s2_test    = s2_unet_test_files_list
#         goes_train = all_goes_train
#         s2_train   = all_s2_train
#     else:
#         n_total   = len(all_goes_train)
#         test_size = max(1, int(0.3 * n_total))
#         goes_test  = all_goes_train[-test_size:]
#         s2_test    = all_s2_train[-test_size:]
#         goes_train2 = all_goes_train[:-test_size]
#         s2_train2   = all_s2_train[:-test_size]

#     ##################################################################################
#     ##################################################################################
#     # NEW Titus
#     n = len(s2_train2)
#     n_train = int(0.8 * n)

#     goes_train = goes_train2[:n_train]
#     s2_train   = s2_train2[:n_train]
#     goes_val   = goes_train2[n_train:]
#     s2_val     = s2_train2[n_train:]
#     ##################################################################################
#     ##################################################################################
    
    # n = len(s2_train2)

    # n_train = int(1 * n)
    # goes_val  = goes_train2[:n_train]
    # s2_val    = s2_train2[:n_train]
    # goes_train = goes_train2[:n_train]
    # s2_train   = s2_train2[:n_train]
    
    ##################################################################################
    ##################################################################################
    # NEW Titus
    if valid_test_dates:
        goes_test = goes_unet_test_files_list
        s2_test   = s2_unet_test_files_list

        goes_pool = all_goes_train
        s2_pool   = all_s2_train
    else:
        n_total   = len(all_goes_train)
        test_size = max(1, int(0.3 * n_total))

        goes_test = all_goes_train[-test_size:]
        s2_test   = all_s2_train[-test_size:]

        goes_pool = all_goes_train[:-test_size]
        s2_pool   = all_s2_train[:-test_size]

    # split remaining data into train and validation
    n_pool = len(s2_pool)
    if n_pool < 3:
        raise ValueError("Not enough samples left after test split to create separate train/val sets.")
    val_size = max(2, int(0.2 * n_pool))

    goes_train = goes_pool[:-val_size]
    s2_train   = s2_pool[:-val_size]

    goes_val   = goes_pool[-val_size:]
    s2_val     = s2_pool[-val_size:]

    print(f"Train samples: {len(goes_train)}")
    print(f"Val samples:   {len(goes_val)}")
    print(f"Test samples:  {len(goes_test)}")
    ##################################################################################
    ##################################################################################

    # Build stacks (GOES reprojected to S2 grid; both padded to 24x24)
    X_train, Y_train, M_train, s2_train_refs = load_stacks_from_lists(goes_train, s2_train)
    X_val,   Y_val,   M_val,   s2_val_refs   = load_stacks_from_lists(goes_val,   s2_val)
    X_test,  Y_test,  M_test,  s2_test_refs  = load_stacks_from_lists(goes_test,  s2_test)

    print("Train shapes:", X_train.shape, Y_train.shape, M_train.shape)
    print("Val shapes:", X_val.shape, Y_val.shape, M_val.shape)
    print("Test shapes:", X_test.shape, Y_test.shape, M_test.shape)

    # # Apply padding. Pad to multiples-of-8 for the model
    X_train, Y_train, M_train, pad_train = pad_batch_to_multiple(X_train, Y_train, M_train, mult=8)
    X_val, Y_val, M_val, pad_val = pad_batch_to_multiple(X_val, Y_val, M_val, mult=8)
    X_test, Y_test, M_test, pad_test = pad_batch_to_multiple(X_test, Y_test, M_test, mult=8)
    
    ######################################
    ######################################
    # NEW Titus
    print("After pad, X_train:", X_train.shape)  # should end with 2
    print("After pad, X_val:  ", X_val.shape)
    print("After pad, X_test: ", X_test.shape)
    ######################################
    ######################################

    # Sanity checks
    for X in (X_train, X_val, X_test):
        if not np.isfinite(X).all():
            raise ValueError("Inputs contain NaN/Inf")

    # ----------------- train/eval/save -----------------
    run_unet(
        unet_dimensions=(None, None, 2),
        load_weights_bool=False,
        load_model=False,
        goes_training_data_4d=X_train,
        s2_training_data_4d=Y_train,
        s2_training_mask_3d=M_train,
        combined_validation_data=(X_val, Y_val),
        val_mask_3d=M_val,
        goes_test_data_final=X_test,
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
