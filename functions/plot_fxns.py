# Packages
import os
import re
import json
import sys
import numpy as np
import pandas as pd
import xarray as xr
import rasterio as rio
import rioxarray as rxr
import geopandas as gpd
from pathlib import Path
from rasterio.plot import show
import matplotlib.pyplot as plt
from shapely.geometry import box
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter
from sklearn.metrics import r2_score, mean_squared_error

##################################################################
##################################################################
## Path to data files

# GOES raw data
GOES_Raw_2km_Albedo_dir = "/bsuhome/tnde/scratch/felix/GOES/data/ABI-L2-LSAC/2021/244/18/"
GOES_raw_data_ex_sep_1 = "/bsuhome/tnde/scratch/felix/GOES/data/ABI-L2-LSAC/2021/244/18/OR_ABI-L2-LSAC-M6_G16_s20212441821171_e20212441823544_c20212441825061.nc"
GOES_2km_ex_may_6_2023 = "/bsuhome/tnde/scratch/felix/GOES/data/ABI-L2-LSAC/2023/156/18/OR_ABI-L2-LSAC-M6_G16_s20231561826179_e20231561828552_c20231561830350.nc"

# /global/cfs/cdirs/m3779/felix

# Modis data
MODIS_raw_data_dir = "/bsuhome/tnde/scratch/felix/modis/modis-data/"
MODIS_raw_data_ex_sep_1 = "/bsuhome/tnde/scratch/felix/modis/modis-data/MCD43A3.A2021244.h09v05.061.2021253060517.hdf"
# MODIS_bsa_dir = "/bsuhome/tnde/scratch/felix/modis/blue_sky_albedo_sail_new/"
MODIS_bsa_dir = "/bsuhome/tnde/scratch/felix/modis/blue_sky_albedo_sail/"
MODIS_bsa_ex_sep_1 = "/bsuhome/tnde/scratch/felix/modis/blue_sky_albedo_sail_new/2021244_modis_blue_sky_albedo_.tif"
# MODIS_interpolated_data_dir = "/bsuhome/tnde/scratch/felix/modis/interpolated-500m_new/"
MODIS_interpolated_data_dir = "/bsuhome/tnde/scratch/felix/modis/interpolated-500m/"

# Goes U-Net Outputs
GOES_500m_data_dir = "/bsuhome/tnde/scratch/UNet/Results_new/"
GOES_500m_unmasked_data_path = "/bsuhome/tnde/scratch/felix/UNet/Results_new/Train-Start=09-01-2021-Train-End=12-31-2022-Test-Start=09-01-2021-Test-End=06-15-2023_not_masked_new.npy"
GOES_500m_masked_data_path = "/bsuhome/tnde/scratch/felix/UNet/Results_new/Train-Start=09-01-2021-Train-End=12-31-2022-Test-Start=09-01-2021-Test-End=06-15-2023_masked_new.npy"
GOES_500m_less_data_path = "/bsuhome/tnde/scratch/felix/UNet/Results_new/Train-Start=09-01-2021-Train-End=12-31-2022-Test-Start=01-01-2023-Test-End=06-15-2023_less_data.npy"
GOES_500m_less_data_v2_path = "/bsuhome/tnde/scratch/felix/UNet/Results_new/Train-Start=09-01-2021-Train-End=12-31-2022-Test-Start=01-01-2023-Test-End=06-15-2023_less_data_v2.npy"
GOES_500m_threshold_60_path = "/bsuhome/tnde/scratch/felix/UNet/Results_new/Train-Start=09-01-2021-Train-End=12-31-2022-Test-Start=01-01-2023-Test-End=06-15-2023_threshold_60.npy"

# GOES U-Net Outputs rasterized
GOES_500m_raster_dir = "/bsuhome/tnde/scratch/felix/GOES/data/500m-raster_new/"
# GOES_500m_masked_output_dir = "/bsuhome/tnde/scratch/felix/GOES/data/500m-masked-raster_new/"
GOES_500m_masked_output_dir = "/bsuhome/tnde/scratch/felix/GOES/data/500m-masked-raster/"
GOES_500m_less_data_output_dir = "/bsuhome/tnde/scratch/felix/GOES/data/500m-less-data_new/"
GOES_500m_less_data_v2_raster_dir = "/bsuhome/tnde/scratch/felix/GOES/data/500m-less-data-v2_new/"
GOES_500m_threshold_60_raster_dir = "/bsuhome/tnde/scratch/felix/GOES/data/500m-threshold-60_new/"
GOES_NaN_Data_dir = "/bsuhome/tnde/scratch/felix/GOES/data/nan_data_new/"
# INVALID_DATES_PATH = "/bsuhome/tnde/scratch/felix/modis/invalid_modis_dates_new.json"
INVALID_DATES_PATH = "/bsuhome/tnde/scratch/felix/modis/invalid_modis_dates.json"

# Individual data files
GOES_500m_05_06_23_file = "/bsuhome/tnde/scratch/felix/GOES/data/500m-masked-raster_new/05-06-2023-GOES-500m_new.tif"
MODIS_500m_05_06_23_file = "/bsuhome/tnde/scratch/felix/modis/blue_sky_albedo_sail_new/2023127_modis_blue_sky_albedo_.tif"
SAIL_field_data_file = "/bsuhome/tnde/scratch/felix/Field-Albedo-Data/albedo_12MST_M1.csv"

# Invalid dates
INVALID_GOES_DATES = [datetime(2022, 12, 16), datetime(2022, 12, 18), datetime(2022, 12, 19), datetime(2023, 1, 7), datetime(2023, 1, 8), datetime(2022, 4, 8), datetime(2023, 5, 11), datetime(2023, 3, 14)]

# Shapefile
shapefile_path = '/bsuhome/tnde/scratch/felix/modis/East_River_SHP/ER_bbox.shp'
Boundary = gpd.read_file(shapefile_path)
Boundary_reproj_lat = Boundary.to_crs(epsg=4326)
boundary_box_lat = [box(*Boundary_reproj_lat.total_bounds)]

Boundary_reproj_utm = Boundary.to_crs(epsg=32613)
boundary_box_utm = [box(*Boundary_reproj_utm.total_bounds)]

# Minimum support for a date to be counted
MIN_VALID_PIX_FRAC = 0.60   # require at least 60% valid pixels
MIN_VALID_PIX_ABS  = 238     # and at least 238 pixels absolutely
MIN_VAR = 1e-8

##################################################################
##################################################################
# Write all 500m GOES data to a raste
def write_goes_to_raster(input_unet_data_path, output_dir, invalid_dates_in, start_date, end_date):
    """
    Writes predicted GOES 500m albedo data (from U-Net) to GeoTIFF format, using MODIS files for spatial reference.

    Args:
        input_unet_data_path (str or Path): Path to `.npy` file containing predicted U-Net GOES data 
                                            with shape (num_days, height, width).
        output_dir (str or Path): Directory where output GeoTIFF files will be saved.
        invalid_dates_in (list of datetime): List of dates to skip (e.g., due to missing data).
        start_date (datetime): Starting date of prediction period.
        end_date (datetime): Ending date of prediction period.

    Returns:
        None
    """
    modis_files = [p for p in Path(MODIS_bsa_dir).iterdir() if p.is_file()]
    goes_data = np.load(input_unet_data_path)  # shape: (N, H, W)

    os.makedirs(output_dir, exist_ok=True)

    # Optional: build a quick YYYYJJJ -> Path index for speed and determinism
    def key_from_path(p):  # expects filenames containing YYYYJJJ
        return re.search(r'(\d{7})', p.name).group(1)
    modis_index = {key_from_path(p): p for p in modis_files if re.search(r'(\d{7})', p.name)}

    current_date = start_date
    array_counter = 0

    while current_date <= end_date and array_counter < goes_data.shape[0]:
        if current_date in invalid_dates_in:
            current_date += timedelta(days=1)
            continue

        date_key = current_date.strftime('%Y%j')
        mfile = modis_index.get(date_key, None)
        if mfile is None:
            print(f"No MODIS file for {current_date:%Y-%m-%d}; not consuming prediction idx {array_counter}.")
            current_date += timedelta(days=1)
            continue

        cur_arr_data = goes_data[array_counter]  # (H, W)

        modis_rxr = rxr.open_rasterio(mfile, masked=True).sel(band=1)
        y, x, crs = modis_rxr['y'], modis_rxr['x'], modis_rxr.rio.crs

        # Sanity: shapes must match exactly
        if cur_arr_data.shape != (y.size, x.size):
            raise ValueError(f"Shape mismatch for {current_date:%Y-%m-%d}: "
                             f"pred {cur_arr_data.shape} vs MODIS {(y.size, x.size)}")

        goes_500m_data_array = xr.DataArray(cur_arr_data, dims=['y', 'x'], coords={'y': y, 'x': x}).rio.write_crs(crs)
        out_path = Path(output_dir) / f"{current_date:%m-%d-%Y}-GOES-500m_new.tif"
        goes_500m_data_array.rio.to_raster(out_path)

        # Only now advance the prediction index
        array_counter += 1
        current_date += timedelta(days=1)

    print("Execution complete!!!")

##################################################################
##################################################################
# ---- Date parsers ----
def parse_goes_date(path: Path):
    """
    Expects GOES file names like 'MM-DD-YYYY-GOES-500m_new.tif'.
    Returns datetime.date or None.
    """
    m = re.search(r'(\d{2}-\d{2}-\d{4})', path.name)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%m-%d-%Y").date()

def parse_modis_date(path: Path):
    """
    Expects MODIS file names containing 'YYYYJJJ' somewhere.
    Returns datetime.date or None.
    """
    m = re.search(r'(\d{7})', path.name)  # YYYYJJJ
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%j").date()

def index_by_date(paths, kind="goes"):
    """
    Build {date: path} using the right parser.
    """
    idx = {}
    for p in paths:
        if not p.is_file():
            continue
        d = parse_goes_date(p) if kind == "goes" else parse_modis_date(p)
        if d is not None:
            idx[d] = p
    return idx

# ---- Shared guards ----
def ensure_2d_dataarray(da):
    # Accepts DataArray with optional band dimension
    if "band" in da.dims:
        return da.sel(band=1)
    return da

def np2d(a):
    arr = np.asarray(a)
    if arr.ndim == 3:  # sometimes (1,H,W)
        arr = np.squeeze(arr, axis=0)
    assert arr.ndim == 2, f"Expected 2D array, got shape {arr.shape}"
    return arr


##################################################################
##################################################################
# ---- R2 ----
def calculate_r2_scores(goes_files, modis_raw_files):
    """
    Calculates masked R-squared scores for GOES 500m predictions against MODIS ground truth data and plots the results.
    Also computes average R-squared across all prediction-ground truth pairs, and tracks dates with poor model performance.

    Args:
        goes_files (list[Path]): List of file paths to predicted GOES GeoTIFF files from U-Net output.
        modis_raw_files (list[Path]): List of file paths to raw MODIS blue-sky albedo GeoTIFFs (used for identifying valid pixels).
        modis_interpolated_files (list[Path]): List of file paths to interpolated MODIS GeoTIFFs (used as ground truth).

    Returns:
        None. Prints statistics, visualizes R² scores, and lists dates with bad predictions.
    """
    goes_idx  = index_by_date([Path(p) for p in goes_files], kind="goes")
    modis_idx = index_by_date([Path(p) for p in modis_raw_files], kind="modis")

    common_dates = sorted(set(goes_idx.keys()) & set(modis_idx.keys()))
    if not common_dates:
        print("No overlapping dates between GOES and MODIS. Check filename patterns and date ranges.")
        print(f"Example GOES parsed dates (up to 5): {list(goes_idx.keys())[:5]}")
        print(f"Example MODIS parsed dates (up to 5): {list(modis_idx.keys())[:5]}")
        return np.nan

    masked_r2_scores = {}
    bad_predictions_missing_pixels = {}
    all_pred, all_true = [], []

    for d in common_dates:
        gpath, mpath = goes_idx[d], modis_idx[d]

        # Load predicted GOES (should be single-band 2D)
        y_pred = np2d(rxr.open_rasterio(gpath).squeeze().values)

        # Load raw MODIS (truth) and mask invalid
        modis_da = ensure_2d_dataarray(rxr.open_rasterio(mpath))
        y_true = np2d(modis_da.values)

        if y_pred.shape != y_true.shape:
            print(f"[WARN] Shape mismatch for {d}: pred {y_pred.shape} vs MODIS {y_true.shape}")
            continue

        mask = np.isfinite(y_true)
        n_valid = int(mask.sum())
        frac_valid = n_valid / mask.size

        if (n_valid < MIN_VALID_PIX_ABS) or (frac_valid < MIN_VALID_PIX_FRAC):
            # too few valid pixels; skip
            continue
        if np.var(y_true[mask]) < MIN_VAR:
            # near-constant truth; R² undefined/unstable
            continue

        r2 = r2_score(y_true[mask], y_pred[mask])
        if r2 > 0:
            masked_r2_scores[d] = r2
            all_pred.append(y_pred[mask]); all_true.append(y_true[mask])
        else:
            pct_missing = round((1 - frac_valid) * 100, 3)
            bad_predictions_missing_pixels[d] = pct_missing

    # Aggregate masked R-square
    if all_pred and all_true:
        agg_pred = np.concatenate(all_pred)
        agg_true = np.concatenate(all_true)
        r2_agg = r2_score(agg_true, agg_pred)
        print(f"\nAverage (masked) R² over kept dates: {r2_agg:.4f}")
    else:
        print("\nNo dates qualified for aggregate R² after filtering.")
        r2_agg = np.nan

    if bad_predictions_missing_pixels:
        print("\nDates with very poor R-square (after filters):")
        for dt, pct in sorted(bad_predictions_missing_pixels.items()):
            print(f"  {dt:%Y-%m-%d} — Missing Pixels: {pct:.2f}%")

    return r2_agg

##################################################################
##################################################################
# ---- RMSE ----
def calculate_RMSE_scores(goes_files, modis_raw_files):
    """
    Computes Root Mean Squared Error (RMSE) scores between U-Net predicted GOES images and MODIS interpolated ground truth.
    It also compares these scores against the percentage of missing pixels in the raw MODIS input.

    Args:
        goes_files (List[Path]): List of predicted GOES GeoTIFF files from U-Net model output.
        modis_raw_files (List[Path]): List of raw MODIS blue-sky albedo GeoTIFFs used to determine valid (non-NaN) pixels.
        modis_interpolated_files (List[Path]): List of MODIS interpolated GeoTIFFs used as the ground truth.

    Returns:
        None. Displays RMSE summary and plots RMSE vs missing pixel percentage.
    """
    goes_idx  = index_by_date([Path(p) for p in goes_files], kind="goes")
    modis_idx = index_by_date([Path(p) for p in modis_raw_files], kind="modis")

    common_dates = sorted(set(goes_idx.keys()) & set(modis_idx.keys()))
    if not common_dates:
        print("No overlapping dates between GOES and MODIS. Check filename patterns and date ranges.")
        print(f"Example GOES parsed dates (up to 5): {list(goes_idx.keys())[:5]}")
        print(f"Example MODIS parsed dates (up to 5): {list(modis_idx.keys())[:5]}")
        return

    masked_rmse_scores = {}
    missing_pixels = {}
    all_pred, all_true = [], []

    for d in common_dates:
        gpath, mpath = goes_idx[d], modis_idx[d]

        y_pred = np2d(rxr.open_rasterio(gpath).squeeze().values)
        modis_da = ensure_2d_dataarray(rxr.open_rasterio(mpath))
        y_true = np2d(modis_da.values)

        if y_pred.shape != y_true.shape:
            print(f"[WARN] Shape mismatch for {d}: pred {y_pred.shape} vs MODIS {y_true.shape}")
            continue

        mask = np.isfinite(y_true)
        n_valid = int(mask.sum())
        frac_valid = n_valid / mask.size

        if (n_valid < MIN_VALID_PIX_ABS) or (frac_valid < MIN_VALID_PIX_FRAC):
            continue

        rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
        masked_rmse_scores[d] = rmse
        missing_pixels[d] = round((1 - frac_valid) * 100, 3)
        all_pred.append(y_pred[mask]); all_true.append(y_true[mask])

    if all_pred and all_true:
        agg_pred = np.concatenate(all_pred); agg_true = np.concatenate(all_true)
        aggregate_rmse = np.sqrt(mean_squared_error(agg_true, agg_pred))
        print(f"\nAggregate masked RMSE: {aggregate_rmse:.5f}")
    else:
        print("\nNo dates qualified for aggregate RMSE after filtering.")

    if masked_rmse_scores:
        rmse_dates = list(masked_rmse_scores.keys())
        rmse_values = list(masked_rmse_scores.values())
        missing_vals = [missing_pixels[d] for d in rmse_dates]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6), sharey=True)
        ax1.scatter(missing_vals, rmse_values, marker='o')
        ax1.set_title("RMSE vs missing pixels (%) in MODIS"); ax1.set_xlabel("% Missing"); ax1.set_ylabel("RMSE"); ax1.grid(True)
        ax2.scatter(rmse_dates, rmse_values, marker='o')
        ax2.set_title("RMSE over time"); ax2.set_xlabel("Date")
        ax2.xaxis.set_major_locator(mdates.MonthLocator()); ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        ax2.tick_params(axis='x', rotation=45); ax2.grid(True)
        plt.tight_layout(w_pad = -0.02); plt.show()

# ---- Missing vs time ----
def plot_pixels_per_date(goes_files, modis_raw_files):
    """
    Computes Root Mean Squared Error (RMSE) scores between U-Net predicted GOES images and MODIS interpolated ground truth.
    It also compares these scores against the percentage of missing pixels in the raw MODIS input.
    """
    goes_idx  = index_by_date([Path(p) for p in goes_files], kind="goes")
    modis_idx = index_by_date([Path(p) for p in modis_raw_files], kind="modis")

    common_dates = sorted(set(goes_idx.keys()) & set(modis_idx.keys()))
    if not common_dates:
        print("No overlapping dates between GOES and MODIS. Check filename patterns and date ranges.")
        print(f"Example GOES parsed dates (up to 5): {list(goes_idx.keys())[:5]}")
        print(f"Example MODIS parsed dates (up to 5): {list(modis_idx.keys())[:5]}")
        return

    masked_rmse_scores = {}
    missing_pixels = {}
    all_pred, all_true = [], []

    for d in common_dates:
        gpath, mpath = goes_idx[d], modis_idx[d]

        y_pred = np2d(rxr.open_rasterio(gpath).squeeze().values)
        modis_da = ensure_2d_dataarray(rxr.open_rasterio(mpath))
        y_true = np2d(modis_da.values)

        if y_pred.shape != y_true.shape:
            print(f"[WARN] Shape mismatch for {d}: pred {y_pred.shape} vs MODIS {y_true.shape}")
            continue

        mask = np.isfinite(y_true)
        n_valid = int(mask.sum())
        frac_valid = n_valid / mask.size

        if (n_valid < MIN_VALID_PIX_ABS) or (frac_valid < MIN_VALID_PIX_FRAC):
            continue

        rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
        masked_rmse_scores[d] = rmse
        missing_pixels[d] = round((1 - frac_valid) * 100, 3)
        all_pred.append(y_pred[mask])
        all_true.append(y_true[mask])

    if all_pred and all_true:
        agg_pred = np.concatenate(all_pred)
        agg_true = np.concatenate(all_true)
        aggregate_rmse = np.sqrt(mean_squared_error(agg_true, agg_pred))
        print(f"\nAggregate masked RMSE: {aggregate_rmse:.5f}")
    else:
        print("\nNo dates qualified for aggregate RMSE after filtering.")

    # ---------- NEW PLOT: % missing vs time ----------
    if masked_rmse_scores:
        # Use the same date set used for RMSE, so keys line up with missing_pixels dict
        dates = list(masked_rmse_scores.keys())
        missing_vals = [missing_pixels[d] for d in dates]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(dates, missing_vals, marker='o')
        ax.set_title("Percentage of missing MODIS pixels over time")
        ax.set_xlabel("Date")
        ax.set_ylabel("% Missing pixels")

        # Format x axis as time
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        ax.tick_params(axis='x', rotation=0)
        ax.grid(True)

        plt.tight_layout()
        plt.show()
        
##################################################################
##################################################################
# Visualize GOES U-Net 500m and MODIS 500m next to each other
def extract_date_from_goes_filename(filename: str) -> datetime:
    """
    Extracts the datetime object from a GOES filename based on the assumed format.

    Expected filename format:
        'MM-DD-YYYY-...tif' or similar, where the first three segments of the filename 
        (split by '-') represent the month, day, and year.

    Args:
        filename (str): The name of the GOES file, e.g., '08-15-2022-GOES-500m.tif'

    Returns:
        datetime: A datetime object parsed from the filename.

    Raises:
        ValueError: If the extracted string cannot be parsed into a datetime object.
    """
    # Extract the 'MM-DD-YYYY' part from the filename
    date_str = "-".join(filename.split("-")[0:3])
    # Convert to datetime object
    date_obj = datetime.strptime(date_str, '%m-%d-%Y')
    return date_obj

##################################################################
##################################################################
# Visualize GOES NaN Mask 500m and MODIS Ground Truth Data (non-masked)
# Oroginal
def extract_date_from_goes_nan_filename(filename):
    """
    Extracts the datetime object from a GOES filename that contains a timestamp in the format 'sYYYYJJJHH'.

    Parameters:
    ----------
    filename : str
        The name of the GOES file, expected to follow the format:
        e.g., 'OR_ABI-L2-LSAC-M6_G16_s20231631826173_e20231631828546_c20231631830241_clipped_reprojected.tif'
        where the start time token 's20231631826173' is the 4th underscore-separated token,
        and the date string to extract is the first 9 characters after the leading 's':
        '202316318' => year 2023, Julian day 163, hour 18.

    Returns:
    -------
    datetime
        A datetime object representing the parsed year, Julian day, and hour.
    """
    # Extract the string starting with 'sYYYYJJJHH' (9 characters: YYYYJJJHH)
    date_str = filename.split("_")[3][1:10]  # Skip the leading 's'

    # Parse the date string using the appropriate datetime format
    date_obj = datetime.strptime(date_str, '%Y%j%H')  # e.g., '202316318' -> 2023-06-12 18:00:00

    return date_obj

##################################################################
##################################################################


