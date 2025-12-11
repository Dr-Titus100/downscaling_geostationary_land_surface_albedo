# Packages
import os
import re
import sys
import json
import numpy as np
import xarray as xr
import rasterio as rio
import rioxarray as rxr
import geopandas as gpd
from pathlib import Path
from affine import Affine
from rasterio.plot import show
import matplotlib.pyplot as plt
from shapely.geometry import box
from rasterio.enums import Resampling
from datetime import datetime, timedelta
from pyproj import Proj, CRS, Transformer
from rioxarray.exceptions import NoDataInBounds
from typing import Iterable, Optional, List, Tuple
from rasterio.warp import calculate_default_transform, reproject, Resampling
# mamba install -c conda-forge rioxarray

##################################################################
##################################################################
## Path to data files
# Albedo data
base_modis_path = '/bsuhome/tnde/scratch/felix/modis/'
base_modis_data_dir = '/bsuhome/tnde/scratch/felix/modis/modis-data'
modis_albedo_filepath = '/bsuhome/tnde/scratch/felix/modis/modis-data/MCD43A3.A2023162.h09v05.061.2023171040831.hdf'
clipped_albedo_output_data = '/bsuhome/tnde/scratch/felix/modis/clipped_modis_data_colorado' 
blue_sky_albedo_colorado_data = '/bsuhome/tnde/scratch/felix/modis/blue_sky_albedo_colorado' 
blue_sky_albedo_ex = '/bsuhome/tnde/scratch/felix/modis/blue_sky_albedo_colorado/2021237_modis_blue_sky_albedo_.tif'
blue_sky_albedo_final_data_dir = '/bsuhome/tnde/scratch/felix/modis/blue_sky_albedo_sail_new'

# AOD data
aod_data_dir = '/bsuhome/tnde/scratch/felix/modis/aod-data'
aod_data_file = '/bsuhome/tnde/scratch/felix/modis/aod-data/MCD19A2.A2023165.h09v05.061.2023166172728.hdf'
clipped_aod_output = '/bsuhome/tnde/scratch/felix/modis/clipped_aod_data_new'

# GOES Data
goes_data_dir = '/bsuhome/tnde/scratch/felix/GOES/data/ABI-L2-LSAC'
goes_data_file = '/bsuhome/tnde/scratch/felix/GOES/data/ABI-L2-LSAC/2023/012/16/OR_ABI-L2-LSAC-M6_G16_s20230121626172_e20230121628546_c20230121630337.nc'
goes_data_file_valid = '/bsuhome/tnde/scratch/felix/GOES/data/ABI-L2-LSAC/2021/244/18/OR_ABI-L2-LSAC-M6_G16_s20212441826171_e20212441828544_c20212441829544.nc'
goes_data_file_valid_name = 'OR_ABI-L2-LSAC-M6_G16_s20212441826171_e20212441828544_c20212441829544.nc'
reprojected_clipped_valid_goes_file = '/bsuhome/tnde/scratch/felix/GOES/data/goes_output_data_new/OR_ABI-L2-LSAC-M6_G16_s20230851926167_e20230851928539_c20230851930241_clipped_reprojected_new.tif'
reprojected_clipped_goes_output_dir = '/bsuhome/tnde/scratch/felix/GOES/data/goes_output_data_new'
nan_goes_data_dir = '/bsuhome/tnde/scratch/felix/GOES/data/nan_data_new'

# ===NOTE===
# LSA = Land Surface Albedo
# DQF = Data Quality Flag

# Shapefile Data
shapefile_path = '/bsuhome/tnde/scratch/felix/modis/East_River_SHP/ER_bbox.shp'
colorado_shapefile = '/bsuhome/tnde/geoscience/albedo_downscaling/GOES-Modis-Data-Preprocessing-main/shapefile_colorado/Colorado_State_Boundary/Colorado_State_Boundary.shp'
# colorado_shapefile = 'Colorado_State_Boundary.shp'
Boundary = gpd.read_file(shapefile_path)
print(Boundary.to_crs(epsg=4326).total_bounds)
colorado_boundary = gpd.read_file(colorado_shapefile)

# Upsampling factor
UPSCALE_FACTOR = 2
GOES_OFFSET = 10000
NAN_PIXEL_THRESHOLD = 0.6
NODATA = -9999.0

# Reproject GOES Bounding Box to GOES CRS
goes_ds_ex_xr = xr.open_dataset(goes_data_file_valid)
# Create GOES CRS for reprojection
goes_imager_proj_info = goes_ds_ex_xr["goes_imager_projection"]
goes_crs = CRS.from_cf(goes_imager_proj_info.attrs)

##################################################################
##################################################################
## Upsample/Downscale AOD and GOES to 500m from 1 KM for accurate clipping
# Use a bi-linear interpolation method
def resample_raster_bilinear(raster: xr.DataArray, upscale_factor: int) -> xr.DataArray:
    """
    Resample a raster dataset using bilinear interpolation by an integer upscale factor.

    Parameters
    ----------
    raster : xarray.DataArray
        The input raster to be resampled. Thus, the X-array dataset to upscale. Must have a defined CRS and use rioxarray.
        NB: This function assumes that raster is already opened with rioxarray and has a defined coordinate reference system (CRS).
    upscale_factor : int
        The factor by which to increase the spatial resolution of the input raster.
        
    NB: We can swap Resampling.bilinear with other methods, such as  Resampling.nearest, Resampling.cubic, etc., if needed.

    Returns
    -------
    xarray.DataArray
        The resampled/upsampled raster with increased spatial resolution using bilinear interpolation.
    """
    # Calculate the new spatial dimensions
    new_height = raster.rio.height * upscale_factor
    new_width = raster.rio.width * upscale_factor
    # print(f'Old height {raster.rio.height}; Old width {raster.rio.width}')
    # print(f'New height {new_height}; New width {new_width}')

    # Perform resampling/upsampling using bilinear interpolation
    resampled_raster = raster.rio.reproject(
        raster.rio.crs,
        shape=(new_height, new_width),
        resampling=Resampling.bilinear
        # resampling=Resampling.cubic
        # resampling=Resampling.nearest
    )
    return resampled_raster

##################################################################
##################################################################
def DQF_analysis(
    dqf_da,
    min_valid_frac = 0.60,     # require at least min_valid_frac*100 percent valid
    min_valid_pixels = 159,    # and at least this many valid pixels. 159 = 40%, 278 = 70%
    verbose = True,
    valid_dqf = True # # If True, apply pixel-wise Data Quality Flag filter. If False, use only missing pixels.
) -> bool:
    """
    Return True if the DQF mask indicates enough usable pixels.
    Works with xarray.DataArray (dask-backed ok).
    """
    # Guard
    if dqf_da is None:
        if verbose:
            print("[DQF] No DQF provided.")
        return False

    # Ensure finite and integer-like (nearest resampling is recommended for DQF,
    # but if it became float, we round it back)
    dqf = dqf_da.where(np.isfinite(dqf_da))
    if not np.issubdtype(dqf.dtype, np.integer):
        dqf = dqf.round().astype("uint8")
    else:
        dqf = dqf.astype("uint8")

    # Build validity mask
    if valid_dqf: 
        valid_values=(0, 1),     # keep pixels with DQF 0 or 1
        is_valid = dqf.isin(list(valid_values))
    else:
        missing_values = (255, )
        is_valid = ~dqf.isin(list(missing_values))
    # print("DQF values", dqf)
    # print("Is valid", is_valid)
    
    # Compute counts (works with/without dask)
    def _compute(x):
        return x.compute() if hasattr(x, "compute") else x

    valid_count = int(_compute(is_valid.sum()))
    total_count = int(_compute(dqf.notnull().sum()))
    valid_frac = (valid_count / total_count) if total_count > 0 else 0.0

    if verbose:
        print(f"[DQF] valid={valid_count} / total={total_count} ({valid_frac:.1%}) "
              f"| threshold: min valid frac>={min_valid_frac:.0%}, pixels>={min_valid_pixels}")
        
    return (valid_frac >= min_valid_frac) and (valid_count >= min_valid_pixels)


##################################################################
##################################################################
def DQF_analysis2(data_in):
    """
    Analyze the Data Quality Flags (DQF) for GOES ABI AOD/COD data and 
    determine whether the data contains too many invalid pixels.

    The check is based on the following GOES ABI DQF bit logic:
    - Bits 3 and 4 indicate retrieval path: if set to `11` (binary), retrieval is invalid.
    - Bit 5 distinguishes between AOD and COD.
    - A fill value of 255 also indicates invalid data.

    Parameters
    ----------
    data_in : xarray.DataArray
        Input data containing DQF values for each pixel. Expected to be an array of integers.

    Returns
    -------
    bool
        True if the dataset passes the DQF test (i.e., more than 50% of pixels are valid),
        False otherwise.
    """
    
    # Based on GOES ABI theoretical algorithm document
    # DQF Bits 3/4: signify retrieval path: If 11: No retrieval
    # Bit 5 (0 or 1) signifies AOD or COD
    dqf_values = data_in.values
    
    invalid_pixels = 0
    total_pixels = dqf_values.size
    
    # Use np.nditer to iterate over all elements
    for value in np.nditer(dqf_values):
        # Fill value is 255 for dqf
        if value == 255:
            invalid_pixels += 1
            print("True")
            print(dqf_values)
        # Bit operations to analyze bit mask
        else:
            shifted_value = value >> 3
            if shifted_value == 7 or shifted_value == 3:
                invalid_pixels += 1
    
    # If half of all pixels are invalid
    if (invalid_pixels/total_pixels) > 0.4:
        return False
    return True


##################################################################
##################################################################
def reproject_clip_and_upsample_goes_raster(
    rxr_in: xr.Dataset,
    file_name_in: str,
    boundary_box_in,
    dst_crs: str = "EPSG:32613",
    dqf_filter = False # If True, apply pixel-wise Data Quality Flag filter. If False, use only missing pixels.
) -> int:
    
    """
    Reproject, clip, and upsample GOES raster data to match MODIS resolution and extent.

    This function processes GOES data by reprojecting it from its native CRS to EPSG:4326,
    clipping it to a given spatial boundary, reprojecting to a destination CRS (e.g., UTM),
    and matching it to a MODIS dataset using bilinear interpolation. It performs a DQF (Data 
    Quality Flag) check before saving the processed output.

    Parameters
    ----------
    rxr_in : xr.Dataset
        The input rioxarray Dataset containing GOES variables ("LSA" and "DQF").
    file_name_in : str
        Name of the original GOES file (used to create the output file name).
    boundary_box_in : geopandas.GeoDataFrame or list-like
        Boundary used for clipping (must be in the same CRS as `dst_crs`).
    dst_crs : str, optional
        The target coordinate reference system for output, by default "EPSG:32613".

    Returns
    -------
    int
        Returns 0 if the processing was successful and the file was saved;
        Returns 1 if the function encountered invalid data or failed during processing.
    """
    
    # set crs for the boundary box
    boundary_box_in = gpd.GeoDataFrame(geometry=boundary_box_in, crs="EPSG:32613")
    
    # Returns 1 on failure, 0 on success
    invalid_pixel_return = 1

    # Ensure GOES CRS is set on inputs
    assert rxr_in.rio.crs is not None, "GOES input needs a CRS"
    assert getattr(boundary_box_in, "crs", None) is not None, "boundary_box_in needs a CRS"

    # Extract vars and attach CRS
    lsa_data = rxr_in["LSA"].squeeze().rio.write_crs(rxr_in.rio.crs, inplace=False)
    dqf_data = rxr_in["DQF"].squeeze().rio.write_crs(rxr_in.rio.crs, inplace=False)
    # print("dqf_data", dqf_data.values)
    # DQF_analysis(rxr_in["DQF"].squeeze())

    # Open MODIS reference, reproject to UTM, and clip with boundary (with CRS awareness)
    modis_file = rxr.open_rasterio(blue_sky_albedo_ex)
    assert modis_file.rio.crs is not None, "MODIS reference needs a CRS"

    modis_reproj = modis_file.rio.reproject(dst_crs)

    # Reproject boundary to match MODIS reproj CRS and verify overlap
    bb_utm = boundary_box_in.to_crs(modis_reproj.rio.crs)

    # Quick overlap check to avoid NoDataInBounds
    if not box(*modis_reproj.rio.bounds()).intersects(bb_utm.unary_union):
        print("No overlap between MODIS reprojection and boundary.")
        return invalid_pixel_return

    modis_clipped = modis_reproj.rio.clip(bb_utm.geometry, all_touched=True, from_disk=True)

    # Reproject GOES to the MODIS grid (separate resampling for LSA vs DQF)
    #  Convert raw LSA values by applying scale offset
    lsa = lsa_data.rio.reproject_match(modis_clipped, resampling=Resampling.bilinear) / GOES_OFFSET 
    dqf = dqf_data.rio.reproject_match(modis_clipped, resampling=Resampling.nearest)
    # print("dqf", dqf)

    goes_clipped = xr.Dataset({"LSA": lsa, "DQF": dqf})
    # print("goes_clipped", goes_clipped["DQF"])

    if dqf_filter:
        # Validate and save
        if DQF_analysis(goes_clipped["DQF"]):
            goes_file_name = os.path.splitext(file_name_in)[0]
            out_name = goes_file_name + "_clipped_reprojected_new.tif"
            clipped_file_path = os.path.join(str(reprojected_clipped_goes_output_dir), out_name)
            os.makedirs(reprojected_clipped_goes_output_dir, exist_ok=True)
            goes_clipped["LSA"].rio.to_raster(clipped_file_path)
            print(f"Clipped to {clipped_file_path}")
            invalid_pixel_return = 0
        else:
            print("Too many invalid pixels")
        # return invalid_pixel_return#, rxr_in, dqf_data
    else:
        # Do analysis to see if DQF is valid to see if raster should be saved
        if DQF_analysis2(goes_clipped["DQF"]):
            # Save data as raster
            goes_file_name = os.path.splitext(file_name_in)[0]
            new_goes_file_name = goes_file_name + "_clipped_reprojected_new.tif"
            clipped_file_path = str(reprojected_clipped_goes_output_dir) + "/" + new_goes_file_name

            os.makedirs(reprojected_clipped_goes_output_dir, exist_ok=True)
            goes_clipped["LSA"].rio.to_raster(clipped_file_path)
            print(f'Clipped to {clipped_file_path}')
            invalid_pixel_return = 0
        else:
            print("Too many invalid pixels")
    return invalid_pixel_return#, rxr_in, dqf_data


##################################################################
##################################################################
# Conditions for looping through directories
def is_valid_year_dir(dir_name: str, start_year: int, end_year: int) -> bool:
    """
    Check if a directory name is a valid 4-digit year within a specified range.

    Parameters
    ----------
    dir_name : str
        Name of the directory to validate.
    start_year : int
        Start of the valid year range (inclusive).
    end_year : int
        End of the valid year range (inclusive).

    Returns
    -------
    bool
        True if dir_name is a 4-digit year within the range; False otherwise.
    """
    try:
        year = int(dir_name)
        return len(dir_name) == 4 and start_year <= year <= end_year
    except ValueError:
        return False

##################################################################
##################################################################
def is_valid_day_dir(dir_name: str, start_date: datetime, end_date: datetime, year: int) -> bool:
    """
    Check if a directory name is a valid 3-digit day-of-year and within a date range.

    Parameters
    ----------
    dir_name : str
        Name of the directory to validate (should represent day-of-year as a 3-digit string).
    start_date : datetime
        Start of the valid date range (inclusive).
    end_date : datetime
        End of the valid date range (inclusive).
    year : int
        The year to which the day-of-year should be applied.

    Returns
    -------
    bool
        True if dir_name represents a valid day-of-year within the date range; False otherwise.
    """
    try:
        day_of_year = int(dir_name)
        cur_date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
        return len(dir_name) == 3 and start_date <= cur_date <= end_date
    except ValueError:
        return False
    
##################################################################
##################################################################
def is_valid_hour_dir(dir_name: str, valid_hour_one: int, valid_hour_two: int) -> bool:
    """
    Check if a directory name is a valid 2-digit hour and matches one of two allowed values.

    Parameters
    ----------
    dir_name : str
        Name of the directory to validate (should be a 2-digit hour string).
    valid_hour_one : int
        First acceptable hour (e.g., 18).
    valid_hour_two : int
        Second acceptable hour (e.g., 21).

    Returns
    -------
    bool
        True if dir_name is a valid hour matching one of the two values; False otherwise.
    """
    try:
        hour = int(dir_name)
        return len(dir_name) == 2 and hour in {valid_hour_one, valid_hour_two}
    except ValueError:
        return False

##################################################################
##################################################################
## Clip and Reproject GOES Data - Walk through directories first
def clip_reproject_goes_data_loop_through_directories(
    base_dir_link: str,
    start_date: str,
    end_date: str,
    file_offset: int,
    boundary_in
) -> None:
    """
    Walk through a nested GOES file directory, clip and reproject files within a date and hour range.

    The expected file structure is: `base_dir/YYYY/DDD/HH/FILENAME.tif`. This function walks 
    through the directory structure, filters valid year/day/hour folders, and then processes 
    a single GOES raster file at specific time intervals by clipping and reprojecting it 
    based on a given spatial boundary.

    Parameters
    ----------
    base_dir_link : str
        Path to the root directory containing GOES data structured by year/day/hour.
    start_date : str
        Start date in "MM/DD/YYYY" format.
    end_date : str
        End date in "MM/DD/YYYY" format.
    file_offset : int
        Number of files to skip within each hour folder before selecting one for processing.
        This helps align with GOES’s 5-minute interval sampling. GOES measures every 5 minutes, 
        offset is number of files to skip to get the correct hours MODIS takes measurements.
    boundary_in : geopandas.GeoDataFrame or list-like
        Spatial boundary used for clipping. Must be in the same CRS as the reprojected output.
        This is the boundary box of East River reprojected to GOES CRS.

    Returns
    -------
    None
        The function prints summary statistics and saves clipped rasters to disk.
    """
    # set crs for the boundary box
    # boundary_in = gpd.GeoDataFrame(geometry=boundary_in, crs="EPSG:32613")
    
    file_count = 0
    valid_data_count = 0
    invalid_data_count = 0

    # Set valid MODIS measurement hours (in UTC)
    hour_one = 18
    hour_two = 19
    
    # # Retrieve valid years from dates for comparisons
    # start_year = int(start_date.split("/")[2])
    # end_year = int(end_date.split("/")[2])

    # Convert date strings to datetime objects
    start_date_obj = datetime.strptime(start_date, "%m/%d/%Y")
    end_date_obj = datetime.strptime(end_date, "%m/%d/%Y")
    start_year = start_date_obj.year
    end_year = end_date_obj.year

    # Because os.walk is topdown, at each level, processing needs to happen to trim down dirs list
    # Traverse the GOES directory tree: YYYY -> DDD -> HH -> Files
    for root, dirs, files in os.walk(base_dir_link, topdown=True):
        # Sort directories for correct time indexing
        dirs.sort()
        files.sort()

        current_dir = os.path.basename(root)
        parent_dir = os.path.basename(os.path.dirname(root))
        grandparent_dir = os.path.basename(os.path.dirname(os.path.dirname(root)))

        # YEAR LEVEL: Filter directories at the year level based on date range
        if current_dir == os.path.basename(base_dir_link):
            dirs[:] = [d for d in dirs if is_valid_year_dir(d, start_year, end_year)]

        # DAY-OF-YEAR LEVEL: Filter directories at the day level
        elif is_valid_year_dir(current_dir, start_year, end_year):
            year = int(current_dir)
            dirs[:] = [d for d in dirs if is_valid_day_dir(d, start_date_obj, end_date_obj, year)]

        # HOUR LEVEL: Filter directories at the hour level
        elif is_valid_day_dir(current_dir, start_date_obj, end_date_obj, int(parent_dir)):
            dirs[:] = [d for d in dirs if is_valid_hour_dir(d, hour_one, hour_two)]

        # FILE LEVEL: Process files in the hour directories
        elif is_valid_hour_dir(current_dir, hour_one, hour_two):
            for idx, goes_file in enumerate(files):
                if idx >= file_offset:
                    goes_file_path = os.path.join(root, goes_file)

                    try:
                        goes_ds = rxr.open_rasterio(goes_file_path)
                        if reproject_clip_and_upsample_goes_raster(goes_ds, goes_file, boundary_in) == 1:
                            invalid_data_count += 1
                            # invalid_data_count += reproject_clip_and_upsample_goes_raster(
                            #     goes_ds, goes_file, boundary_in
                            # )
                        else:
                            valid_data_count += 1
                        file_count += 1
                    except Exception as e:
                        print(f"Error processing {goes_file_path}: {e}")
                    break  # Only process one file per hour folder after offset
    
    # 330 invalid pixel when threshold was 0.1 (more than 10% invalid ws discarded): therefore 165 days of data missing
    # 1 pixel with more than 50% of data invalid
    print(f"Total invalid files due to DQF filtering: {invalid_data_count}")
    print(f"Total valid files due to DQF filtering: {valid_data_count}")
    print(f"Total file count: {file_count}")
    print("Processing complete.")


##################################################################
##################################################################
## Reproject all rasters into the reprojected raster folder
def reproject_raster(
    src_file: str,
    dst_file: str,
    dst_crs: str = 'EPSG:4326'
) -> None:
    """
    Source: https://rasterio.readthedocs.io/en/latest/topics/reproject.html
    Reproject a raster file to a new coordinate reference system (CRS).

    This function reads a source raster, computes the target transform and shape for the new CRS,
    and writes a new raster file with the reprojected data using the specified resampling method.

    Parameters
    ----------
    src_file : str
        Path to the input raster file (source).
    dst_file : str
        Path to the output raster file (destination).
    dst_crs : str, optional
        EPSG code or PROJ string for the destination CRS. Default is 'EPSG:4326'.

    Returns
    -------
    None
        The function writes the reprojected raster to disk and returns nothing.

    Notes
    -----
    This function uses `Resampling.nearest` for reprojection.
    For other options, see: https://rasterio.readthedocs.io/en/latest/topics/reproject.html
    """

    # Open the source raster
    with rio.open(src_file) as src:
        # Calculate target transform, width, and height for new CRS
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )

        # Prepare output metadata
        dst_meta = src.meta.copy()
        dst_meta.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        # Create the destination raster and reproject each band
        with rio.open(dst_file, 'w', **dst_meta) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rio.band(src, band_idx),
                    destination=rio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest
                    # resampling=Resampling.cubic
                    # resampling=Resampling.bilinear
                )

##################################################################
##################################################################
## Individual calculations
def print_rxr_metadata(rxr_ds_in: xr.Dataset) -> None:
    """
    Print metadata information for a rioxarray (xarray) Dataset.

    This utility function displays:
    - General dataset structure
    - Data variables
    - Coordinate variables
    - Global attributes

    Parameters
    ----------
    rxr_ds_in : xarray.Dataset
        A dataset opened using rioxarray or xarray containing raster data.

    Returns
    -------
    None
        Prints the dataset structure and metadata to the console.
    """
    print("=== Metadata Summary ===\n")

    print(">> Dataset Overview:")
    print(rxr_ds_in)

    print("\n>> Data Variables:")
    print(rxr_ds_in.data_vars)

    print("\n>> Data Coordinates:")
    print(rxr_ds_in.coords)

    print("\n>> Global Attributes:")
    print(rxr_ds_in.attrs)


##################################################################
##################################################################
## Plot Reprojected Plots
def plot_band_data(file_path_in: str) -> None:
    """
    Read and display the first band of a raster file as a 2D image using xarray and matplotlib.

    This function:
    - Opens a raster file using Rasterio
    - Reads the first band of data (1-based indexing)
    - Converts it into an xarray.DataArray
    - Attaches spatial metadata (transform and CRS)
    - Plots the raster band using matplotlib

    Parameters
    ----------
    file_path_in : str
        Path to the raster file to be visualized.

    Returns
    -------
    None
        The function displays a plot of the first raster band and does not return a value.

    Notes
    -----
    - Assumes that the raster contains at least one band.
    - Uses masked=True to handle no-data values.
    """
    with rio.open(file_path_in, masked=True) as src:
        # Read the first band (Rasterio uses 1-based indexing). rasterio uses 1-based indexing for bands
        band1 = src.read(1)

        # Get spatial metadata. Get the transform and CRS from the source dataset
        transform = src.transform
        crs = src.crs

        # Wrap the raster band in an xarray DataArray
        band1_data = xr.DataArray(
            band1,
            dims=("y", "x"),
            coords={
                "y": range(band1.shape[0]),
                "x": range(band1.shape[1])
            },
            name="Band_1"
        )

        # Attach transform and CRS using rioxarray
        band1_data.rio.write_transform(transform, inplace=True)
        band1_data.rio.write_crs(crs, inplace=True)

        # Plot the raster band
        fig, ax = plt.subplots(figsize=(8, 6))
        band1_data.plot(ax=ax)
        ax.set_title("Raster Band 1")
        plt.tight_layout()
        plt.show()

##################################################################
##################################################################
## Plot GOES and MODIS Data
def extract_date_from_modis_filename(filename: str) -> datetime:
    """
    Extract the acquisition date from a MODIS filename using its Julian date format.

    MODIS filenames often start with a 7-digit string representing the date in `YYYYJJJ` format,
    where `JJJ` is the Julian day of the year.

    Parameters
    ----------
    filename : str
        MODIS filename containing the Julian date as a prefix (e.g., '2022184...').

    Returns
    -------
    datetime
        A datetime object representing the extracted date.

    Raises
    ------
    ValueError
        If the Julian date cannot be parsed correctly.
    """
    julian_date_str = filename.split("_")[0]
    return datetime.strptime(julian_date_str, "%Y%j")

##################################################################
##################################################################
def extract_datetime_from_goes_filename(filename: str) -> Optional[str]:
    """
    Extract the 14-digit GOES acquisition datetime string from the filename.

    GOES filenames typically include a substring of the form `_sYYYYJJJHHMMSS_` where:
    - `YYYY` is the year,
    - `JJJ` is the Julian day,
    - `HHMMSS` is the hour, minute, and second.

    Parameters
    ----------
    filename : str
        GOES filename containing the `_sYYYYJJJHHMMSS_` pattern.

    Returns
    -------
    str or None
        The extracted datetime string (e.g., '2022184123456') if matched; otherwise, None.
    """
    match = re.search(r'_s(\d{14})_', filename)
    return match.group(1) if match else None

##################################################################
##################################################################
def visualize_all_tif_data(goes_dir_in: str, modis_dir_in: str) -> Tuple[List[datetime], List[datetime]]:
    """
    Visualize and compare GOES and MODIS Blue Sky Albedo .tif datasets over time.

    The function:
    - Iterates through GOES and MODIS raster files
    - Matches them by date (using Julian day for MODIS and timestamp for GOES)
    - Displays paired plots of GOES and MODIS albedo
    - Tracks and reports dates with and without valid data

    Parameters
    ----------
    goes_dir_in : str
        Path to the directory containing GOES .tif files.
    modis_dir_in : str
        Path to the directory containing MODIS .tif files.

    Returns
    -------
    Tuple[List[datetime], List[datetime]]
        A tuple containing:
        - List of dates with no valid GOES data
        - List of dates with valid GOES data
    """
    goes_files = sorted(Path(goes_dir_in).glob("*.tif"), key=lambda x: extract_datetime_from_goes_filename(x.name))
    modis_files = sorted(Path(modis_dir_in).glob("*.tif"), key=lambda x: extract_date_from_modis_filename(x.name))

    previous_goes_date = None
    total_no_data_dates = 0
    total_data_dates = 0
    dates_with_no_data = []
    dates_with_data = []

    for goes_file in goes_files:
        try:
            # Extract date from GOES filename
            goes_timestamp_str = extract_datetime_from_goes_filename(goes_file.name)
            actual_date = datetime.strptime(goes_timestamp_str[:7], "%Y%j")
            goes_display_hour = f"{goes_timestamp_str[7:9]}:{goes_timestamp_str[9:11]}"

            # Track missing dates (gaps)
            if previous_goes_date:
                delta = actual_date - previous_goes_date
                if delta.days > 1:
                    gap_dates = [previous_goes_date + timedelta(days=i) for i in range(1, delta.days)]
                    dates_with_no_data.extend(gap_dates)
                    total_no_data_dates += len(gap_dates)
            previous_goes_date = actual_date

            goes_date_str = actual_date.strftime('%m-%d-%Y')

            # Match MODIS file to current GOES date
            modis_file_path = None
            modis_display_hour = "Local Solar Noon: 18:30"
            modis_date_str = "Unknown"
            for modis_file in modis_files:
                modis_date = extract_date_from_modis_filename(modis_file.name)
                if modis_date.strftime('%Y%j') == actual_date.strftime('%Y%j'):
                    modis_file_path = modis_file
                    modis_date_str = modis_date.strftime('%m-%d-%Y')
                    break

            # Set up plot
            fig, (ax1, ax2) = plt.subplots(2, figsize=(15, 7))
            fig.subplots_adjust(hspace=0.4)

            # Read and process GOES raster
            with rxr.open_rasterio(goes_file) as goes:
                goes_data = goes.sel(band=1)
                masked_goes = goes_data.where(goes_data > 5)

                valid_pixel_count = np.count_nonzero(~np.isnan(masked_goes))

                # Determine data validity based on pixel threshold
                if valid_pixel_count > 5:
                    dates_with_no_data.append(actual_date)
                    total_no_data_dates += 1
                    print(f"[!] Invalid data for {goes_date_str}")
                else:
                    dates_with_data.append(actual_date)
                    total_data_dates += 1

                # GOES plot
                goes_plot = goes_data.plot.imshow(ax=ax1, cmap='viridis', add_colorbar=False)
                ax1.set_title(f"GOES Blue Sky Albedo - {goes_date_str} at {goes_display_hour}")
                ax1.set_xlabel("Longitude")
                ax1.set_ylabel("Latitude")
                fig.colorbar(goes_plot, ax=ax1, orientation='vertical', fraction=0.036, pad=0.04)

            # Read and plot MODIS data if available
            if modis_file_path:
                with rxr.open_rasterio(modis_file_path) as modis:
                    modis_data = modis.sel(band=1)
                    modis_plot = modis_data.plot.imshow(ax=ax2, cmap='viridis', add_colorbar=False)
                    ax2.set_title(f"MODIS Blue Sky Albedo - {modis_date_str} at {modis_display_hour}")
                    ax2.set_xlabel("Longitude")
                    ax2.set_ylabel("Latitude")
                    fig.colorbar(modis_plot, ax=ax2, orientation='vertical', fraction=0.036, pad=0.04)
            else:
                ax2.set_title(f"No matching MODIS data for {goes_date_str}")
                ax2.axis('off')

            plt.show()

        except Exception as e:
            print(f"Error processing {goes_file.name}: {e}")
            continue

    print(f"\nSummary:")
    print(f"Total dates with valid GOES data: {total_data_dates}")
    print(f"Total dates with no GOES data: {total_no_data_dates}")
    
    return dates_with_no_data, dates_with_data

##################################################################
##################################################################
## Filter out MODIS images if there is more than 60% of data that is null
# Match GOES Image with MODIS Image using a NaN Mask and interpolating NaN Values the same way MODIS is interpolated
def mask_goes_to_match_modis(
    goes_dir_in: str,
    modis_dir_in: str,
    goes_out_dir: str
) -> None:
    """
    Mask GOES raster data using NaN values from corresponding MODIS data for the same date.

    For each MODIS file (1 per day), there may be 2 corresponding GOES files (e.g., at 18:30 and 19:30 UTC).
    This function identifies those GOES files, and replaces GOES pixel values with NaN wherever the
    corresponding MODIS raster has NaN (missing) values. The masked GOES rasters are saved to disk.

    Parameters
    ----------
    goes_dir_in : str
        Directory containing GOES raster (.tif) files.
    modis_dir_in : str
        Directory containing MODIS raster (.tif) files used as the NaN mask.
    goes_out_dir : str
        Directory where the masked GOES raster files will be saved.

    Returns
    -------
    None
        The function saves the masked GOES files to disk and does not return anything.
    """
    # Load and sort input file lists
    goes_files = sorted(Path(goes_dir_in).iterdir(), key=lambda x: extract_datetime_from_goes_filename(x.name))
    modis_files = sorted(Path(modis_dir_in).iterdir(), key=lambda x: extract_date_from_modis_filename(x.name))

    # Ensure the output directory exists
    os.makedirs(goes_out_dir, exist_ok=True)

    # Iterate through each MODIS file and mask matching GOES files
    for modis_file in modis_files:
        cur_date_str = modis_file.name.split("_")[0]  # Format: YYYYJJJ

        masked_goes_count = 0
        for goes_file in goes_files:
            if cur_date_str in goes_file.name:
                # Load MODIS and GOES data
                modis_bsa = rxr.open_rasterio(modis_file, masked=True).sel(band=1)
                goes_rxr = rxr.open_rasterio(goes_file, masked=True)

                # Create a NaN mask from the MODIS raster
                modis_nan_mask = modis_bsa.isnull()

                # Apply the mask: wherever MODIS is NaN, set GOES to NaN
                goes_rxr_masked = goes_rxr.where(~modis_nan_mask, np.nan)

                # Save the masked GOES raster to output directory
                output_path = os.path.join(goes_out_dir, goes_file.name)
                goes_rxr_masked.rio.to_raster(output_path)
                # print(f"Saved masked GOES file to: {output_path}")

                masked_goes_count += 1

            # Each MODIS date is expected to have at most two GOES files
            if masked_goes_count >= 2:
                break

            
##################################################################
##################################################################
def _ensure_da_with_band(obj: xr.Dataset | xr.DataArray) -> xr.DataArray:
    """
    Ensure an xarray object has a leading ``band`` dimension and return it as a DataArray.

    This helper normalizes inputs so downstream raster I/O can uniformly treat data as
    multiband. Behavior:
      - If ``obj`` is an ``xarray.Dataset``, it is converted to a stacked
        ``xarray.DataArray`` using ``to_array(dim="band")`` with shape ``(band, y, x)``.
      - If ``obj`` is an ``xarray.DataArray`` **without** a ``"band"`` dimension,
        a singleton band dimension is added (``band=[1]``).
      - If ``obj`` is an ``xarray.DataArray`` **with** a ``"band"`` dimension, it is
        returned unchanged.

    Parameters
    ----------
    obj : xr.Dataset | xr.DataArray
        The input dataset/array to normalize.

    Returns
    -------
    xr.DataArray
        A DataArray that always includes a ``"band"`` dimension (1..N). For single-band
        inputs lacking a band dimension, the resulting shape will be ``(band=1, y, x)``.

    Notes
    -----
    - This function does **not** modify coordinate reference system (CRS) or affine
      transform metadata; those should already be attached (e.g., via ``rioxarray``).
    - The function does not copy data unless necessary (e.g., stacking a Dataset).
    """
    if isinstance(obj, xr.Dataset):
        da = obj.to_array(dim="band")  # (band, y, x)
    else:
        da = obj
        if "band" not in da.dims:
            # add a singleton band dim so we can write multiband safely
            da = da.expand_dims(band=[1])
    return da

##################################################################
##################################################################
def _write_geotiff_with_nodata(da: xr.DataArray, out_path: Path, nodata=NODATA):
    """
    Write a GeoTIFF to ``out_path`` with an explicit NoData value encoded.

    The input DataArray is coerced to a floating dtype if necessary (``float32``) so
    that NoData can be reliably represented, then the NoData value is written into the
    raster metadata using ``rioxarray`` before saving.

    Parameters
    ----------
    da : xr.DataArray
        Raster data to write. Must include spatial metadata compatible with ``rioxarray``
        (e.g., CRS via ``.rio.crs`` and transform via ``.rio.transform``).
    out_path : Path
        Destination file path for the GeoTIFF.
    nodata :
        The NoData value to encode in the output. Defaults to the module-level ``NODATA``.

    Returns
    -------
    None
        Writes the GeoTIFF to disk; no value is returned.

    Notes
    -----
    - If ``da.dtype`` is not floating, it is cast to ``float32`` prior to writing.
    - ``encoded=True`` ensures the NoData flag is stored in the file's metadata.
    - Any existing NoData setting on ``da`` is overwritten by the provided ``nodata``.
    """
    # ensure float and consistent nodata
    if not np.issubdtype(da.dtype, np.floating):
        da = da.astype("float32")
    da.rio.write_nodata(nodata, encoded=True, inplace=True)
    da.rio.to_raster(out_path)


##################################################################
##################################################################
def _open_raster_strict(
    path: Path,
    prefer_subdataset_regex: Optional[str] = None,   # e.g., r"AOD_550.*Combined"
) -> xr.DataArray:
    """
    Open a raster file and guarantee an xr.DataArray with a 'band' dimension.

    Handles:
      - GeoTIFF -> DataArray (adds singleton band if absent)
      - NetCDF -> Dataset -> to_array('band')
      - HDF/others -> picks a subdataset (by regex if provided, else first)
      - List result from rioxarray.open_rasterio -> stack along 'band'
    """
    # If the container has subdatasets (e.g., HDF4/5), select one explicitly
    with rio.Env():
        with rio.open(path) as src:
            subdatasets = list(src.subdatasets)

    if subdatasets:
        # choose subdataset
        sd_to_open = None
        if prefer_subdataset_regex:
            pat = re.compile(prefer_subdataset_regex)
            for sd in subdatasets:
                if pat.search(sd):
                    sd_to_open = sd
                    break
        if sd_to_open is None:
            sd_to_open = subdatasets[0]  # fall back to the first

        obj = rxr.open_rasterio(sd_to_open, masked=True)

    else:
        obj = rxr.open_rasterio(path, masked=True)

    # Normalize to DataArray with 'band'
    if isinstance(obj, list):
        # rioxarray can return a list if you pass a sequence of paths or certain multi-sds cases
        if not obj:
            raise ValueError(f"{path.name}: empty list returned by open_rasterio.")
        if not all(isinstance(x, xr.DataArray) for x in obj):
            raise TypeError(f"{path.name}: mixed types in list from open_rasterio.")
        da = xr.concat(obj, dim="band")
        da = da.assign_coords(band=np.arange(1, da.sizes["band"] + 1))
        return da

    if isinstance(obj, xr.DataArray):
        if "band" not in obj.dims:
            obj = obj.expand_dims(band=[1])
        return obj

    if isinstance(obj, xr.Dataset):
        da = obj.to_array(dim="band")
        da = da.assign_coords(band=np.arange(1, da.sizes["band"] + 1))
        return da

    raise TypeError(f"{path.name}: unsupported type from open_rasterio: {type(obj)}")

    
##################################################################
##################################################################
## Clip AOD data to East River Shapefile
def clip_modis_data(
    data_dir_in: str,
    output_clip_dir: str,
    desired_bands_in: list,
    boundary_box_in,
    aod_format: bool
) -> None:
    """
    Clip MODIS raster data to a specified boundary and save selected bands to disk.

    This function reads MODIS GeoTIFF files, optionally processes AOD-format rasters 
    (e.g., by selecting the first orbit and resampling), extracts selected bands, clips 
    them to a spatial boundary, and saves the result as a new raster.

    Parameters
    ----------
    data_dir_in : str
        Directory path containing input MODIS raster files.
    output_clip_dir : str
        Path to the output directory where clipped rasters will be saved.
    desired_bands_in : list
        Indices or band selectors to extract from the MODIS raster (e.g., [0] or [1, 2]).
    boundary_box_in : GeoDataFrame or list-like
        Polygon or shapefile geometry used for spatial clipping.
    aod_format : bool
        Indicates if the input data is in AOD (Aerosol Optical Depth) format. If True,
        the function selects the first orbit and first band and upsamples the raster
        using bilinear interpolation.

    Returns
    -------
    None
        The function saves the clipped raster(s) to disk and prints confirmation messages.
    """
    
    NODATA_FLOAT = -9999.0  # shared nodata for all bands (float32)

    data_dir = Path(data_dir_in)
    clipped_dir = Path(base_modis_path) / output_clip_dir
    clipped_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over all files in the input data directory
    for file_path in data_dir.iterdir():
        if not file_path.is_file():
            continue

        # Open MODIS raster with masking for nodata
        modis_file = rxr.open_rasterio(file_path, masked=True)

        # Handle AOD-format files: select first orbit, first band, and resample
        if aod_format:
            # AOD has another dimension based on orbit overpasses
            # Information is stored Orbit_amount and Orbit_time_stamp
            # Get the first overpass
            modis_file = modis_file[0].isel(band=0)  # Select first overpass and band
            modis_file = resample_raster_bilinear(modis_file, UPSCALE_FACTOR)

        # Select specified bands from raster
        modis_selected = modis_file[desired_bands_in]

        # Clip raster to provided boundary
        modis_clipped = modis_selected.rio.clip(
            boundary_box_in,
            all_touched=True,
            from_disk=True,
            drop=True
        ).squeeze()

        # Build output path
        new_file_name = f"{file_path.stem}_clipped_new.tif"
        output_path = clipped_dir / new_file_name

        # ---- OPTION A: enforce single nodata and dtype; write one multiband GeoTIFF ----
        if isinstance(modis_clipped, xr.Dataset):
            # Convert all vars to float32, stack to (band, y, x), set one nodata, write
            da_out = modis_clipped.astype("float32").to_array(dim="band")
            da_out.rio.write_nodata(NODATA_FLOAT, encoded=True, inplace=True)
            da_out.rio.to_raster(output_path)
        else:
            # Single-band DataArray
            da_out = modis_clipped.astype("float32")
            da_out.rio.write_nodata(NODATA_FLOAT, encoded=True, inplace=True)
            da_out.rio.to_raster(output_path)

        print(f"Clipped and saved to: {output_path}")
##################################################################
##################################################################


##################################################################
##################################################################


##################################################################
##################################################################

