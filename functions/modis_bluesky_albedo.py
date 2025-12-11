# Packages
import math
import time
import numpy as np
import pandas as pd
import xarray as xr
import rasterio as rio
from io import StringIO
import rioxarray as rxr
import geopandas as gpd
from pathlib import Path
from netCDF4 import date2num
from rasterio.plot import show
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, Tuple, Union, Sequence

##################################################################
##################################################################
## Path to data files
modis_albedo_data_dir = "/bsuhome/tnde/scratch/felix/modis/reprojected_colorado_modis_data_lat_lon_new"
aod_data_dir = "/bsuhome/tnde/scratch/felix/modis/reprojected_aod_data_new"
aod_lookup_table_file = "/bsuhome/tnde/scratch/felix/CERES/CERES_SYN1deg-Day_Terra-Aqua-MODIS_Ed4.1_Subset_20000301-20240731.nc"
diffuse_skylight_ratio_lookup = "sw_lut.csv"
ex_format_tif = "/bsuhome/tnde/scratch/felix/modis/reprojected_colorado_modis_data_lat_lon_new/MCD43A3.A2022314.h09v05.061.2022323034956_clipped_reprojected_new"

# The offset 0.001 is used because the MODIS albedo products store values as scaled integers, not as true reflectance/albedo in the range 0–1.
AOD_OFFSET = 0.001
ALBEDO_OFFSET = 0.001 
SAIL_LOCATION = (39, -106)

# scp -r H:/airborne_lidar/SouthernIdaho2018_21_tile_index/lasheight/ tnde@borah-login.boisestate.edu:/bsuhome/tnde/scratch/carbon_estimation/airborne_lidar/
##################################################################
##################################################################
def get_albedo_values(dir_in: str, black_sky_bool: bool) -> Dict[str, 'xarray.DataArray']:
    """
    Extract albedo values from raster files and return them as a dictionary keyed by date.
    Hence, date as the key and value albedo as the value

    Each raster file is assumed to contain multiple bands:
    - Band 1: Black-sky albedo
    - Band 2: White-sky albedo

    The function reads either Band 1 or Band 2 based on the `black_sky_bool` flag and
    stores the result in a dictionary with the corresponding date as the key.

    Parameters
    ----------
    dir_in : str
        Directory containing albedo raster files (.tif).
    black_sky_bool : bool
        If True, extract black-sky albedo (band 1); otherwise, extract white-sky albedo (band 2).

    Returns
    -------
    Dict[str, xarray.DataArray]
        Dictionary mapping each date string to its corresponding albedo DataArray.
        The date is extracted from the filename by splitting on '.' and slicing from index 1.
    """
    albedo_dict = {}
    dir_path = Path(dir_in)

    # Get all files and sort by filename
    sorted_files = sorted(dir_path.iterdir(), key=lambda x: x.name)

    for file in sorted_files:
        if file.is_file():
            # Extract date from filename (assumes format: something.DYYYYJJJ.something.tif)
            try:
                date = file.name.split(".")[1][1:]  # Example: "D2022180" -> "2022180"
            except IndexError:
                print(f"Filename format unexpected: {file.name}")
                continue

            # Open raster file and select the appropriate band
            raster = rxr.open_rasterio(file, masked=True)
            albedo_band = raster.sel(band=1) if black_sky_bool else raster.sel(band=2)

            # Store in dictionary with date as key
            albedo_dict[date] = albedo_band

    return albedo_dict

##################################################################
##################################################################
def get_aod_values(dir_in: str) -> Dict[str, xr.DataArray]:
    """
    Extract Aerosol Optical Depth (AOD) values from raster files and return them in a dictionary keyed by date.
    Hence, a dictionary with key as the date and value as the AOD value

    Each raster file is assumed to contain AOD data in Band 1.
    The date is extracted from the filename using a split by "." and slicing the second element
    (e.g., filename format: something.D2022180.something.tif -> date: '2022180').

    Parameters
    ----------
    dir_in : str
        Directory containing AOD raster files (.tif).

    Returns
    -------
    Dict[str, xarray.DataArray]
        Dictionary where keys are date strings (e.g., '2022180') and values are AOD data arrays.
    """
    aod_dict = {}
    dir_path = Path(dir_in)

    # Sort files by name to ensure consistent order
    sorted_files = sorted(dir_path.iterdir(), key=lambda x: x.name)

    for file in sorted_files:
        if file.is_file():
            try:
                # Extract date string from filename (e.g., D2022180 -> 2022180)
                date = file.name.split(".")[1][1:]
            except IndexError:
                print(f"[Warning] Skipping file with unexpected name format: {file.name}")
                continue

            # Open raster file and select Band 1 as AOD
            raster = rxr.open_rasterio(file, masked=True)
            aod_data = raster.sel(band=1)

            # Store the AOD DataArray with the extracted date
            aod_dict[date] = aod_data

    return aod_dict

##################################################################
##################################################################
## Get AOD Values from CERES Time Series Data
def get_aod_static_data(
    dir_in: str,
    date: str,
    location: Tuple[float, float] = SAIL_LOCATION
) -> np.ndarray:
    """
    Retrieve static AOD (Aerosol Optical Depth) data for a given location and date.

    The function opens a NetCDF dataset containing daily AOD values (`ini_aod55_daily`)
    and extracts the value nearest to the specified geographic location and time.

    Parameters
    ----------
    dir_in : str
        Path to the input NetCDF file containing AOD data.
    date : str
        Target date in the format 'YYYYJJJ' (e.g., '2022180' for June 29, 2022).
    location : Tuple[float, float], optional
        Geographic location as (latitude, longitude). Default is `SAIL_LOCATION`.

    Returns
    -------
    np.ndarray
        The AOD value(s) at the nearest grid point to the given location and date.

    Notes
    -----
    - Assumes the NetCDF variable is named 'ini_aod55_daily'.
    - Uses nearest-neighbor search in both space and time.
    - Assumes daily temporal resolution and 2D spatial coverage.
    """
    # Convert input date string (e.g., "2022180") to datetime object
    target_date = datetime.strptime(date, "%Y%j")

    # Open the NetCDF dataset and extract nearest AOD value
    with xr.open_dataset(dir_in) as aod_ds:
        aod_value = aod_ds["ini_aod55_daily"].sel(
            lat=location[0],
            lon=location[1],
            time=target_date,
            method="nearest"
        ).values

    return aod_value

##################################################################
##################################################################
## Calculate Solar Zenith Angle
def get_solar_zenith(latitudes: Union[float, Sequence[float]], date: str) -> np.ndarray:
    """
    Estimate the solar zenith angle for given latitudes and a specific date.

    This approximation is based on a simplified model using the Earth's declination angle.
    The function assumes local solar noon, meaning the sun is at its highest point in the sky.

    Parameters
    ----------
    latitudes : float or Sequence[float]
        One or more latitude values (in degrees, positive for northern hemisphere).
    date : str
        Date in 'YYYYDDD' format where DDD is the Julian day (e.g., '2022180' for June 29, 2022).

    Returns
    -------
    np.ndarray
        Array of solar zenith angles (in degrees) for each latitude at local solar noon.

    Notes
    -----
    - The solar zenith angle is defined as the angle between the sun and the vertical direction.
    - This function assumes solar noon and does not account for longitude or time zone.
    - Results are approximate and best suited for large-scale or conceptual models.
    """
    # Ensure latitudes are a NumPy array for vectorized operations
    latitudes = np.array(latitudes, dtype=np.float64)

    # Parse the year and day-of-year from the input date string
    year = int(date[:4])
    day_of_year = int(date[4:])

    # Estimate solar declination angle (in degrees)
    # Approximation formula: decl = -23.45° * cos(360/365 * (day_of_year + 10))
    decl = -23.45 * np.cos(np.radians((360 / 365) * (day_of_year + 10)))

    # Calculate solar altitude angle at solar noon
    solar_altitude = 90 - np.abs(latitudes - decl)

    # Compute solar zenith angle as the complement of solar altitude
    solar_zenith = 90 - solar_altitude

    return solar_zenith

##################################################################
##################################################################
def convert_last_digit_of_float(value: float, decimals: int = 2) -> str:
    """
    Rounds the decimal portion of the float to be an even decimal (rounded down)
    Used to index the lookup table for the diffuse skylight ratiio
    
    Parameters:
        value (string): The string of the float value
        decimals (int): The number of decimal places to round to.
        
    Returns:
        str: The rounded float value with an even last digit converted to float.
    """     
    float_value = round(float(value), 2)
    
    if float_value == 0:
        return "0"
    
    # Multiply the value by 100, take the floor to round down
    floored_value = math.floor(float_value * 100)
    
    # Extract the last digit
    last_digit = floored_value % 10
    
    # Check if the last digit is even
    if last_digit % 2 != 0:
        # If the last digit is odd, subtract 1 to make it even
        floored_value -= 1
    
    # Divide by 100 to get the final result with two decimal places
    even_value = floored_value / 100.0
    
    if even_value == 0:
        return "0"
    
    # Convert the result back to a string in the format 0.nn
    return f"{even_value:.2f}" #str(even_value)

##################################################################
##################################################################
# This function is using the AOD values from MCD Array
def calculate_blue_sky_albedo(
    black_sky_albedo_arr_in: Dict[str, xr.DataArray],
    white_sky_albedo_arr_in: Dict[str, xr.DataArray],
    lookup_table_in,
    black_sky_tif_in: str
) -> Dict[str, xr.DataArray]:
    """
    Calculate blue sky albedo by combining black-sky and white-sky albedo using
    diffuse skylight ratios derived from a lookup table indexed by solar zenith angle and AOD.

    Parameters
    ----------
    black_sky_albedo_arr_in : Dict[str, xr.DataArray]
        Dictionary of black sky albedo rasters indexed by date (format: 'YYYYDDD').
    white_sky_albedo_arr_in : Dict[str, xr.DataArray]
        Dictionary of white sky albedo rasters indexed by date.
    lookup_table_in : pandas.DataFrame
        Lookup table of diffuse skylight ratios with solar zenith angles as rows and
        AOD values (as strings) as columns.
    black_sky_tif_in : str
        Path to a sample black-sky GeoTIFF file for reading the spatial transform.

    Returns
    -------
    Dict[str, xr.DataArray]
        Dictionary of computed blue sky albedo rasters (as xarray.DataArray), indexed by date.
    """
    blue_sky_albedo_arr = {}

    # Retrieve geospatial transform from a reference raster
    with rio.open(black_sky_tif_in) as src:
        transform = src.transform

    # Loop through all black-sky albedo entries
    for date, bsa_arr in black_sky_albedo_arr_in.items():
        # Retrieve static AOD value for this date
        aod_value = get_aod_static_data(aod_lookup_table_file, date)
        converted_aod_value = convert_last_digit_of_float(aod_value)

        # Ensure proper format for zero AOD edge case
        if float(converted_aod_value) == 0.0:
            converted_aod_value = f"{float(converted_aod_value):.2f}"

        # Extract data arrays and coordinates
        bsa_values = bsa_arr.values
        wsa_values = white_sky_albedo_arr_in[date].values

        # Create a NaN-filled array for blue sky albedo
        blsa_data = np.full_like(bsa_values, np.nan)

        # Build meshgrid of y/x pixel centers
        y_coords, x_coords = np.meshgrid(bsa_arr['y'].values, bsa_arr['x'].values, indexing='ij')
        _, lat = rio.transform.xy(transform, y_coords, x_coords, offset='center')

        # Compute solar zenith angles and round them for table lookup
        solar_zenith_angles = get_solar_zenith(lat, date)
        rounded_sza = np.round(solar_zenith_angles).astype(int)

        # Identify valid pixels (non-NaN in both albedo arrays)
        valid_mask = ~np.isnan(bsa_values) & ~np.isnan(wsa_values)

        # Retrieve lookup values for valid solar zenith angles and AOD
        valid_sza = rounded_sza[valid_mask]
        lookup_results = lookup_table_in.loc[valid_sza, converted_aod_value].values

        # Initialize and assign diffuse skylight ratios
        diffuse_ratios = np.full_like(bsa_values, np.nan)
        diffuse_ratios[valid_mask] = lookup_results

        # Compute blue sky albedo using weighted average
        blsa_data[valid_mask] = (
            (1 - diffuse_ratios[valid_mask]) * (bsa_values[valid_mask] * ALBEDO_OFFSET) +
            diffuse_ratios[valid_mask] * (wsa_values[valid_mask] * ALBEDO_OFFSET)
        )

        # Store result as xarray.DataArray
        blue_sky_albedo_arr[date] = xr.DataArray(
            blsa_data,
            coords=bsa_arr.coords,
            dims=bsa_arr.dims,
            attrs=bsa_arr.attrs
        )

    return blue_sky_albedo_arr

##################################################################
##################################################################
## Write Blue Sky Albedo Data to Raster
def write_to_raster(
    blue_sky_val: xr.DataArray,
    format_tif: str,
    output_path_tif: str
) -> None:
    """
    Write a blue sky albedo DataArray to a GeoTIFF file using the spatial metadata
    from a reference raster file.

    This function ensures that the output raster:
    - Matches the coordinate reference system (CRS),
    - Uses the same affine transform,
    - Inherits other formatting properties (e.g., data type, compression) from the reference raster.

    Parameters
    ----------
    blue_sky_val : xarray.DataArray
        The computed blue sky albedo data to be written to a raster file.
    format_tif : str
        Path to the reference GeoTIFF file (e.g., black-sky albedo raster) for metadata extraction.
    output_path_tif : str
        Path where the output raster will be saved.

    Returns
    -------
    None
        The function writes the file to disk and returns nothing.
    """
    # Extract CRS, transform, and metadata from the reference raster
    # Grab existing metadata and crs to format blue sky albedo exactly the same as black sky albedo
    with rio.open(format_tif) as src:
        metadata = src.profile
        crs = src.crs
        transform = src.transform

    # Write CRS and transform to the DataArray
    rioxarray_data = blue_sky_val.rio.write_crs(crs)
    rioxarray_data = rioxarray_data.rio.write_transform(transform)

    # Save the DataArray to a GeoTIFF using the reference metadata
    rioxarray_data.rio.to_raster(output_path_tif, **metadata)

##################################################################
##################################################################

