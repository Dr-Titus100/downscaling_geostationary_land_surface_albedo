"""
Sentinel-2 20 m Snow-Albedo Workflow with Terrain Correction + BRDF Fusion
----------------------------------------------------------------------------
What this script does
1) Reads an AOI from a shapefile and projects processing to EPSG:32613 (UTM 13N).
2) Queries Sentinel‑2 L2A (surface reflectance) over 2021‑09‑01 .. 2023‑06‑15 via Planetary Computer STAC.
3) Builds 20 m stacks of reflectance + SCL, masks clouds/shadows but KEEPS snow (SCL=11).
4) Loads Copernicus DEM (GLO‑30), computes slope/aspect, and performs SCS+C terrain illumination correction.
5) Computes broadband albedo (Lambertian/HDRF NTB) at 20 m per date and exports GeoTIFFs.
6) BRDF Fusion (MODIS MCD43A3 v061):
   * pulls daily BSA/WSA shortwave albedo (500 m),
   * upsamples to 20 m and spatially sharpens using a ratio of S2 20 m HDRF vs its 500 m box‑averaged version,
   * exports fused 20 m BSA and WSA per date.

Outputs
- s2_albedo20m_topocorr_keepSnow_YYYY-MM-DD.tif        (S2 HDRF NTB, terrain‑corrected)
- s2_albedo20m_topocorr_snowOnly_YYYY-MM-DD.tif        (S2 HDRF NTB, snow mask applied)
- s2_fused_BSA20m_YYYY-MM-DD.tif                        (BRDF‑normalized, fused)
- s2_fused_WSA20m_YYYY-MM-DD.tif                        (BRDF‑normalized, fused)
- s2_fused_BSA20m_snowOnly_YYYY-MM-DD.tif               (BRDF‑normalized, fused, snow mask applied)
- s2_fused_WSA20m_snowOnly_YYYY-MM-DD.tif               (BRDF‑normalized, fused, snow mask applied)
- Optional median composites of the above

Dependencies (install once)
    pip install geopandas pystac-client planetary-computer stackstac xarray rioxarray rasterio shapely pyproj dask[distributed] scipy

Notes
- Assumes Planetary Computer public access; no Azure auth is required for public data.
- NTB albedo is an HDRF approximation (Lambertian). BRDF fusion yields BSA/WSA structure at 20 m.
- If you want stricter cloud buffering, see the optional morphological dilation near the mask step.
"""

import re
import stackstac
import numpy as np
import pandas as pd
import xarray as xr
import rasterio as rio
from pyproj import CRS
import rioxarray as rxr
import geopandas as gpd
from pathlib import Path
import earthaccess as ea
import datetime as datetime
from datetime import timedelta
import planetary_computer as pc
from pystac_client import Client
from shapely.geometry import mapping
from rasterio.enums import Resampling
from pystac_client import Client as StacClient

# -------------------- User Inputs --------------------
shapefile_path = "/bsuhome/tnde/scratch/felix/modis/East_River_SHP/ER_bbox.shp"
DATE_RANGE = "2021-08-23/2021-12-31"    # inclusive range # "2021-08-23/2023-06-15" "2021-09-01/2023-06-15"
EPSG_UTM = 32613                        # UTM zone for the AOI (provided)
CLOUD_MAX = 20                          # cloud% pre-filter (we will also SCL-mask)
OUT_DIR = Path("/bsuhome/tnde/scratch/felix/Sentinel-2/s2_albedo_20m_topocorr_fused"); OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR2 = Path("/bsuhome/tnde/scratch/felix/Sentinel-2"); OUT_DIR.mkdir(parents=True, exist_ok=True)

# NTB weights (Lambertian/HDRF) across S2 bands
NTB_WEIGHTS = {
    "B02":0.1836, "B03":0.1759, "B04":0.1456, "B05":0.1347,
    "B06":0.1233, "B07":0.1134, "B08":0.1001, "B11":0.0231, "B12":0.0003
}

# S2 bands to read (20 m target). B05/B06/B07 native 20 m, others resampled.
REFL_BANDS = ["B02","B03","B04","B05","B06","B07","B08","B11","B12"]
# REFL_BANDS = ["B02","B03","B04","B08","B11","B12"]  # reflectance-only
SCL_BAND   = ["SCL"]  # scene classification

# SCL classes to EXCLUDE (keep snow=11). Exclude water too (6) by default.
BAD_SCL = [0, 1, 2, 3, 6, 8, 9, 10]  # no-data, saturated/defective, dark, water, cloud shadow, clouds, cirrus

# # Sentinel-2 Scene Classification  Layer (SCL)
# # The classification scheme includes 12 distinct classes: 
# Class 0 - No data: Pixels where no data is available.
# Class 1 - Saturated or Defective: Pixels with corrupted or unusable data.
# Class 2 - Dark area pixels: Pixels that are unusually dark, possibly due to shadows or specific surface conditions.
# Class 3 - Cloud shadows: Pixels directly underneath clouds.
# Class 4 - Vegetation: Pixels covered by plants and trees.
# Class 5 - Bare soils: Pixels showing exposed soil or deserts (non vegetated areas).
# Class 6 - Water: Pixels representing lakes, rivers, or oceans.
# Class 7 - Cloud low probability / Unclassified: Areas suspected of having clouds or where classification was uncertain.
# Class 8 - Cloud medium probability: Pixels with a moderate likelihood of being clouds.
# Class 9 - Cloud high probability: Pixels with a high certainty of being clouds.
# Class 10 - Thin cirrus: High-altitude, thin clouds that are nearly transparent.
# Class 11 - Snow or ice: Pixels covered by snow or ice.

# # classes to drop (keep snow=11)
# BAD_SCL = [0, 1, 2, 3, 7, 8, 9, 10]

# BRDF fusion settings
MCD_COLLECTION = "modis-mcd43a3-061"          # MODIS daily albedo (BSA/WSA) v061
MCD_ASSETS = {"BSA":"Albedo_BSA_shortwave", "WSA":"Albedo_WSA_shortwave"}
MCD_RES_NATIVE = 500                           # meters
RATIO_EPS = 1e-3                               # avoid divide-by-zero in ratio sharpen

# -------------------- Helper Functions --------------------
def clamp_reflectance(arr: xr.DataArray) -> xr.DataArray:
    """Guard rails for surface reflectance values."""
    return xr.where((arr < 0) | (arr > 1.5), np.nan, arr)

def get_sun_angles_deg(item):
    """Return (solar_zenith_deg, solar_azimuth_deg) from a Sentinel-2 STAC item."""
    props = item.properties
    zen = props.get("s2:mean_solar_zenith")
    az  = props.get("s2:mean_solar_azimuth")
    if zen is None:
        elev = props.get("view:sun_elevation")
        if elev is None:
            raise ValueError("No solar geometry in item properties.")
        zen = 90.0 - elev
    if az is None:
        az = props.get("view:sun_azimuth")
        if az is None:
            raise ValueError("No solar azimuth in item properties.")
    return float(zen), float(az)

def build_time_series_angles(items, time_coord):
    """Create xarray DataArrays (time) of solar zenith/azimuth in radians."""
    zen_list, az_list = [], []
    for it in items:
        ze, az = get_sun_angles_deg(it)
        zen_list.append(np.deg2rad(ze))
        az_list.append(np.deg2rad(az))
    theta0 = xr.DataArray(np.array(zen_list), coords={"time": time_coord}, dims=("time",))
    phi0   = xr.DataArray(np.array(az_list),  coords={"time": time_coord}, dims=("time",))
    return theta0, phi0

def compute_slope_aspect(dem_m: xr.DataArray):
    """Compute slope [rad] and aspect [rad, 0..2π clockwise from north] from DEM (projected)."""
    xres, yres = dem_m.rio.resolution()
    xres, yres = abs(xres), abs(yres)
    dz_dy, dz_dx = np.gradient(dem_m.values.astype("float64"), abs(yres), abs(xres))
    slope = np.arctan(np.hypot(dz_dx, dz_dy))
    aspect = np.arctan2(dz_dx, -dz_dy)
    aspect = np.where(aspect < 0, aspect + 2*np.pi, aspect)
    slope_da = xr.DataArray(slope, coords=dem_m.coords, dims=dem_m.dims, name="slope")
    aspect_da = xr.DataArray(aspect, coords=dem_m.coords, dims=dem_m.dims, name="aspect")
    return slope_da, aspect_da

def scs_plus_c_correct(refl: xr.DataArray,
                       cos_i: xr.DataArray,
                       cos_theta0: xr.DataArray | np.ndarray,
                       valid_mask: xr.DataArray,
                       max_samples: int = 200_000) -> xr.DataArray:
    """
    SCS+C topographic correction.
    Parameters
    ----------
    refl : (time, band, y, x) reflectance
    cos_i : (time, y, x)      cosine of local incidence angle
    cos_theta0 : (time,)      cosine of solar zenith (per scene/time)
    valid_mask : (time, y, x) boolean where True = keep
    """
    
    # refl: (time, band, y, x), cos_i: (time, y, x), valid_mask: (time, y, x)
    # cos_theta0: (time,) or 1D np array

    # Ensure cos_theta0 is a DataArray with matching 'time'
    # Force cos_theta0 to be a 1D DataArray aligned by POSITION to refl.time
    if isinstance(cos_theta0, xr.DataArray):
        vals = np.asarray(cos_theta0.values)
    else:
        vals = np.asarray(cos_theta0)
    if vals.shape[0] != refl.sizes["time"]:
        raise ValueError("cos_theta0 length does not match refl.time length.")

    # Align by position, not by label (works with duplicate time labels)
    cos_theta0 = xr.DataArray(vals, dims=["time"], coords={"time": refl.time})
    
    # Broadcast to raster shape
    cos_t0_full = cos_theta0.broadcast_like(cos_i)
    
    # We will compute a per-band correction using only valid pixels.
    corrected = []
    for b in refl.band.values:
        r = refl.sel(band=b)
        # Use only valid pixels
        r_valid = r.where(valid_mask)
        ci_valid = cos_i.where(valid_mask)
        c0_valid = cos_t0_full.where(valid_mask)

        # ---- estimation for "C" goes here (placeholder) ----
        # For example, estimate C via robust ratio of means on valid samples:
        #   r = a * ci + b  => C = b / a
        # Sample to limit memory
        cnt_da = r_valid.count(dim=("time","y","x")).compute()  # 0-D DataArray
        if int(cnt_da.values) == 0:
            corrected.append(r)
            continue

        # Take a subset of valid indices if huge
        # (xarray-friendly simple approach: use nanmeans directly)
        mean_r  = r_valid.mean(dim=("y", "x"), skipna=True)
        mean_ci = ci_valid.mean(dim=("y", "x"), skipna=True)

        # Avoid divide-by-zero; simple guard
        C = (mean_r / (mean_ci + 1e-12)).clip(min=0)  # shape: (time,)

        # Broadcast C to raster shape
        C_full = C.broadcast_like(ci_valid)

        # SCS+C correction: R' = R * (cosθ0 + C) / (cos i + C)
        r_corr = r * (c0_valid + C_full) / (ci_valid + C_full)
        corrected.append(r_corr)

    corrected = xr.concat(corrected, dim="band").assign_coords(band=refl.band)
    return corrected

def iso(t):  # xarray/numpy datetime --> International Organization for Standardization (ISO) date (YYYY-MM-DD)
    return np.datetime_as_string(np.asarray(t).astype("datetime64[D]"), unit="D")

# Helper to select the nearest MCD slice to a given S2 time (tolerance 8 days)
# If there is no item within tolerance, it will still select nearest; adjust as needed.
def select_nearest_mcd(mcd_da: xr.DataArray, tval: np.datetime64) -> xr.DataArray:
    return mcd_da.sel(time=tval, method="nearest")

def pick_asset_key(keys, prefer):
    # Prefer full names like 'Albedo_BSA_shortwave'; else try shorter 'BSA_shortwave'
    for k in keys:
        if prefer in k:
            return k
    short = prefer.replace("Albedo_", "")
    for k in keys:
        if short in k:
            return k
    raise KeyError(f"Asset containing '{prefer}' not found in: {keys}")

# find subdataset hrefs for a given file
def find_sds_hrefs(hdf_path: Path):
    sds = []
    with rio.open(hdf_path) as ds:
        for s in ds.subdatasets:
            s_lower = s.lower()
            # match both 'albedo_bsa_shortwave' and 'bsa_shortwave' (same for wsa)
            if ("bsa_shortwave" in s_lower) or ("albedo_bsa_shortwave" in s_lower):
                sds.append(("BSA", s))
            if ("wsa_shortwave" in s_lower) or ("albedo_wsa_shortwave" in s_lower):
                sds.append(("WSA", s))
    return sds  # list of tuples: [("BSA", sds_href), ("WSA", sds_href), ...]

# Extract date from filenames (e.g., MCD43A3.A2023080.h09v05.061....)
# Date parser from HDF filename (Ayyyyddd). We will parse the 'AYYYYDDD' field and convert to datetime64[D]
def date_from_fname(p: Path) -> np.datetime64:
    """
    Parse MODIS 'Ayyyydoy' from filename and return np.datetime64[D].
    Accepts ...A2021234... or ... .A2021234. ... patterns.
    """
    m = re.search(r"\.A(\d{7})\.", p.name) or re.search(r"A(\d{7})", p.name)
    if not m:
        raise ValueError(f"Cannot parse Ayyyydoy from filename: {p.name}")
    yyyydoy = m.group(1)
    year = int(yyyydoy[:4]); doy = int(yyyydoy[4:])
    d = (datetime.datetime(year, 1, 1) + datetime.timedelta(days=doy - 1)).date()
    # ensure day precision
    return np.datetime64(d, "D")

def open_sds_reproject(sds_href: str, dst_epsg: int, dst_res=500) -> xr.DataArray:
    da = rxr.open_rasterio(sds_href, masked=True).squeeze()
    # NOTE: subdatasets carry their own CRS; ensure present
    if da.rio.crs is None:
        raise ValueError(f"SDS has no CRS: {sds_href}")
    da_utm = da.rio.reproject(
        CRS.from_epsg(dst_epsg),
        resolution=dst_res,
        resampling=Resampling.bilinear,
    ).astype("float32")
    return da_utm

def build_stack_from_sds(pairs: list[tuple[Path, str]], epsg: int) -> xr.DataArray:
    # sort by (date, filename) to be deterministic within a day
    pairs_sorted = sorted(pairs, key=lambda t: (date_from_fname(t[0]), t[0].name))
    first = open_sds_reproject(pairs_sorted[0][1], epsg)
    tpl = first
    rasters, times = [first], [date_from_fname(pairs_sorted[0][0])]
    for p, href in pairs_sorted[1:]:
        da_utm = open_sds_reproject(href, epsg)
        if (da_utm.rio.transform() != tpl.rio.transform()) or (da_utm.rio.crs != tpl.rio.crs):
            da_utm = da_utm.rio.reproject_match(tpl, resampling=Resampling.bilinear)
        rasters.append(da_utm)
        times.append(date_from_fname(p))
    st = xr.concat(rasters, dim="time").assign_coords(time=np.array(times))
    st = st.rio.write_crs(CRS.from_epsg(epsg))
    return st

###################################################################################
###################################################################################
