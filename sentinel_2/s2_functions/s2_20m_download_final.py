# packages
import os
import sys
import csv
import json
import math
import warnings
import numpy as np
import pandas as pd
import pystac_client
import datetime as dt
import rasterio as rio
import geopandas as gpd
import planetary_computer
from datetime import timezone
import matplotlib.pyplot as plt
from rasterio.io import MemoryFile
from collections import defaultdict
from scipy.signal import convolve2d
from shapely.geometry import mapping
from rasterio.enums import Resampling
from rasterio.transform import Affine
from affine import Affine as AffineCls
from rasterio.plot import plotting_extent
from rasterio import warp, windows, features
from pystac.extensions.eo import EOExtension as eo
from rasterio.errors import NotGeoreferencedWarning
from rasterio import warp, windows, features, enums

# stack writer
try:
    import xarray as xr
    XR_AVAILABLE = True
except Exception:
    XR_AVAILABLE = False
    
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

# ----------------------------
# USER CONFIG
# ----------------------------
# Paths
shapefile_path = "/bsuhome/tnde/scratch/felix/modis/East_River_SHP/ER_bbox.shp"
out_dir = "/bsuhome/tnde/scratch/felix/Sentinel-2/s2_albedo_outputs"
os.makedirs(out_dir, exist_ok=True)

# Time, AOI
time_of_interest = "2021-09-01/2022-08-31" #"2021-09-01/2023-06-15" # (full time range of interest) "2022-12-31/2023-06-16" # "2022-01-15/2022-02-06" # "2021-09-01/2021-12-31"

# Processing resolution + target geometry:
# We use 20 m grid to accommodate B8A/B11/B12 natively; 10 m bands are upsampled with bilinear.
TARGET_RES = 20  # meters

# Snow detection
# NOTE (Source: Copernicus website)
# The Sentinel-2 normalised difference snow index can be used to differentiate 
# between cloud and snow cover as snow absorbs in the short-wave infrared light, 
# but reflects the visible light, whereas cloud is generally reflective in both wavelengths. 
# NDSI is calculated by taking the difference between the green band reflectance 
# and the SWIR band reflectance, then dividing by their sum
# Hence, NDSI = (Green - SWIR) / (Green + SWIR)
USE_NDSI_FOR_SNOW = True    # Create snow mask using Normalized Difference Snow Index (NDSI) mask (common thresholds). If False use scene classification (SCL) class 11 when 
NDSI_THRESH = 0.4 # Or values above 0.42 are usually snow according to Copernicus. Link: https://custom-scripts.sentinel-hub.com/sentinel-2/ndsi/
GREEN_MIN = 0.2

# MODIS VIS/NIR fallback if VIS/NIR coefficients below are not filled
USE_MODIS_FALLBACK_FOR_VIS_NIR = False # Planetary Computer does not host the MODIS MCD43 data, so we can't fall back on it.
BLUE_SKY_DIFFUSE_FRACTION = 0.25  # tweak if there is no meteorological data; ~0.1 clear, ~0.5 hazy

# ----------------------------
# Coefficients
# ----------------------------
# Bidirectional Reflectance Distribution Function (BRDF)
# BRDF c-factor global kernel weights (Roy 2016; HLS docs) for MODIS-equivalent bands
# We will map these to S2 bands: BLUE=B02, GREEN=B03, RED=B04, NIR=B8A (or B08), SWIR1=B11, SWIR2=B12
# Ref: Harmonized Landsat Sentinel-2 (HLS) Product User Guide Table 4 (Claverie et al.) 
# Link: https://lpdaac.usgs.gov/documents/1698/HLS_User_Guide_V2.pdf
BRDF_COEFFS = { # BRDF coefficients used for the c-factor approach (Roy et al. 2016 and 2017)
    "BLUE":  {"fiso": 0.0774, "fgeo": 0.0079,  "fvol": 0.0372},
    "GREEN": {"fiso": 0.1306, "fgeo": 0.0178,  "fvol": 0.0580},
    "RED":   {"fiso": 0.1690, "fgeo": 0.0227,  "fvol": 0.0574},
    "NIR":   {"fiso": 0.3093, "fgeo": 0.0330,  "fvol": 0.1535},
    "SWIR1": {"fiso": 0.3430, "fgeo": 0.0453,  "fvol": 0.1154},
    "SWIR2": {"fiso": 0.2658, "fgeo": 0.0387,  "fvol": 0.0639},
}

# Sentinel-2 Multi-Spectral Instrument (MSI) narrow-to-broadband albedo coefficients (Li et al., 2018, Table 2) for SW, VIS, NIR.
# IMPORTANT: Fill VIS and NIR with the exact numbers from Table 2 of Li et al., 2018 (Remote Sensing of Environment (RSE) 217:352–365).
# Link to Li et. al.: https://www.sciencedirect.com/science/article/pii/S0034425718304024
# The SW set below is a commonly cited one for S2 (documented in multiple repos/papers); can be verified with the paper.
# Format is linear: alpha = const + sum_i (wi * band_i), using reflectance in [0...1].
# Compute broadband shortwave albedo (Wang et al., 2022; Li et al., 2018)
# αSW = 0.2688*B02 + 0.0362*B03 + 0.1501*B04 + 0.3045*B8A + 0.1644*B11 + 0.0356*B12 - 0.0049 #snow-free

# --- From Table 2 of Li et. al., 2018. Both snow and snow-free coefficients ---
ALBEDO_COEFFS = { # All values are taken from Li et al. (2018)
    "SW": { # Shortwave (0.30–5.0 µm)
        # snow
        "snow": {"const": -0.0001, "B02": -0.1992, "B03": 2.3002, "B04": -1.9121, "B8A": 0.6715, "B11": -2.2728, "B12": 1.9341},
        # snow-free
        "snow_free": {"const": -0.0049, "B02": 0.2688, "B03": 0.0362, "B04": 0.1501, "B8A": 0.3045, "B11": 0.1644, "B12": 0.0356},
    },
    "VIS": { # Visible (0.30–0.70 µm)
        # snow
        "snow": {"const": -0.0052, "B02": 0.8421, "B03": 0.1487, "B04": 0.0088, "B8A": None,    "B11": None,    "B12": None},
        # snow-free
        "snow_free": {"const": -0.0048, "B02": 0.5673, "B03": 0.1407, "B04": 0.2359, "B8A": None,    "B11": None,    "B12": None},
    },
    "NIR": { # Near-IR (0.70–5.0 µm)
        # snow
        "snow": {"const": -0.0221, "B02": None,   "B03": None,   "B04": None, "B8A": 0.6793,  "B11": 0.0244,  "B12": 0.6192},   
        # snow-free
        "snow_free": {"const": -0.0073, "B02": None,   "B03": None,   "B04": None, "B8A": 0.5595,  "B11": 0.3844,  "B12": 0.0290},
    },
}


# Reference geometry for BRDF normalization (HLS uses nadir, SZA=45 degrees, RAA=0)
REF_SZA_DEG = 45.0
REF_VZA_DEG = 0.0
REF_RAA_DEG = 0.0


# # Sentinel-2 Scene Classification Layer (SCL)
# # The classification scheme includes 12 distinct classes: 
# NOTE: 
# Scene classification was developed to distinguish between cloudy pixels, 
# clear pixels and water pixels of Sentinel-2 data and is a result of ESA's Scene classification algorithm. 
# Twelve different classifications are provided including classes of clouds, vegetation, soils/desert, water and snow. 
# It does not constitute a land cover classification map in a strict sense.
# Class 0 - No data: Pixels where no data is available.
# Class 1 - Saturated or Defective: Pixels with corrupted or unusable data.
# Class 2 - Topographic casted shadows (called "Dark features/Shadows" for data before 2022-01-25): 
            # Pixels that are unusually dark, possibly due to shadows or specific surface conditions.
# Class 3 - Cloud shadows: Pixels directly underneath clouds.
# Class 4 - Vegetation: Pixels covered by plants and trees.
# Class 5 - Bare soils: Pixels showing exposed soil or deserts (non vegetated areas).
# Class 6 - Water: Pixels representing lakes, rivers, or oceans.
# Class 7 - Cloud low probability / Unclassified: Areas suspected of having clouds or where classification was uncertain.
# Class 8 - Cloud medium probability: Pixels with a moderate likelihood of being clouds.
# Class 9 - Cloud high probability: Pixels with a high certainty of being clouds.
# Class 10 - Thin cirrus: High-altitude, thin clouds that are nearly transparent.
# Class 11 - Snow or ice: Pixels covered by snow or ice.

# Cloud classes to mask in SCL (Sen2Cor L2A)
# SCL_MASK_CLASSES = {0, 1, 3, 7, 8, 9, 10}  # no data, saturated, cloud shadow, unclassified, cloud medium probability, clouds high probability, cirrus
SCL_MASK_CLASSES = {0, 1, 3, 7, 9, 10}  # no data, saturated, cloud shadow, unclassified, clouds high probability, cirrus
SCL_SNOW_CLASS = 11  # snow/ice
CUTOFF = dt.datetime(2022, 1, 25) # Processing baseline change cutoff date
HARMONIZE_BANDS = {"B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12"}  # excluding B10
# NOTE: B10 is the cirrus band on Sentinel-2 Multi-Spectral Instrument (MSI).
# It is purpose is to detect thin, high-level cirrus clouds.
# It sits in a strong water-vapor absorption region, so there’s little true surface signal.
# It is used for cloud screening, not for surface products.
# It is not included in common narrow-to-broadband albedo formulas (e.g., Li et al.) and should be kept excluding B10 from albedo/BRDF computations.
# In many L2A pipelines, B10 is not provided as surface reflectance (or isn’t useful even if present) because of that absorption.
# The new-baseline DN values would be clipped at 1000 and then −1000, so most dark/near-zero pixels end up exactly 0 after scaling to reflectance — i.e., we learn nothing new.
# It will not improve comparability for surface products because B10 isn’t used for them.
# If we use B10 for cirrus screening, we typically want the signal as delivered by the processor; altering it may invalidate thresholds we rely on.

# AOI from shapefile (WGS84)
Boundary = gpd.read_file(shapefile_path)
Boundary_wgs84 = Boundary.to_crs(epsg=4326)
AOI_BOUNDS_LL = tuple(Boundary_wgs84.total_bounds)
AOI_GEOM = mapping(Boundary_wgs84.union_all())
area_of_interest = AOI_GEOM

# Index for NetCDF/CSV
records = []
time_index = []
sw_keep_stack_hard = []
sw_snow_stack_hard = []
vis_keep_stack_hard = [] 
vis_snow_stack_hard = []
nir_keep_stack_hard = []
nir_snow_stack_hard = []

sw_keep_stack_soft = []
sw_snow_stack_soft = []
vis_keep_stack_soft = [] 
vis_snow_stack_soft = []
nir_keep_stack_soft = []
nir_snow_stack_soft = []

# NEW: stacks for BRDF-derived albedos
bsa_keep_stack_hard = []
bsa_keep_stack_soft = []
wsa_keep_stack_hard = []
wsa_keep_stack_soft = []
blue_sky_keep_stack_hard = []
blue_sky_keep_stack_soft = []

bsa_snow_stack_hard = []
bsa_snow_stack_soft = []
wsa_snow_stack_hard = []
wsa_snow_stack_soft = []
blue_sky_snow_stack_hard = []
blue_sky_snow_stack_soft = []

# NEW: Quality Assurance (QA) stacks
qa_p_snow_stack = []
qa_snow_mask_stack = []
qa_choice_hard_stack = []
qa_choice_soft_stack = []
qa_ndsi_stack = []

# Map bands to BRDF keys
BRDF_MAP = {"B02": "BLUE", "B03": "GREEN", "B04": "RED", "B8A": "NIR", "B11": "SWIR1", "B12": "SWIR2"}

# ----------------------------------
# Compute diffuse fraction, fdiff
# ----------------------------------
# Read diffuse skylight ratio from Lookup Table (LUT)
diffuse_skylight_ratio_lookup = "/bsuhome/tnde/geoscience/albedo_downscaling/GOES-Modis-Data-Preprocessing-main/sw_lut.csv"
diffuse_lut = pd.read_csv(diffuse_skylight_ratio_lookup, index_col=0)  # sw_lut.csv
USE_AOT = True # Use Aerosol Optical Thickness (AOT) from Sentinel-2 to compute the diffuse fraction, fdiff. Use Aerosol Optical Depth (AOD) from MODIS if False.

function_path = os.path.expanduser("~/geoscience/albedo_downscaling/functions")
sys.path.append(function_path)
# import MODIS helper functions.
from modis_bluesky_albedo import get_aod_static_data, convert_last_digit_of_float, SAIL_LOCATION, aod_lookup_table_file

# Read diffuse skylight ratio from Lookup Table (LUT)
def lookup_diffuse_fraction_from_lut(sza_deg: float, aod_value: float) -> float:
    """
    Look up diffuse fraction f_diff in sw_lut.csv given SZA (deg) and AOD.

    - sza_deg: solar zenith angle in degrees (can be mean SZA for the scene)
    - aod_value: AOD or AOT at 550 nm (dimensionless)

    Returns:
        f_diff (scalar float)
    """
    # clip SZA to LUT range [0, 89] and round to nearest integer
    sza_idx = int(np.clip(np.round(sza_deg), 0, 89))

    # clip AOD to LUT column range (0–0.98 here)
    aod_clipped = float(np.clip(aod_value, 0.0, 0.98))

    # turn AOD into a string column name that matches the CSV
    aod_key = convert_last_digit_of_float(aod_clipped)  # e.g. "0.14"
    if float(aod_key) == 0.0:
        aod_key = f"{float(aod_key):.2f}"  # ensure "0.00" not "0" for the column name

    # lookup in the LUT
    f_diff = float(diffuse_lut.loc[sza_idx, aod_key])
    return f_diff

# ---------------------------------------------------
# Helpers: BSA, WSA, and Blue-sky using MODIS BRDF
# ---------------------------------------------------
# ==== MODIS RTLSR kernel integrals for BSA/WSA (RossThick and LiSparse-Reciprocal) ====
# Black-sky (directional-hemispherical) kernel integrals are quadratic in cos(SZA)
G_BSA = {
    "vol": (-0.007574, -0.070987,  0.307588),   # g0, g1, g2 for volumetric term
    "geo": (-1.284909, -0.166314,  0.041840),   # g0, g1, g2 for geometric term
}

# White-sky (bi-hemispherical, isotropic) integrals are constants
G_WSA = {
    "iso": 1.0,
    "vol": 0.189184,
    "geo": -1.377622,
}

# --------------------------------------------------------------
# Helpers: BRDF kernels (Ross-Thick and Li-Sparse-Reciprocal)
# --------------------------------------------------------------
def deg2rad(a): return a * np.pi / 180.0 # convert a value/measurement from degrees to radians

def ross_thick_kernel(sza_deg, vza_deg, raa_deg):
    """
    Ross-Thick volumetric kernel.
    """
    sza = deg2rad(sza_deg); vza = deg2rad(vza_deg); raa = deg2rad(raa_deg)
    cos_xi = np.cos(sza)*np.cos(vza) + np.sin(sza)*np.sin(vza)*np.cos(raa)
    xi = np.arccos(np.clip(cos_xi, -1, 1))
    Kvol = ((np.pi/2 - xi)*np.cos(xi) + np.sin(xi)) / (np.cos(sza) + np.cos(vza)) - np.pi/4
    return Kvol

def li_sparse_kernel(sza_deg, vza_deg, raa_deg):
    """
    Li-Sparse-Reciprocal geometric kernel.
    """
    sza = deg2rad(sza_deg); vza = deg2rad(vza_deg); raa = deg2rad(raa_deg)
    tan_s = np.tan(sza); tan_v = np.tan(vza)
    sec_s = 1.0/np.cos(sza); sec_v = 1.0/np.cos(vza)
    # overlap angle
    cos_psi = -np.cos(raa)
    D = np.sqrt(tan_s**2 + tan_v**2 - 2*tan_s*tan_v*np.cos(raa))
    # Avoid divide-by-zero
    D = np.where(D == 0, 1e-6, D)
    temp = (1.0/np.pi) * (D - np.sin(D)) * (sec_s + sec_v)
    Kgeo = temp - (sec_s + sec_v - (1.0/np.pi)*(np.sin(D)))  # simplified common implementation
    return Kgeo

def brdf_c_factor(band_key, sza_deg, vza_deg, raa_deg):
    """
    Compute c-factor for one band using global kernel weights.
    """
    coeff = BRDF_COEFFS[band_key]
    # Kernels at observed geometry
    Kvol_o = ross_thick_kernel(sza_deg, vza_deg, raa_deg)
    Kgeo_o = li_sparse_kernel(sza_deg, vza_deg, raa_deg)
    num_o = coeff["fiso"] + coeff["fvol"] * Kvol_o + coeff["fgeo"] * Kgeo_o

    # Kernels at reference geometry
    Kvol_r = ross_thick_kernel(REF_SZA_DEG, REF_VZA_DEG, REF_RAA_DEG)
    Kgeo_r = li_sparse_kernel(REF_SZA_DEG, REF_VZA_DEG, REF_RAA_DEG)
    num_r = coeff["fiso"] + coeff["fvol"] * Kvol_r + coeff["fgeo"] * Kgeo_r

    c = num_r / np.maximum(num_o, 1e-6)
    return c

def k_bsa_terms(sza_deg: float):
    mu0 = np.cos(np.deg2rad(np.clip(sza_deg, 0, 89.999)))
    kvol = G_BSA["vol"][0] + G_BSA["vol"][1]*mu0 + G_BSA["vol"][2]*(mu0**2)
    kgeo = G_BSA["geo"][0] + G_BSA["geo"][1]*mu0 + G_BSA["geo"][2]*(mu0**2)
    return kvol, kgeo

def k_wsa_terms():
    return G_WSA["vol"], G_WSA["geo"]

def to_bsa_wsa_band(ref_obs, band_key_s2, sza_deg, vza_deg=0.0, raa_deg=0.0):
    """
    Convert observed (topography-corrected) reflectance in one S2 band to BSA and WSA
    using MODIS RTLSR BRDF with HLS global weights.

    ref_obs: 2D array (NaNs where invalid/cloud)
    band_key_s2: one of {"B02","B03","B04","B8A","B11","B12"}
    """
    # Map S2 band to HLS weight row already in BRDF_COEFFS
    brdf_key = {"B02":"BLUE","B03":"GREEN","B04":"RED","B8A":"NIR","B11":"SWIR1","B12":"SWIR2"}[band_key_s2]
    w = BRDF_COEFFS[brdf_key]

    # Kernels at observation geometry (same as c-factor uses)
    Kvol_o = ross_thick_kernel(sza_deg, vza_deg, raa_deg)
    Kgeo_o = li_sparse_kernel(sza_deg, vza_deg, raa_deg)
    denom = w["fiso"] + w["fvol"]*Kvol_o + w["fgeo"]*Kgeo_o
    denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)

    # Target = BSA at this SZA
    Kvol_bsa, Kgeo_bsa = k_bsa_terms(sza_deg)
    num_bsa = w["fiso"] + w["fvol"]*Kvol_bsa + w["fgeo"]*Kgeo_bsa
    c_bsa = num_bsa / denom
    rho_bsa = np.clip(ref_obs * np.float32(c_bsa), 0.0, 1.0)

    # Target = WSA (isotropic)
    Kvol_wsa, Kgeo_wsa = k_wsa_terms()
    num_wsa = w["fiso"] + w["fvol"]*Kvol_wsa + w["fgeo"]*Kgeo_wsa
    c_wsa = num_wsa / denom
    rho_wsa = np.clip(ref_obs * np.float32(c_wsa), 0.0, 1.0)
    return rho_bsa, rho_wsa

# --- Per-band BSA, WSA, Blue-sky: HARD switch and SOFT blend (no narrow-to-broadband) ---
def blue_sky_from_bsa_wsa(bsa_band, wsa_band, fdiff=BLUE_SKY_DIFFUSE_FRACTION):
    # Blue-sky = fdiff * WSA + (1 - fdiff) * BSA
    return np.clip(fdiff * wsa_band + (1.0 - fdiff) * bsa_band, 0.0, 1.0).astype("float32")

def soft_blend_albedo(snow_prob, snow_albedo, snow_free_albedo):
    """
    Compute soft blend albedo values. Useful for blending near patchy pixel edges.
    snow_prob:        Probability that a pixel is covered in snow. Obtain by a combination of SCL and NDSI mask and a piecewise-linear 
                      ramp centered on NDSI threshold. It lets us soft blend snow vs. snow-free coefficients near the boundary instead of hard switching.
    snow_albedo:      Albedo computed from snow present broadband coefficients
    snow_free_albedo: Albedo computed from snow-free broadband coefficients
    """
    soft_blend_a = (snow_prob * snow_albedo + (1.0 - snow_prob) * snow_free_albedo).astype("float32")
    return soft_blend_a

# -----------------------------------
# Helpers: Topographic C-correction
# -----------------------------------
def slope_aspect(dem, transform):
    """
    Compute slope [rad] and aspect [rad] from DEM using Horn's method.
    """
    # pixel sizes
    dx = transform.a
    dy = -transform.e

    # Sobel/Horn derivatives
    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]]) / (8*dx)
    kernel_y = np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]]) / (8*dy)

    dzdx = convolve2d(dem, kernel_x, mode="same", boundary="symm")
    dzdy = convolve2d(dem, kernel_y, mode="same", boundary="symm")

    slope = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(dzdy, -dzdx)
    aspect = np.where(aspect < 0, 2*np.pi + aspect, aspect)
    return slope, aspect

def topo_c_correction(refl, slope, aspect, sza_deg, saa_deg):
    """
    Apply C-correction topographic normalization per band array.
    """
    sza = deg2rad(sza_deg)
    saa = deg2rad(saa_deg)

    # incidence angle on slope: cos(i) = cos(slope)*cos(sza) + sin(slope)*sin(sza)*cos(saa - aspect)
    cos_i = np.cos(slope)*np.cos(sza) + np.sin(slope)*np.sin(sza)*np.cos(saa - aspect)
    cos_sza = np.cos(sza)

    # linear regression of R vs cos(i): R = a * cos(i) + b  (compute over valid pixels)
    valid = np.isfinite(refl) & np.isfinite(cos_i)
    if valid.sum() < 100:
        return refl  # not enough pixels to fit

    x = cos_i[valid].ravel()
    y = refl[valid].ravel()
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    if abs(a) < 1e-8:
        return refl
    c = b / a

    corrected = refl * ((cos_sza + c) / np.maximum(cos_i + c, 1e-6))
    return corrected

# ---------------------------------------------
# Helpers: albedo narrow-to-broadband (NB->BB)
# ---------------------------------------------
def nb_to_bb(coeffs, bands):
    """
    coeffs = dict with const and per-band weights; bands = dict of arrays keyed by 'B02','B03','B04','B8A','B11','B12' in reflectance [0...1].
    """
    const = coeffs["const"]
    if const is None:
        raise ValueError("Broadband coefficient set is not populated (const=None). Fill from Li et al., 2018.")
    out = np.full_like(next(iter(bands.values())), const, dtype=np.float32)
    for k, w in coeffs.items():
        if k == "const" or w is None: 
            continue
        if k not in bands:
            raise KeyError(f"Band {k} not in provided band stack.")
        out = out + np.float32(w) * np.float32(bands[k])
    return np.clip(out, 0.0, 1.0)

def nb_to_bb_set(coeffs_set: dict, bands: dict):
    """Compute broadband for BOTH 'snow' and 'snow_free' sets; returns (snow_val, now_free_val)."""
    snow_val    = nb_to_bb(coeffs_set["snow"], bands)
    now_free_val  = nb_to_bb(coeffs_set["snow_free"], bands)
    return snow_val, now_free_val

def ndsi_soft_probability(ndsi, center=0.3, halfwidth=0.1, use_ndsi_as_p=False):
    """
    Map NDSI to [0,1] smoothly around the threshold: 
    p=0 at (center-halfwidth), p=1 at (center+halfwidth), linear in between.
    """
    if use_ndsi_as_p: #use NDSI values straightforward as snow probabilities.
        p = ndsi
    else:
        lo = center - halfwidth
        hi = center + halfwidth
        p = (ndsi - lo) / (hi - lo)
    return np.clip(p, 0.0, 1.0).astype("float32")

# ----------------------------
# Helpers: IO and resampling
# ----------------------------
def read_window(asset_href, window):
    with rio.open(asset_href) as ds:
        data = ds.read(window=window, out_dtype="float32")
        transform = ds.window_transform(window)
        crs = ds.crs
        nodata = ds.nodata
    return data, transform, crs, nodata

def reproject_match(src_data, src_transform, src_crs, dst_transform, dst_crs, dst_shape, resampling=Resampling.bilinear):
    dst = np.zeros((src_data.shape[0], dst_shape[0], dst_shape[1]), dtype="float32")
    for i in range(src_data.shape[0]):
        warp.reproject(
            source=src_data[i],
            destination=dst[i],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling,
            num_threads=2,
        )
    return dst

def write_coglike(path, array, transform, crs, nodata=np.nan, dtype="float32", description=None):
    profile = {
        "driver": "GTiff",
        "height": array.shape[-1],
        "width": array.shape[-2],
        "count": 1 if array.ndim == 2 else array.shape[0],
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "compress": "deflate",
        "predictor": 2,
        "nodata": nodata,
        "BIGTIFF": "IF_SAFER",
    }
    with rio.open(path, "w", **profile) as dst:
        if array.ndim == 2:
            dst.write(array, 1)
        else:
            dst.write(array)
        if description:
            dst.update_tags(**{"DESCRIPTION": description})
        dst.update_tags(FORMULA="0.2688*B02+0.0362*B03+0.1501*B04+0.3045*B8A+0.1644*B11+0.0356*B12-0.0049")
        # add overviews
        overviews = [2, 4, 8, 16]
        dst.build_overviews(overviews, Resampling.nearest)
        dst.update_tags(ns="rio_overview", resampling="nearest")

# ----------------------------------------------------
# Helpers: Adjusting for Processing Baseline Change
# ----------------------------------------------------  
# We had to worry about the Sentinel-2 processing baseline change (2022-01-25). 
# The “new baseline” adds a +1000 DN offset to radiance/reflectance bands (so negatives aren’t clamped). 
# To make pre- and post-2022-01-25 imagery comparable, 
# we should remove that offset on the new-baseline scenes before any BRDF/topo/snow/albedo steps.
def is_new_baseline(item):
    """Prefer the item property; fallback to date."""
    pb = item.properties.get("s2:processing_baseline")
    if pb:
        # e.g. "04.00" or "04.01" => new baseline
        try:
            return float(pb.split(".")[0]) >= 4.0
        except Exception:
            pass
    return item.datetime >= CUTOFF

def harmonize_new_dn_to_old(raw_dn):
    """
    Reverse the +1000 offset used by the new baseline:
    clip at 1000 to mimic the old clamping, then subtract 1000.
    """
    # raw_dn is the integer/scaled DN as read from the GeoTIFF
    return np.clip(raw_dn, 1000, None) - 1000

# Helper: read DEM (Copernicus DEM GLO-30) over AOI and reproject to target grid
def load_dem_to_target(catalog, W, H, dst_transform, dst_crs):
    # Find a DEM item intersecting the AOI
    dem_search = catalog.search(
        collections=["cop-dem-glo-30"],
        intersects=area_of_interest,
    )
    dem_item = next(dem_search.items(), None)
    if dem_item is None:
        raise RuntimeError("DEM not found in AOI (cop-dem-glo-30).")

    dem_href = dem_item.assets["data"].href

    # Read AOI subset from the DEM in its native grid
    with rio.open(dem_href) as dem_ds:
        # AOI bounds in DEM CRS
        dem_bounds = warp.transform_bounds("EPSG:4326", dem_ds.crs, *AOI_BOUNDS_LL)
        dem_win = windows.from_bounds(*dem_bounds, transform=dem_ds.transform).round_offsets().round_lengths()

        dem_arr = dem_ds.read(1, window=dem_win, out_dtype="float32")
        dem_src_transform = dem_ds.window_transform(dem_win)
        dem_src_crs = dem_ds.crs

    # Reproject DEM subset to the Sentinel-2 target grid (20 m)
    dem_match = np.empty((H, W), dtype="float32")
    warp.reproject(
        source=dem_arr,
        destination=dem_match,
        src_transform=dem_src_transform,
        src_crs=dem_src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        num_threads=2,
    )
    return dem_match

# try VIS/NIR or fallback to MODIS VIS/NIR
def try_vis_nir_or_fallback(kind, per_band):
    coeff = ALBEDO_COEFFS[kind]
    if coeff["const"] is not None:
        return nb_to_bb(coeff, per_band)
    elif USE_MODIS_FALLBACK_FOR_VIS_NIR:
        # Pull MODIS MCD43A3 (VIS/NIR) for the date and AOI; resample to grid.
        # NOTE: For brevity we just use the nearest composite date.
        mcd = catalog.search(collections=["modis-43a3-061"], intersects=area_of_interest, datetime=date)
        mitems = list(mcd.get_items())
        if not mitems:
            return np.full_like(sw, np.nan, dtype=np.float32)
        # asset names are usually like "Albedo_VIS_BSA" / "Albedo_NIR_BSA" etc.
        asset_key = f"Albedo_{kind}_BSA"
        if asset_key not in mitems[0].assets:
            # try WSA
            asset_key = f"Albedo_{kind}_WSA"
            if asset_key not in mitems[0].assets:
                return np.full_like(sw, np.nan, dtype=np.float32)
        href = mitems[0].assets[asset_key].href
        with rio.open(href) as ds:
            win = windows.from_bounds(*warped_aoi_bounds, transform=ds.transform)
            win = win.round_offsets().round_lengths()
            data, s_tr, s_crs, _ = read_window(href, win)
        val = (data[0].astype("float32") / 1000.0)  # MCD43 scale
        match = reproject_match(val[None, ...], s_tr, s_crs, dst_transform, dst_crs, (H, W), resampling=Resampling.bilinear)[0]
        return np.clip(match, 0.0, 1.0)
    else:
        raise RuntimeError(f"{kind} albedo coefficients not provided.")
           
# Write NetCDF stack if xarray present
def stack_to_da(stack, name):
    data = np.stack(stack, axis=0).astype("float32")  # (time, y, x)
    return xr.DataArray(
        data,
        dims=("time", "y", "x"),
        coords={"time": np.array(time_index)},
        name=name,
    )

###################################################################################################################
###################################################################################################################
if __name__ == "__main__":
    # ----------------------------
    # Planetary Computer STAC
    # ----------------------------
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    
    # Search Sentinel-2 L2A
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=area_of_interest,
        datetime=time_of_interest,
        # query={"eo:cloud_cover": {"lt": 60}}, # up to 60% cloud cover
    )

    # New:
    items = list(search.items())
    print(f"Returned {len(items)} items")

    if not items:
        raise SystemExit("No items found for query.")
        
    # Prepare a target grid (use first item's 20 m grid)
    sample_item = items[0]
    with rio.open(sample_item.assets["B11"].href) as ds20:
        aoi_bounds = features.bounds(area_of_interest)
        warped_aoi_bounds = warp.transform_bounds("epsg:4326", ds20.crs, *aoi_bounds)
        aoi_window = windows.from_bounds(*warped_aoi_bounds, transform=ds20.transform)
        # snap to full pixels
        aoi_window = aoi_window.round_offsets().round_lengths()
        H = aoi_window.height
        W = aoi_window.width
        dst_transform = ds20.window_transform(aoi_window)
        dst_crs = ds20.crs
        
    DEM = load_dem_to_target(catalog, W, H, dst_transform, dst_crs)
    SLOPE, ASPECT = slope_aspect(DEM, dst_transform)
    # print("Slope",SLOPE, "Aspect", ASPECT)
    
    for item in items:
        date = item.datetime.date().isoformat()
        print(f"Processing {item.id} ({date})")

        # Read S2 bands (reflectance scale is 0...10000 usually; convert to [0...1])
        band_hrefs = {b: item.assets[b].href for b in ["B02", "B03", "B04", "B8A", "B11", "B12"]}
        scl_href = item.assets.get("SCL").href if "SCL" in item.assets else None

        per_band = {}
        for b, href in band_hrefs.items():
            # Read window in native band grid
            with rio.open(href) as ds:
                win = windows.from_bounds(*warped_aoi_bounds, transform=ds.transform)
                win = win.round_offsets().round_lengths()
                data, src_transform, src_crs, _ = read_window(href, win)
                # Reproject to 20 m grid
                if ds.res[0] != TARGET_RES or ds.crs != dst_crs or src_transform != dst_transform:
                    out = reproject_match(data, src_transform, src_crs, dst_transform, dst_crs, (H, W),
                                          resampling=Resampling.bilinear)
                else:
                    out = data
            # WITHOUT ACCOUNTING FOR PROCESSING BASELINE
            # Uncomment this code and run it if you do not want to account for the processing baseline change
            # Do not forget to comment the lines below
            # arr = (out[0].astype("float32") / 10000.0)
            # per_band[b] = np.where(np.isfinite(arr), arr, np.nan)

            # ACCOUNTING FOR PROCESSING BASELINE
            # harmonize first, then scale to reflectance
            raw = out[0].astype("float32")  # DNs (0...10000-ish, or 0...~11000 for new baseline)
            if is_new_baseline(item) and b in HARMONIZE_BANDS:
                raw = harmonize_new_dn_to_old(raw)
            arr = raw / 10000.0  # reflectance in [0...1] (approx; keep later clipping)
            per_band[b] = np.where(np.isfinite(arr), arr, np.nan)

        # SCL mask
        mask_valid = np.ones((H, W), dtype=bool)
        snow_mask = np.zeros((H, W), dtype=bool)
        
        if scl_href:
            with rio.open(scl_href) as ds:
                win = windows.from_bounds(*warped_aoi_bounds, transform=ds.transform)
                win = win.round_offsets().round_lengths()
                scl, s_tr, s_crs, _ = read_window(scl_href, win)
            scl_match = reproject_match(scl, s_tr, s_crs, dst_transform, dst_crs, (H, W), resampling=Resampling.nearest)[0]
            scl = scl_match.astype(np.uint8)
            mask_valid &= ~np.isin(scl, list(SCL_MASK_CLASSES))
                

        # NDSI snow (needs GREEN=B03, SWIR1=B11)
        if USE_NDSI_FOR_SNOW:
            ndsi = (per_band["B03"] - per_band["B11"])/np.maximum(per_band["B03"] + per_band["B11"], 1e-6)
            snow_mask |= (ndsi >= NDSI_THRESH) & (per_band["B03"] >= GREEN_MIN)
            # --- soft probability for blending near patchy edges ---
            # Use NDSI-derived probability; where SCL says “definitely snow”, force p=1
            # p_snow = ndsi_soft_probability(ndsi, center=NDSI_THRESH, halfwidth=0.10, use_ndsi_as_p=False)
            p_snow = ndsi_soft_probability(ndsi, center=0.3, halfwidth=0.10, use_ndsi_as_p=False)
            p_snow = np.where(snow_mask, 1.0, p_snow).astype("float32")
        else:
            # --- soft probability for blending near patchy edges ---
            # Use NDSI-derived probability; where SCL says “definitely snow”, force p=1
            ndsi = (per_band["B03"] - per_band["B11"])/np.maximum(per_band["B03"] + per_band["B11"], 1e-6)
            # p_snow = ndsi_soft_probability(ndsi, center=NDSI_THRESH, halfwidth=0.10, use_ndsi_as_p=False)
            p_snow = ndsi_soft_probability(ndsi, center=0.3, halfwidth=0.10, use_ndsi_as_p=False)
            p_snow = np.where(snow_mask, 1.0, p_snow).astype("float32")
            snow_mask |= (scl == SCL_SNOW_CLASS)
            
        # ---------------- Quality Assurance (QA) RASTERS: Encoding data quality information—not a physical variable ----------------
        # p_snow in [0...1]; NaN where invalid (clouds, etc.)
        qa_p_snow = np.where(mask_valid, p_snow.astype("float32"), np.float32(np.nan)) # the exact weight used in soft blending; 0 -> snow-free, 1 -> snow

        # Snow mask as uint8 with explicit invalid code (255). The binary mask used for hard switching (with explicit invalid)
        qa_snow_mask = np.full((H, W), 255, dtype=np.uint8)         # 255 = invalid
        qa_snow_mask[mask_valid & ~snow_mask] = 0                   # 0 = not snow
        qa_snow_mask[mask_valid &  snow_mask] = 1                   # 1 = snow

        # HARD choice map: Which coefficient set was applied at each pixel for hard products
        # 0 = invalid ; 1 = used snow-free coefficients ; 2 = used snow coefficients
        qa_choice_hard = np.zeros((H, W), dtype=np.uint8)
        qa_choice_hard[mask_valid & ~snow_mask] = 1
        qa_choice_hard[mask_valid &  snow_mask] = 2

        # SOFT choice map, binned from p_snow: A readable bin of the soft weight
        # 0 = invalid ; 1 = mostly snow-free (p<0.3) ; 2 = mixed (0.3...0.5) ; 3 = mostly snow (p>0.5)
        qa_choice_soft = np.zeros((H, W), dtype=np.uint8)
        valid_idx = mask_valid & np.isfinite(qa_p_snow)
        qa_choice_soft[valid_idx & (qa_p_snow < 0.3)]  = 1
        qa_choice_soft[valid_idx & (qa_p_snow >= 0.3) & (qa_p_snow <= 0.5)] = 2
        qa_choice_soft[valid_idx & (qa_p_snow > 0.5)]  = 3

        # export NDSI itself as QA: Useful to audit/tune thresholds (center and halfwidth)
        qa_ndsi = np.where(mask_valid, ndsi.astype("float32"), np.float32(np.nan))
        
        qa_prefix = os.path.join(out_dir, f"{date}_S2_QA")
        # p_snow (float32, NaN=invalid)
        write_coglike(qa_prefix + "_p_snow_qa.tif", qa_p_snow, dst_transform, dst_crs,
                      description="Snow probability from NDSI soft ramp (0...1)")

        # snow_mask (uint8, 0/1 with 255=invalid)
        write_coglike(qa_prefix + "_snow_mask_qa.tif", qa_snow_mask, dst_transform, dst_crs,
                      nodata=255, dtype="uint8",
                      description="Snow mask from SCL/NDSI (1=snow,0=no-snow,255=invalid)")

        # choice_hard (uint8, 0/1/2)
        write_coglike(qa_prefix + "_choice_hard_qa.tif", qa_choice_hard, dst_transform, dst_crs,
                      nodata=0, dtype="uint8",
                      description="Hard switch: 2=snow coeffs, 1=snow-free coeffs, 0=invalid")

        # choice_soft (uint8, 0...3)
        write_coglike(qa_prefix + "_choice_soft_qa.tif", qa_choice_soft, dst_transform, dst_crs,
                      nodata=0, dtype="uint8",
                      description="Soft blend bins: 3=mostly snow, 2=mixed, 1=mostly no-snow, 0=invalid")

        # NDSI (float32)
        write_coglike(qa_prefix + "_ndsi_qa.tif", qa_ndsi, dst_transform, dst_crs,
                      description="Normalized Difference Snow Index (float), NaN=invalid")

        # --- stack for NetCDF ---
        qa_p_snow_stack.append(qa_p_snow)
        qa_snow_mask_stack.append(qa_snow_mask)
        qa_choice_hard_stack.append(qa_choice_hard)
        qa_choice_soft_stack.append(qa_choice_soft)
        qa_ndsi_stack.append(qa_ndsi)
            
        # BRDF c-factor normalization per band
        # SZA/SAA from properties if available; otherwise approximate (S2 is near-nadir -> VZA≈0, RAA≈0)
        sza = item.properties.get("s2:mean_solar_zenith_angle") or item.properties.get("s2:mean_solar_zenith") \
              or item.properties.get("eo:sun_zenith") or item.properties.get("sun_zenith") or 35.0
        saa = item.properties.get("s2:mean_solar_azimuth_angle") or item.properties.get("s2:mean_solar_azimuth") \
              or item.properties.get("eo:sun_azimuth") or item.properties.get("sun_azimuth") or 180.0
        vza = item.properties.get("s2:mean_view_zenith_angle") or 0.0
        vaa = item.properties.get("s2:mean_view_azimuth_angle") or saa
        raa = abs((saa - vaa + 540) % 360 - 180)  # relative azimuth [0...180]

        # === NEW: Compute S2->(BSA/WSA/Blue-sky) at 20 m using MODIS BRDF (keep existing outputs unchanged) ===
        # Make a copy of the raw per-band reflectance, then apply topographic correction *just for this branch*
        bands_for_bsa = {}
        for b in per_band:
            bands_for_bsa[b] = topo_c_correction(per_band[b], SLOPE, ASPECT, sza_deg=sza, saa_deg=saa)
            bands_for_bsa[b] = np.where(mask_valid, bands_for_bsa[b], np.nan)  # apply cloud mask

        if USE_AOT: # Use Aerosol Optical Thickness (AOT) from Sentinel-2 to compute the diffuse fraction, fdiff.
            # Read AOT asset if available
            aot_href = item.assets.get("AOT").href if "AOT" in item.assets else None
            if aot_href is not None:
                with rio.open(aot_href) as ds_aot:
                    win = windows.from_bounds(*warped_aoi_bounds, transform=ds_aot.transform)
                    win = win.round_offsets().round_lengths()
                    aot_raw, a_tr, a_crs, _ = read_window(aot_href, win)

                # Reproject to target 20 m grid. S2 AOT is 10 m grid by default.
                aot_match = reproject_match(
                    aot_raw, a_tr, a_crs,
                    dst_transform, dst_crs,
                    (H, W),
                    resampling=Resampling.bilinear
                )[0]

                # Scale AOT to physical value (Sentinel-2 L2A AOT is typically scaled)
                # Check metadata; often scale factor is 0.001:
                # Used because Sentinel-2 L2A AOT is stored as: 
                # stored: 0-3000
                # physical AOT: 0-3
                # scale_factor = 0.001
                aot_scale = 0.001
                aot = aot_match.astype("float32") * aot_scale

                # Mask with validity mask
                aot = np.where(mask_valid, aot, np.nan)

                # Use a scene-representative AOT (e.g., mean over valid pixels)
                aot_scene = float(np.nanmean(aot))
            else: # Fallback if no AOT band: we could use a climatological constant/value
                aot_scene = 0.15  # a reasonable climatological constant for clean atmospheric conditions
            fdiff_sw = lookup_diffuse_fraction_from_lut(sza_deg=sza, aod_value=aot_scene)
        else: # Use Aerosol Optical Depth (AOD) from MODIS to compute the diffuse fraction, fdiff.
            # Convert date to "YYYYJJJ" for CERES AOD lookup
            # Example: 2022-06-29 -> "2022180"
            dt_obj = item.datetime
            yyyy = dt_obj.year
            doy = dt_obj.timetuple().tm_yday
            date_yyyyjjj = f"{yyyy}{doy:03d}"

            # Get AOD from CERES/MODIS time series
            aod_value = float(get_aod_static_data(aod_lookup_table_file, date_yyyyjjj))  # scalar

            # Compute diffuse fraction from LUT using SZA and AOD
            fdiff_sw = lookup_diffuse_fraction_from_lut(sza_deg=sza, aod_value=aod_value)

        # Per-band BSA/WSA reflectance at this date's SZA
        r_bsa, r_wsa = {}, {}
        for b in ["B02","B03","B04","B8A","B11","B12"]:
            r_bsa[b], r_wsa[b] = to_bsa_wsa_band(bands_for_bsa[b], 
                                                 b, 
                                                 sza_deg=sza, 
                                                 vza_deg=vza, 
                                                 raa_deg=raa)

        # --- NEW: Li et. al., 2018, Table-2 on BRDF-normalized bands (per-pixel snow logic) ---
        # fdiff = BLUE_SKY_DIFFUSE_FRACTION  # we set this value to 0.25 by default.

        # Compute BSA/WSA broadband for snow and snow-free, then hard/soft combine
        BROADBANDS = ("SW", "VIS", "NIR")
        bsa_bb_hard, bsa_bb_soft = {}, {}
        wsa_bb_hard, wsa_bb_soft = {}, {}
        blue_sky_bb_hard, blue_sky_bb_soft = {}, {}

        for bb in BROADBANDS:
            # Snow vs. snow-free broadband from BSA bands
            bsa_snow, bsa_snow_free = nb_to_bb_set(ALBEDO_COEFFS[bb], r_bsa)
            # Snow vs. snow-free broadband from WSA bands
            wsa_snow, wsa_snow_free = nb_to_bb_set(ALBEDO_COEFFS[bb], r_wsa)
            
            # Hard switch
            bsa_hard = np.where(snow_mask, bsa_snow, bsa_snow_free).astype("float32")
            wsa_hard = np.where(snow_mask, wsa_snow, wsa_snow_free).astype("float32")

            # # Soft blend. All 2 options work well equally the same way.
            # # Option 1
            # bsa_soft = (p_snow * bsa_snow + (1.0 - p_snow) * bsa_snow_free).astype("float32")
            # wsa_soft = (p_snow * wsa_snow + (1.0 - p_snow) * wsa_snow_free).astype("float32")
            # Option 2
            bsa_soft = soft_blend_albedo(p_snow, bsa_snow, bsa_snow_free)
            wsa_soft = soft_blend_albedo(p_snow, wsa_snow, wsa_snow_free)

            # ---- band-dependent diffuse fraction ----
            if bb == "SW":
                fdiff_bb = fdiff_sw          # LUT-based diffuse fraction for SW
            else:
                fdiff_bb = BLUE_SKY_DIFFUSE_FRACTION  # constant for VIS and NIR

            # # Blue-sky (linear mix of WSA/BSA). All 3 options work well equally the same way.
            # # Option 1
            # blue_sky_hard = (fdiff_bb * wsa_hard + (1.0 - fdiff_bb) * bsa_hard).astype("float32")
            # blue_sky_soft = (fdiff_bb * wsa_soft + (1.0 - fdiff_bb) * bsa_soft).astype("float32")
            # # Option 2
            blue_sky_hard = blue_sky_from_bsa_wsa(bsa_hard, wsa_hard, fdiff_bb) # already clipped
            blue_sky_soft = blue_sky_from_bsa_wsa(bsa_soft, wsa_soft, fdiff_bb) # already clipped
            # # Option 3
            # blue_sky_hard = np.where(snow_mask, blue_sky_snow, blue_sky_snow_free).astype("float32")
            # blue_sky_soft = soft_blend_albedo(p_snow, blue_sky_snow, blue_sky_snow_free)
        
            # Clip to [0,1]
            for broadband_albedo in (bsa_hard, wsa_hard, blue_sky_hard, bsa_soft, wsa_soft, blue_sky_soft):
                np.clip(broadband_albedo, 0.0, 1.0, out=broadband_albedo)

            # stash
            bsa_bb_hard[bb], bsa_bb_soft[bb] = bsa_hard, bsa_soft
            wsa_bb_hard[bb], wsa_bb_soft[bb] = wsa_hard, wsa_soft
            blue_sky_bb_hard[bb], blue_sky_bb_soft[bb] = blue_sky_hard, blue_sky_soft

            
        # Save (COG-like) — uses existing write_coglike() function
        base = os.path.join(out_dir, f"{date}_S2")
        # SW
        write_coglike(base + "_BSA20m_SW_hard.tif",  
                      bsa_bb_hard["SW"],  
                      dst_transform, dst_crs,
                      description="BRDF-normalized BSA SW albedo (Li18 Table2), hard snow switch")
        write_coglike(base + "_BSA20m_SW_soft.tif",  
                      bsa_bb_soft["SW"],  
                      dst_transform, dst_crs,
                      description="BRDF-normalized BSA SW albedo (Li18 Table2), soft snow blend")

        write_coglike(base + "_WSA20m_SW_hard.tif",  
                      wsa_bb_hard["SW"],  
                      dst_transform, dst_crs,
                      description="BRDF-normalized WSA SW albedo (Li18 Table2), hard snow switch")
        write_coglike(base + "_WSA20m_SW_soft.tif",  
                      wsa_bb_soft["SW"],  
                      dst_transform, dst_crs,
                      description="BRDF-normalized WSA SW albedo (Li18 Table2), soft snow blend")

        write_coglike(base + "_BLUE20m_SW_hard.tif", 
                      blue_sky_bb_hard["SW"], 
                      dst_transform, dst_crs,
                      description=f"Blue-sky SW albedo (Li18 Table2), hard (fdiff={fdiff_bb})")
        write_coglike(base + "_BLUE20m_SW_soft.tif", 
                      blue_sky_bb_hard["SW"], 
                      dst_transform, dst_crs,
                      description=f"Blue-sky SW albedo (Li18 Table2), soft (fdiff={fdiff_bb})")

        # VIS
        write_coglike(base + "_BSA20m_VIS_hard.tif",  
                      bsa_bb_hard["VIS"],  
                      dst_transform, dst_crs,
                      description="BRDF-normalized BSA VIS albedo (Li18 Table2), hard")
        write_coglike(base + "_BSA20m_VIS_soft.tif",  
                      bsa_bb_soft["VIS"],  
                      dst_transform, dst_crs,
                      description="BRDF-normalized BSA VIS albedo (Li18 Table2), soft")

        write_coglike(base + "_WSA20m_VIS_hard.tif",  
                      wsa_bb_hard["VIS"],  
                      dst_transform, dst_crs,
                      description="BRDF-normalized WSA VIS albedo (Li18 Table2), hard")
        write_coglike(base + "_WSA20m_VIS_soft.tif",  
                      wsa_bb_soft["VIS"],  
                      dst_transform, dst_crs,
                      description="BRDF-normalized WSA VIS albedo (Li18 Table2), soft")

        write_coglike(base + "_BLUE20m_VIS_hard.tif", 
                      blue_sky_bb_hard["VIS"], 
                      dst_transform, dst_crs,
                      description=f"Blue-sky VIS albedo (Li18 Table2), hard (fdiff={BLUE_SKY_DIFFUSE_FRACTION})")
        write_coglike(base + "_BLUE20m_VIS_soft.tif", 
                      blue_sky_bb_hard["VIS"], 
                      dst_transform, dst_crs,
                      description=f"Blue-sky VIS albedo (Li18 Table2), soft (fdiff={BLUE_SKY_DIFFUSE_FRACTION})")

        # NIR
        write_coglike(base + "_BSA20m_NIR_hard.tif",  
                      bsa_bb_hard["NIR"],  
                      dst_transform, dst_crs,
                      description="BRDF-normalized BSA NIR albedo (Li18 Table2), hard")
        write_coglike(base + "_BSA20m_NIR_soft.tif",  
                      bsa_bb_soft["NIR"],  
                      dst_transform, dst_crs,
                      description="BRDF-normalized BSA NIR albedo (Li18 Table2), soft")

        write_coglike(base + "_WSA20m_NIR_hard.tif",  
                      wsa_bb_hard["NIR"],  
                      dst_transform, dst_crs,
                      description="BRDF-normalized WSA NIR albedo (Li18 Table2), hard")
        write_coglike(base + "_WSA20m_NIR_soft.tif",  
                      wsa_bb_soft["NIR"],  
                      dst_transform, dst_crs,
                      description="BRDF-normalized WSA NIR albedo (Li18 Table2), soft")

        write_coglike(base + "_BLUE20m_NIR_hard.tif", 
                      blue_sky_bb_hard["NIR"], 
                      dst_transform, dst_crs,
                      description=f"Blue-sky NIR albedo (Li18 Table2), hard (fdiff={BLUE_SKY_DIFFUSE_FRACTION})")
        write_coglike(base + "_BLUE20m_NIR_soft.tif", 
                      blue_sky_bb_hard["NIR"], 
                      dst_transform, dst_crs,
                      description=f"Blue-sky NIR albedo (Li18 Table2), soft (fdiff={BLUE_SKY_DIFFUSE_FRACTION})")

        
        # # Variants
        # bsa_keep = bsa_sw.copy()                         # clouds already NaN from mask above
        # wsa_keep = wsa_sw.copy()
        # blue_keep = blue_sw.copy()

        # Compute snow over only snow-covered pixels: snow-only (non-snow masked). Otherwise, the products are "keep-snow" (only clouds masked)
        # Hard switch
        bsa_snow_only_albedo_hard = np.where(snow_mask, bsa_bb_hard["SW"], np.nan)   # snow-only
        wsa_snow_only_albedo_hard = np.where(snow_mask, wsa_bb_hard["SW"], np.nan)
        blue_sky_snow_only_albedo_hard = np.where(snow_mask, blue_sky_bb_hard["SW"], np.nan)
        # Soft blend
        bsa_snow_only_albedo_soft = np.where(snow_mask, bsa_bb_soft["SW"], np.nan)   # snow-only
        wsa_snow_only_albedo_soft = np.where(snow_mask, wsa_bb_soft["SW"], np.nan)
        blue_sky_snow_only_albedo_soft = np.where(snow_mask, blue_sky_bb_soft["SW"], np.nan)

        # Save (COG-like) — uses existing write_coglike() function
        base_new = os.path.join(out_dir, f"{date}_S2")  # prefix for new files
        write_coglike(base_new + "_BSA20m_snowOnly_hard.tif", 
                      bsa_snow_only_albedo_hard, 
                      dst_transform, dst_crs,
                      description="Sentinel-2A black-sky broadband shortwave albedo (20 m), snow only hard")
        write_coglike(base_new + "_BSA20m_snowOnly_soft.tif", 
                      bsa_snow_only_albedo_soft, 
                      dst_transform, dst_crs,
                      description="Sentinel-2A black-sky broadband shortwave albedo (20 m), snow only soft")

        write_coglike(base_new + "_WSA20m_snowOnly_hard.tif", 
                      wsa_snow_only_albedo_hard, 
                      dst_transform, dst_crs,
                      description="Sentinel-2A white-sky broadband shortwave albedo (20 m), snow only hard")
        write_coglike(base_new + "_WSA20m_snowOnly_soft.tif", 
                      wsa_snow_only_albedo_soft, 
                      dst_transform, dst_crs,
                      description="Sentinel-2A white-sky broadband shortwave albedo (20 m), snow only soft")

        write_coglike(base_new + "_BLUE20m_snowOnly_hard.tif", 
                      blue_sky_snow_only_albedo_hard, 
                      dst_transform, dst_crs,
                      description=f"Sentinel-2A blue-sky broadband shortwave albedo (20 m), snow only hard (fdiff={fdiff_bb})")
        write_coglike(base_new + "_BLUE20m_snowOnly_hard_soft.tif", 
                      blue_sky_snow_only_albedo_soft, 
                      dst_transform, dst_crs,
                      description=f"Sentinel-2A blue-sky broadband shortwave albedo (20 m), snow only soft (fdiff={fdiff_bb})")

        
        # --- stack for NetCDF ---
        bsa_keep_stack_hard.append(bsa_bb_hard["SW"])
        bsa_keep_stack_soft.append(bsa_bb_soft["SW"])
        wsa_keep_stack_hard.append(wsa_bb_hard["SW"])
        wsa_keep_stack_soft.append(wsa_bb_soft["SW"])
        blue_sky_keep_stack_hard.append(blue_sky_bb_hard["SW"])
        blue_sky_keep_stack_soft.append(blue_sky_bb_soft["SW"])
        
        bsa_snow_stack_hard.append(bsa_snow_only_albedo_hard)
        bsa_snow_stack_soft.append(bsa_snow_only_albedo_soft)
        wsa_snow_stack_hard.append(wsa_snow_only_albedo_hard)
        wsa_snow_stack_soft.append(wsa_snow_only_albedo_soft)
        blue_sky_snow_stack_hard.append(blue_sky_snow_only_albedo_hard)
        blue_sky_snow_stack_soft.append(blue_sky_snow_only_albedo_soft)
        # === END NEW ===

        for b in per_band:
            cfac = brdf_c_factor(BRDF_MAP[b], sza, vza, raa)
            per_band[b] = per_band[b] * cfac  # normalize to ref geometry

        # Topographic C-correction per band
        for b in per_band:
            per_band[b] = topo_c_correction(per_band[b], SLOPE, ASPECT, sza_deg=sza, saa_deg=saa)

        # Cloud mask applied
        for b in per_band:
            per_band[b] = np.where(mask_valid, per_band[b], np.nan)

        # --- NEW: per-pixel Li et al. (2018) broadband with snow vs snow-free sets ---
        # Uses topography-corrected + BRDF-normalized reflectance already in per_band (and masked for clouds)
        # Hard switch (boolean mask) and soft blend (probability) for SW, VIS, NIR.

        # Compute snow and no-snow versions for each broadband
        sw_snow,  sw_now_free  = nb_to_bb_set(ALBEDO_COEFFS["SW"],  per_band)
        vis_snow, vis_now_free = nb_to_bb_set(ALBEDO_COEFFS["VIS"], per_band)
        nir_snow, nir_now_free = nb_to_bb_set(ALBEDO_COEFFS["NIR"], per_band)

        # HARD SWITCH by snow mask
        sw_hard  = np.where(snow_mask, sw_snow,  sw_now_free).astype("float32")
        vis_hard = np.where(snow_mask, vis_snow, vis_now_free).astype("float32")
        nir_hard = np.where(snow_mask, nir_snow, nir_now_free).astype("float32")

        # SOFT BLEND by p_snow
        sw_soft  = (p_snow * sw_snow  + (1.0 - p_snow) * sw_now_free).astype("float32")
        vis_soft = (p_snow * vis_snow + (1.0 - p_snow) * vis_now_free).astype("float32")
        nir_soft = (p_snow * nir_snow + (1.0 - p_snow) * nir_now_free).astype("float32")

        # Clip to [0,1] and keep NaNs where mask already created them
        for a in (sw_hard, vis_hard, nir_hard, sw_soft, vis_soft, nir_soft):
            np.clip(a, 0.0, 1.0, out=a)
    
        # Variants: keep-snow (only clouds masked) and snow-only (non-snow masked)
        # hard switch version
        sw_keep_hard = sw_hard.copy()
        sw_snow_hard = np.where(snow_mask, sw_hard, np.nan)

        vis_keep_hard = vis_hard.copy()
        vis_snow_hard = np.where(snow_mask, vis_hard, np.nan)

        nir_keep_hard = nir_hard.copy()
        nir_snow_hard = np.where(snow_mask, nir_hard, np.nan)
        
        # soft blend version
        sw_keep_soft = sw_soft.copy()
        sw_snow_soft = np.where(snow_mask, sw_soft, np.nan)

        vis_keep_soft = vis_soft.copy()
        vis_snow_soft = np.where(snow_mask, vis_soft, np.nan)

        nir_keep_soft = nir_soft.copy()
        nir_snow_soft = np.where(snow_mask, nir_soft, np.nan)

        # Write GeoTIFFs (COG-like)
        # hard switch and soft blend versions
        prefix = os.path.join(out_dir, f"{date}")
        write_coglike(prefix + "_albedoSW_hard.tif", sw_hard,  dst_transform, dst_crs, description="SW albedo (Li18 Table2), hard snow switch")
        write_coglike(prefix + "_albedoSW_soft.tif", sw_soft,  dst_transform, dst_crs, description="SW albedo (Li18 Table2), soft snow blend")
        write_coglike(prefix + "_albedoVIS_hard.tif", vis_hard, dst_transform, dst_crs, description="VIS albedo (Li18 Table2), hard snow switch")
        write_coglike(prefix + "_albedoVIS_soft.tif", vis_soft, dst_transform, dst_crs, description="VIS albedo (Li18 Table2), soft snow blend")
        write_coglike(prefix + "_albedoNIR_hard.tif", nir_hard, dst_transform, dst_crs, description="NIR albedo (Li18 Table2), hard snow switch")
        write_coglike(prefix + "_albedoNIR_soft.tif", nir_soft, dst_transform, dst_crs, description="NIR albedo (Li18 Table2), soft snow blend")

        # keep snow vs snow-only: hard switch version
        write_coglike(prefix + "_albedoSW_keepAll_hard.tif", sw_keep_hard, dst_transform, dst_crs, description="SW albedo (Li18 Table2), hard keep snow")
        write_coglike(prefix + "_albedoSW_snowOnly_hard.tif", sw_snow_hard, dst_transform, dst_crs, description="SW albedo (Li18 Table2), hard snow only")
        write_coglike(prefix + "_albedoVIS_keepAll_hard.tif", vis_keep_hard, dst_transform, dst_crs, description="VIS albedo (Li18 Table2), hard keep snow")
        write_coglike(prefix + "_albedoVIS_snowOnly_hard.tif", vis_snow_hard, dst_transform, dst_crs, description="VIS albedo (Li18 Table2), hard snow only")
        write_coglike(prefix + "_albedoNIR_keepAll_hard.tif", nir_keep_hard, dst_transform, dst_crs, description="NIR albedo (Li18 Table2), hard keep snow")
        write_coglike(prefix + "_albedoNIR_snowOnly_hard.tif", nir_snow_hard, dst_transform, dst_crs, description="NIR albedo (Li18 Table2), hard snow only")

        # keep snow vs snow-only: soft blend version
        write_coglike(prefix + "_albedoSW_keepAll_soft.tif", sw_keep_soft, dst_transform, dst_crs, description="SW albedo (Li18 Table2), soft keep snow")
        write_coglike(prefix + "_albedoSW_snowOnly_soft.tif", sw_snow_soft, dst_transform, dst_crs, description="SW albedo (Li18 Table2), soft snow only")
        write_coglike(prefix + "_albedoVIS_keepAll_soft.tif", vis_keep_soft, dst_transform, dst_crs, description="VIS albedo (Li18 Table2), soft keep snow")
        write_coglike(prefix + "_albedoVIS_snowOnly_soft.tif", vis_snow_soft, dst_transform, dst_crs, description="VIS albedo (Li18 Table2), soft snow only")
        write_coglike(prefix + "_albedoNIR_keepAll_soft.tif", nir_keep_soft, dst_transform, dst_crs, description="NIR albedo (Li18 Table2), soft keep snow")
        write_coglike(prefix + "_albedoNIR_snowOnly_soft.tif", nir_snow_soft, dst_transform, dst_crs, description="NIR albedo (Li18 Table2), soft snow only")

        # Build stacks for NetCDF
        # hard switch version
        sw_keep_stack_hard.append(sw_keep_hard)
        sw_snow_stack_hard.append(sw_snow_hard)
        vis_keep_stack_hard.append(vis_keep_hard)
        vis_snow_stack_hard.append(vis_snow_hard)
        nir_keep_stack_hard.append(nir_keep_hard)
        nir_snow_stack_hard.append(nir_snow_hard)
        
        # soft blend version
        sw_keep_stack_soft.append(sw_keep_soft)
        sw_snow_stack_soft.append(sw_snow_soft)
        vis_keep_stack_soft.append(vis_keep_soft)
        vis_snow_stack_soft.append(vis_snow_soft)
        nir_keep_stack_soft.append(nir_keep_soft)
        nir_snow_stack_soft.append(nir_snow_soft)
        
        # time_index.append(np.datetime64(item.datetime))
        dt_utc = item.datetime.astimezone(timezone.utc).replace(tzinfo=None)  # make it naive UTC
        time_index.append(np.datetime64(dt_utc, 'ns'))  # store as numpy datetime64

        # Record
        records.append({
            "date": date,
            "item_id": item.id,
            # hard switch version
            "sw_keep_hard": os.path.basename(prefix + "_albedoSW_keepAll_hard.tif"),
            "sw_snow_hard": os.path.basename(prefix + "_albedoSW_snowOnly_hard.tif"),
            "vis_keep_hard": os.path.basename(prefix + "_albedoVIS_keepAll_hard.tif"),
            "vis_snow_hard": os.path.basename(prefix + "_albedoVIS_snowOnly_hard.tif"),
            "nir_keep_hard": os.path.basename(prefix + "_albedoNIR_keepAll_hard.tif"),
            "nir_snow_hard": os.path.basename(prefix + "_albedoNIR_snowOnly_hard.tif"),
            # soft blend version
            "sw_keep_soft": os.path.basename(prefix + "_albedoSW_keepAll_soft.tif"),
            "sw_snow_soft": os.path.basename(prefix + "_albedoSW_snowOnly_soft.tif"),
            "vis_keep_soft": os.path.basename(prefix + "_albedoVIS_keepAll_soft.tif"),
            "vis_snow_soft": os.path.basename(prefix + "_albedoVIS_snowOnly_soft.tif"),
            "nir_keep_soft": os.path.basename(prefix + "_albedoNIR_keepAll_soft.tif"),
            "nir_snow_soft": os.path.basename(prefix + "_albedoNIR_snowOnly_soft.tif"),
        })

    # Save index CSV
    with open(os.path.join(out_dir, "index.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    # Write NetCDF stack if xarray present
    if XR_AVAILABLE:
        ds = xr.Dataset({
            # hard switch version
            "albedo_sw_keep_hard":  stack_to_da(sw_keep_stack_hard,  "albedo_sw_keep_hard"),
            "albedo_sw_snow_hard":  stack_to_da(sw_snow_stack_hard,  "albedo_sw_snow_hard"),
            "albedo_vis_keep_hard": stack_to_da(vis_keep_stack_hard, "albedo_vis_keep_hard"),
            "albedo_vis_snow_hard": stack_to_da(vis_snow_stack_hard, "albedo_vis_snow_hard"),
            "albedo_nir_keep_hard": stack_to_da(nir_keep_stack_hard, "albedo_nir_keep_hard"),
            "albedo_nir_snow_hard": stack_to_da(nir_snow_stack_hard, "albedo_nir_snow_hard"),

            # soft blend version
            "albedo_sw_keep_soft":  stack_to_da(sw_keep_stack_soft,  "albedo_sw_keep_soft"),
            "albedo_sw_snow_soft":  stack_to_da(sw_snow_stack_soft,  "albedo_sw_snow_soft"),
            "albedo_vis_keep_soft": stack_to_da(vis_keep_stack_soft, "albedo_vis_keep_soft"),
            "albedo_vis_snow_soft": stack_to_da(vis_snow_stack_soft, "albedo_vis_snow_soft"),
            "albedo_nir_keep_soft": stack_to_da(nir_keep_stack_soft, "albedo_nir_keep_soft"),
            "albedo_nir_snow_soft": stack_to_da(nir_snow_stack_soft, "albedo_nir_snow_soft"),
            
            # NEW: BRDF-derived broadband shortwave albedos (keep-all)
            "bsa_sw_keep_hard": stack_to_da(bsa_keep_stack_hard, "bsa_sw_keep_hard"),
            "bsa_sw_keep_soft": stack_to_da(bsa_keep_stack_soft, "bsa_sw_keep_soft"),
            "wsa_sw_keep_hard": stack_to_da(wsa_keep_stack_hard, "wsa_sw_keep_hard"),
            "wsa_sw_keep_soft": stack_to_da(wsa_keep_stack_soft, "wsa_sw_keep_soft"),
            "blue_sw_keep_hard": stack_to_da(blue_sky_keep_stack_hard, "blue_sky_sw_keep_hard"),
            "blue_sw_keep_soft": stack_to_da(blue_sky_keep_stack_soft, "blue_sky_sw_keep_soft"),
            
            # NEW: BRDF-derived broadband shortwave albedos (snow-only)
            "bsa_sw_snow_hard": stack_to_da(bsa_snow_stack_hard, "bsa_sw_snow_hard"),
            "bsa_sw_snow_soft": stack_to_da(bsa_snow_stack_soft, "bsa_sw_snow_soft"),
            "wsa_sw_snow_hard": stack_to_da(wsa_snow_stack_hard, "wsa_sw_snow_hard"),
            "wsa_sw_snow_soft": stack_to_da(wsa_snow_stack_soft, "wsa_sw_snow_soft"),
            "blue_sw_snow_hard": stack_to_da(blue_sky_snow_stack_hard, "blue_sky_sw_snow_hard"),
            "blue_sw_snow_soft": stack_to_da(blue_sky_snow_stack_soft, "blue_sky_sw_snow_soft"),
            
            # NEW: QA stacks
            "qa_p_snow_stack": stack_to_da(qa_p_snow_stack, "qa_p_snow_stack"),
            "qa_snow_mask_stack": stack_to_da(qa_snow_mask_stack, "qa_snow_mask_stack"),
            "qa_choice_hard_stack": stack_to_da(qa_choice_hard_stack, "qa_choice_hard_stack"),
            "qa_choice_soft_stack": stack_to_da(qa_choice_soft_stack, "qa_choice_soft_stack"),
            "qa_ndsi_stack": stack_to_da(qa_ndsi_stack, "qa_ndsi_stack"),
            
        })

        # add a few helpful attributes
        ds.attrs.update({
            "note": "Includes BRDF-derived BSA/WSA/Blue-sky broadband SW albedo at 20 m (keep-snow and snow-only).",
            "default blue_sky_diffuse_fraction": str(BLUE_SKY_DIFFUSE_FRACTION),
            "brdf_model": "RTLSR (RossThick/LiSparse-Reciprocal), HLS global weights",
            "nb_to_bb_formula_sw_snow_free": "0.2688*B02+0.0362*B03+0.1501*B04+0.3045*B8A+0.1644*B11+0.0356*B12-0.0049",
        })

        nc_path = os.path.join(out_dir, "albedo_stack3.nc")
        ds.to_netcdf(nc_path)
        print(f"Wrote NetCDF stack: {nc_path}")

    print("Done!!!")
    
    

    
    