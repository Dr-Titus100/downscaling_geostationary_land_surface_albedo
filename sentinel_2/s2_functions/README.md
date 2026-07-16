# Sentinel-2 and TSI Helper Scripts

This directory contains reusable scripts for Sentinel-2 20 m albedo production, terrain correction, MODIS BRDF fusion, ARM Total Sky Imager matching, and `.cdf` inspection.

## Contents

| File | Main responsibility |
|---|---|
| `s2_20m_download_final.py` | Comprehensive Sentinel-2 L2A retrieval and 20 m SW, VIS, NIR, BSA, WSA, blue-sky, snow, and QA workflow. |
| `s2_20m_albedo_topocorr_brdf_fusion.py` | Alternate terrain-corrected shortwave albedo and MODIS BSA/WSA fusion workflow. |
| `tsi_functions.py` | TSI filtering, temporal matching, image access, cloud diagnostics, and spatial cloud-fraction helpers. |
| `auto_open_cdf.py` | Utility for identifying and inventorying NetCDF-style or NASA CDF files. |
| `packages.py` | Shared import manifest used by scripts in this directory. |

These are research modules with module-level configuration rather than an installed Python package.

## Setup and imports

```bash
conda activate sail_env
```

For notebook imports:

```python
from pathlib import Path
import sys

repo_root = Path("/path/to/downscaling_geostationary_land_surface_albedo")
helper_dir = repo_root / "sentinel_2" / "s2_functions"
sys.path.insert(0, str(helper_dir))

from s2_20m_download_final import *
from tsi_functions import match_tsi_to_s2
```

The modules use the local `packages.py`. Put this directory before other project directories that also contain a file named `packages.py`.

Some modules read the configured AOI at import time. Correct their paths before importing.

## `packages.py`

The shared manifest centralizes standard, geospatial, remote-sensing, scientific, plotting, STAC, and machine-learning imports needed by the helper scripts. It does not install dependencies. Use `environment.yml` to create the environment and add optional packages only when the selected workflow needs them.

## `s2_20m_download_final.py`

### Purpose

This is the principal Sentinel-2 albedo workflow. It supports:

- Planetary Computer Sentinel-2 L2A search;
- AOI clipping and 20 m grid construction;
- reflectance scaling and processing-baseline harmonization;
- SCL and cloud masking;
- NDSI- or SCL-based snow selection;
- SW, VIS, and NIR broadband albedo;
- BRDF normalization;
- BSA, WSA, and blue-sky products;
- aerosol-based diffuse fractions;
- daily GeoTIFFs and QA products; and
- stacked outputs and metadata summaries.

### Configuration

Review the active values for:

```text
shapefile_path
out_dir
time_of_interest
TARGET_RES
USE_NDSI_FOR_SNOW
NDSI_THRESH
GREEN_MIN
SCL_MASK_CLASSES
CUTOFF
HARMONIZE_BANDS
diffuse_skylight_ratio_lookup
USE_AOT
```

The target resolution is currently 20 m. Confirm the date interval before every run.

### Inputs

Common Sentinel-2 assets include B02, B03, B04, B8A, B11, B12, SCL, AOT, and viewing/solar metadata. Additional bands may support QA or RGB visualization. Do not treat categorical layers as continuous reflectance.

### Baseline harmonization

The script contains a processing-baseline cutoff and a list of reflectance bands to harmonize. Verify item metadata and do not apply reflectance offsets to SCL or QA layers.

### Snow and cloud handling

The active SCL exclusions are configurable. Snow can be identified using NDSI and a minimum green reflectance or by SCL class 11. The script tracks hard and soft choices and creates mask and probability QA products.

### Broadband and BRDF products

The script stores separate snow and snow-free coefficients for SW, VIS, and NIR. Reflectance must be scaled to the expected range before applying them.

BRDF normalization uses configured reference geometry and can produce directional, BSA, WSA, and blue-sky variants. Record the exact product definition in output metadata.

### Diffuse fraction

Blue-sky albedo uses the shortwave lookup table in:

```text
../../GOES-Modis-Data-Preprocessing-main/sw_lut.csv
```

Confirm that aerosol and solar-zenith values fall within the table domain.

### Running

The script is commonly imported by `../s2_20m_download_final.ipynb`. Inspect top-level executable sections before running it directly because research versions may perform work during import.

## `s2_20m_albedo_topocorr_brdf_fusion.py`

### Purpose

This alternate workflow:

1. reads and projects the AOI;
2. queries Sentinel-2 L2A;
3. creates reflectance and SCL stacks on a 20 m grid;
4. masks clouds and shadows while retaining snow;
5. loads Copernicus DEM GLO-30;
6. calculates slope and aspect;
7. applies SCS+C terrain correction;
8. calculates nine-band shortwave albedo; and
9. sharpens MODIS BSA and WSA to 20 m using Sentinel-2 spatial ratios.

### Configuration

Review:

```text
shapefile_path
DATE_RANGE
EPSG_UTM
CLOUD_MAX
OUT_DIR
OUT_DIR2
NTB_WEIGHTS
REFL_BANDS
BAD_SCL
MCD_COLLECTION
MCD_ASSETS
MCD_RES_NATIVE
RATIO_EPS
```

### Band set

The narrow-to-broadband calculation uses:

```text
B02, B03, B04, B05, B06, B07, B08, B11, B12
```

Preserve the association between each coefficient and band.

### Terrain and fusion validation

Check DEM units and CRS, slope/aspect conventions, solar azimuth, illumination near zero, correction behavior in shadow, MODIS acquisition dates, ratio extremes, and output artifacts.

Typical outputs include terrain-corrected keep-snow and snow-only albedo and fused BSA/WSA variants.

## `tsi_functions.py`

### Configuration

Update the East River shapefile path before import. Verify all distances and radii are expressed in the CRS units expected by each helper.

### Clear-sky filtering

The module reads TSI variables such as:

```text
time
percent_thin
percent_opaque
```

It removes fill values, applies thresholds, and returns valid dates. Check whether the variables are percentages and pass thresholds explicitly.

### Temporal matching

`match_tsi_to_s2(...)` finds TSI files from the Sentinel-2 UTC date, selects valid cloud observations, and chooses the nearest timestamp. The documented maximum-gap behavior should be checked in the active code and enforced by the calling workflow when required.

### Visualization and spatial helpers

The module also supports signed Sentinel-2 image access, RGB stretching, TSI time-series plots, coordinate conversion, circular masks, asset reads, and cloud-fraction comparison.

## `auto_open_cdf.py`

Run:

```bash
python sentinel_2/s2_functions/auto_open_cdf.py /path/to/file.cdf
```

The utility detects common NetCDF and NASA CDF formats, tries compatible readers, and prints dimensions, variables, shapes, data types, and previews. Install SpacePy separately only when NASA CDF support is required.

## Validation checklist

Before a full run, verify:

- AOI file and CRS;
- STAC item IDs and signed assets;
- target CRS, transform, and 20 m resolution;
- reflectance scale and harmonization;
- continuous versus categorical resampling;
- SCL, cloud, and snow rules;
- broadband coefficient and band order;
- BRDF angle units and conventions;
- aerosol lookup coverage;
- DEM alignment and terrain correction;
- TSI units, fill values, thresholds, UTC times, and maximum time gap; and
- output names, units, masks, and metadata.

## Common problems

### Import fails at AOI loading

Correct the module-level shapefile path before import.

### Planetary Computer assets return authorization errors

Re-sign the item or asset.

### A band or asset key is missing

Inspect the STAC asset names and update the mapping without confusing B08 and B8A.

### Albedo contains extreme values

Check scaling, harmonization, masks, BRDF denominators, coefficients, band order, and aerosol lookup.

### TSI matching gives unexpected results

Normalize timestamps to UTC, pass explicit cloud thresholds, and enforce an explicit temporal tolerance.
