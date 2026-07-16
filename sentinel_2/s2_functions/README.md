# Sentinel-2 and TSI Helper Scripts

This directory contains reusable scripts for Sentinel-2 20 m albedo production, terrain correction, BRDF fusion, ARM Total Sky Imager matching, and `.cdf` format inspection.

## Contents

| File | Main responsibility |
|---|---|
| `s2_20m_download_final.py` | Comprehensive Sentinel-2 L2A retrieval and 20 m SW/VIS/NIR, BSA/WSA, blue-sky, snow, and QA workflow. |
| `s2_20m_albedo_topocorr_brdf_fusion.py` | Alternate terrain-corrected shortwave albedo and MODIS BRDF-fusion workflow. |
| `tsi_functions.py` | TSI clear-sky filtering, TSI-to-Sentinel temporal matching, image access, cloud diagnostics, and spatial cloud-fraction helpers. |
| `auto_open_cdf.py` | Command-line utility that distinguishes NetCDF-style `.cdf` files from NASA CDF and prints an inventory. |

These files are research scripts with module-level configuration. They are not installed as a Python package.

## Setup

Activate the environment from the repository root:

```bash
conda activate sail_env
```

For notebook imports:

```python
from pathlib import Path
import sys

repo_root = Path("/path/to/downscaling_geostationary_land_surface_albedo")
sys.path.append(str(repo_root / "sentinel_2" / "s2_functions"))
```

Then import the required module:

```python
from s2_20m_download_final import *
from tsi_functions import match_tsi_to_s2
```

Be aware that `s2_20m_download_final.py` and `tsi_functions.py` read the configured AOI shapefile when imported. Correct the path first.

## `s2_20m_download_final.py`

### Purpose

This script implements the most complete Sentinel-2 albedo workflow in the repository. It is designed for East River scenes but can be adapted to another AOI by changing the geometry, projected CRS, target resolution, coefficients, masks, and ancillary data.

### User configuration

Review these settings near the top of the file:

```text
shapefile_path
out_dir
time_of_interest
TARGET_RES
USE_NDSI_FOR_SNOW
NDSI_THRESH
GREEN_MIN
USE_MODIS_FALLBACK_FOR_VIS_NIR
BLUE_SKY_DIFFUSE_FRACTION
SCL_MASK_CLASSES
CUTOFF
HARMONIZE_BANDS
diffuse_skylight_ratio_lookup
USE_AOT
```

The current target resolution is 20 m. The active date range in the file may cover only a subset of the full project period, so confirm it before running.

### Input bands and products

The workflow uses Sentinel-2 L2A assets that include:

- B02 blue;
- B03 green;
- B04 red;
- B8A narrow near-infrared;
- B11 SWIR1;
- B12 SWIR2;
- SCL scene classification;
- AOT aerosol optical thickness when enabled; and
- acquisition/view/sun metadata from the STAC item.

Additional bands can be used for QA or RGB display. B10 is intentionally excluded from surface-albedo calculations.

### Baseline harmonization

The script defines January 25, 2022 as the processing-baseline cutoff and applies harmonization to specified reflectance bands. Confirm the item's processing metadata rather than relying only on acquisition date when applying the workflow to newly processed data.

### Scene classification mask

The active `SCL_MASK_CLASSES` defines which pixels are excluded. This set is configurable and differs from the alternate terrain-correction script.

Categorical SCL data must be resampled with nearest-neighbor. Continuous reflectance can be resampled with bilinear interpolation.

### Snow choice

When `USE_NDSI_FOR_SNOW=True`, snow is identified with an NDSI threshold and minimum green reflectance. Otherwise, SCL class 11 is used.

The script tracks hard and soft choices and creates QA arrays such as:

- snow probability;
- snow mask;
- hard choice;
- soft choice; and
- NDSI.

### Albedo coefficients

The script stores separate Li et al. coefficient sets for snow and snow-free surfaces for:

- SW: shortwave;
- VIS: visible; and
- NIR: near-infrared.

Each formula includes a constant and selected Sentinel-2 bands. Verify that reflectance is scaled to `0-1` before applying the formula.

### BRDF normalization

The script maps Sentinel-2 bands to MODIS-equivalent BRDF coefficient groups and applies a c-factor approach using configured reference geometry:

```text
reference SZA = 45 degrees
reference VZA = 0 degrees
reference RAA = 0 degrees
```

The workflow can produce black-sky, white-sky, and blue-sky albedo. Record whether an output represents observed directional reflectance, normalized reflectance, BSA, WSA, or blue-sky albedo.

### Diffuse fraction

The blue-sky calculation uses the shortwave lookup table at:

```text
GOES-Modis-Data-Preprocessing-main/sw_lut.csv
```

When `USE_AOT=True`, Sentinel-2 AOT contributes to the lookup. If another aerosol source is selected, configure the source and matching method explicitly.

### Output collections

The script initializes time-indexed collections for:

- SW, VIS, and NIR keep-all and snow-only variants;
- hard and soft snow choices;
- BSA, WSA, and blue-sky keep-all variants;
- BSA, WSA, and blue-sky snow-only variants; and
- QA products.

It can write daily rasters and build stacked outputs. Inspect active output cells or function calls before a full run because research versions may contain optional or commented export paths.

### Running

The file is commonly imported from `../s2_20m_download_final.ipynb`, which provides interactive control and plotting. If running it directly, first inspect whether all executable workflow sections are guarded by `if __name__ == "__main__":` or run at import time.

For production use, prefer an explicit driver script or notebook that records all settings and selected item IDs.

## `s2_20m_albedo_topocorr_brdf_fusion.py`

### Purpose

This script creates terrain-corrected Sentinel-2 shortwave albedo and fuses it with MODIS MCD43A3 black-sky and white-sky albedo.

Its top-level documentation describes these steps:

1. read and project the AOI to EPSG:32613;
2. query Sentinel-2 L2A through the Planetary Computer;
3. build reflectance and SCL stacks on a 20 m grid;
4. mask clouds/shadows while keeping snow;
5. load Copernicus DEM GLO-30;
6. calculate slope and aspect;
7. apply SCS+C terrain correction;
8. calculate narrow-to-broadband shortwave albedo; and
9. sharpen/upscale MODIS BSA and WSA to 20 m using Sentinel-2 spatial ratios.

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

The current `DATE_RANGE` in the source covers part of 2021 even though the workflow description refers to the larger project interval. Set the intended interval deliberately.

### Narrow-to-broadband weights

The script uses the following bands:

```text
B02, B03, B04, B05, B06, B07, B08, B11, B12
```

Weights are stored in `NTB_WEIGHTS`. Preserve their band association when refactoring or converting arrays.

### Terrain correction

`compute_slope_aspect(...)` calculates terrain derivatives from projected DEM spacing. `scs_plus_c_correct(...)` applies a scene/band correction using local incidence and solar zenith.

Validate:

- DEM units and CRS;
- slope/aspect orientation;
- solar azimuth convention;
- illumination values near zero;
- correction behavior in deep shadow; and
- before/after relationships with slope, aspect, and field measurements.

### BRDF fusion

The fusion workflow obtains daily MCD43A3 BSA and WSA, resamples them to the Sentinel-2 grid, and sharpens them with a ratio between 20 m Sentinel-2 albedo and a coarser box-averaged Sentinel-2 representation.

Use `RATIO_EPS` to protect division by near-zero values, but inspect extreme ratios and output artifacts.

### Documented outputs

The script header lists outputs such as:

```text
s2_albedo20m_topocorr_keepSnow_YYYY-MM-DD.tif
s2_albedo20m_topocorr_snowOnly_YYYY-MM-DD.tif
s2_fused_BSA20m_YYYY-MM-DD.tif
s2_fused_WSA20m_YYYY-MM-DD.tif
s2_fused_BSA20m_snowOnly_YYYY-MM-DD.tif
s2_fused_WSA20m_snowOnly_YYYY-MM-DD.tif
```

Optional median composites may also be produced.

## `tsi_functions.py`

### Configuration

The module reads the East River shapefile at import time and creates a WGS84 AOI. Update:

```python
shapefile_path = "/path/to/East_River.shp"
```

before importing it.

### Clear-sky date filtering

`dates_with_clear_sky(...)` scans TSI `.cdf` files for:

```text
time
percent_thin
percent_opaque
```

It removes fill values of `-100`, applies thin/opaque thresholds, and returns unique dates as calendar or Julian strings.

The default thresholds are 5 in the function, which assumes the variables are expressed as percentages. Check the units in the CDF metadata.

### Sentinel-2 temporal matching

`match_tsi_to_s2(...)`:

- interprets Sentinel-2 item times in UTC;
- parses `YYYYMMDD` from TSI filenames;
- finds files from the same UTC date;
- filters valid thin/opaque observations;
- selects the nearest valid TSI timestamp; and
- returns aligned lists of TSI times, Sentinel-2 items, and TSI filenames.

The current function defaults for `thin_threshold` and `opaque_threshold` are 100, while the docstring discusses 5 percent. Pass the intended thresholds explicitly.

The code comments describe a 30-minute maximum gap, but the enforcing block is commented. Add or restore a tested tolerance in the calling workflow if temporal proximity is required.

### Visualization helpers

The module includes functions to:

- find TSI files by date;
- obtain scene-level cloud percentages from STAC properties;
- open signed Sentinel-2 RGB or single-band assets;
- stretch imagery for display;
- construct missing TSI time coordinates;
- plot thin and opaque cloud time series near the matched time; and
- display the corresponding Sentinel-2 scene.

### Spatial cloud-fraction helpers

Later portions of the module support coordinate transformation, AOI/radius masks, Sentinel-2 asset reads, and cloud-fraction comparison. Review each helper's expected CRS and whether its distance or radius is in meters before use.

## `auto_open_cdf.py`

This command-line tool helps determine whether a `.cdf` file is:

- NetCDF-4 stored in HDF5;
- classic NetCDF-3;
- NASA Common Data Format; or
- an unknown format.

Run:

```bash
python sentinel_2/s2_functions/auto_open_cdf.py /path/to/file.cdf
```

The script:

1. reads file magic bytes;
2. tries Xarray and `netCDF4` for NetCDF;
3. falls back to `spacepy.pycdf` for NASA CDF; and
4. prints dimensions, variables, shapes, dtypes, and a small preview.

### Optional NASA CDF dependency

Install SpacePy if NASA CDF support is required:

```bash
mamba install -c conda-forge spacepy
```

ARM `.cdf` files are often NetCDF-formatted, so SpacePy may not be necessary for the TSI products used here.

## Validation checklist

Before processing a full Sentinel-2 interval, verify:

- AOI file and CRS;
- STAC item count and identifiers;
- signed asset access;
- target CRS and 20 m transform;
- reflectance scale and baseline harmonization;
- continuous versus categorical resampling;
- cloud/SCL classes;
- NDSI and snow thresholds;
- broadband coefficients and band order;
- BRDF angle units and conventions;
- AOT/AOD lookup coverage;
- DEM alignment for terrain correction;
- TSI variable units and fill values;
- UTC timestamps and maximum time gap; and
- output naming and metadata.

## Common problems

### Import fails at `gpd.read_file`

The module-level shapefile path is wrong. Update it before importing.

### STAC assets return 403 or token errors

Re-sign items through the Planetary Computer. Do not reuse expired URLs.

### `KeyError` for a band or asset

Inspect `item.assets.keys()`. Asset names can vary between collections or client representations. Update the mapping without confusing B08 and B8A.

### Albedo array contains extreme values

Check reflectance scaling, baseline harmonization, BRDF denominator values, coefficient/band order, aerosol lookup, and whether clouds/shadows were masked.

### TSI CDF cannot be opened

Use `auto_open_cdf.py` to determine the container type. If it is NASA CDF, install SpacePy; if it is NetCDF, inspect the available engine and file integrity.

### TSI and Sentinel-2 scenes do not match

Check UTC handling, filename date parsing, threshold units, fill values, and the currently unenforced time-gap limit.

## Relationship to the rest of the repository

- `sw_lut.csv` is in `../../GOES-Modis-Data-Preprocessing-main/`.
- Sentinel-2 notebooks that call these helpers are in the parent directory.
- Selected 20 m shortwave outputs are used by `../../s2_modis_downscaling/modis_s2_unet2.py`.
- The root README describes the complete GOES-to-MODIS-to-Sentinel workflow.

## Reproducibility

Save a configuration record for each run containing paths, item IDs, date range, AOI, target grid, coefficients, masks, thresholds, aerosol source, DEM source, TSI matching rules, software environment, and repository commit.