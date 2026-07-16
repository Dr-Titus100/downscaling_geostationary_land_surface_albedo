# GOES-MODIS Data Preprocessing

This directory contains the interactive workflows and supporting reference files used to prepare GOES and MODIS data for the first U-Net downscaling stage.

The preprocessing establishes a common East River raster grid, screens unusable scenes, calculates MODIS blue-sky albedo, and creates date-matched predictor and target files.

## Contents

| Path | Description |
|---|---|
| `GOES_Modis_U-Net_Data_Preprocessing.ipynb` | Main notebook for AOI preparation, GOES clipping/reprojection, MODIS clipping/reprojection, scene checks, and GOES-to-MODIS masking. |
| `calculate_modis_blue_sky_albedo.ipynb` | Computes MODIS blue-sky albedo from black-sky and white-sky albedo using a diffuse-skylight lookup. |
| `sw_lut.csv` | Shortwave diffuse-skylight-ratio lookup table indexed by solar zenith angle and aerosol optical depth. |
| `shapefile_colorado/` | Colorado state-boundary files in several spatial formats. |
| `.gitignore` | Directory-specific ignore rule. |

Most operations are implemented in `../functions/data_preprocessing.py` and `../functions/modis_bluesky_albedo.py`.

## Prerequisites

Create and activate the project environment from the repository root:

```bash
mamba env create -f environment.yml
conda activate sail_env
jupyter lab
```

Before importing the helper modules, update their absolute path constants. `data_preprocessing.py` reads an AOI shapefile, a Colorado boundary, and an example GOES file while the module is imported.

## Required inputs

### GOES

- NOAA GOES-R ABI Level-2 LSAC NetCDF files;
- variables named `LSA`, `DQF`, and `goes_imager_projection`;
- directory hierarchy `YYYY/DDD/HH/`; and
- scenes covering the AOI near the selected MODIS observation time.

### MODIS

- MCD43A3 black-sky and white-sky shortwave albedo;
- optional MCD19A2 aerosol optical depth or the configured CERES AOD NetCDF;
- valid spatial metadata and subdataset names; and
- a seven-digit `YYYYDDD` date in filenames used later for matching.

### Spatial and ancillary data

- East River AOI shapefile;
- Colorado state boundary when using the regional clipping cells;
- `sw_lut.csv`;
- output directories with sufficient storage; and
- optional invalid-date JSON used by later modeling steps.

## Configure paths

Review the constants at the top of:

```text
../functions/data_preprocessing.py
../functions/modis_bluesky_albedo.py
```

The current code includes paths for:

- raw and clipped MODIS data;
- MODIS blue-sky output;
- MODIS AOD data;
- raw GOES data;
- clipped and reprojected GOES data;
- GOES data masked with MODIS missing pixels;
- East River and Colorado shapefiles;
- a reference MODIS raster;
- a reference GOES NetCDF; and
- a CERES AOD lookup file.

Update the notebook helper import and style paths as well:

```python
from pathlib import Path
import sys

repo_root = Path("/path/to/downscaling_geostationary_land_surface_albedo")
sys.path.append(str(repo_root / "functions"))

import matplotlib.pyplot as plt
plt.style.use(repo_root / "MNRAS.mplstyle")
```

## Notebook 1: GOES and MODIS preprocessing

Open `GOES_Modis_U-Net_Data_Preprocessing.ipynb` and run cells in order.

### 1. Build AOI geometries

The notebook reprojects the East River and Colorado boundaries into:

- the native MODIS CRS;
- EPSG:32613 for East River UTM processing; and
- the grid or CRS required for GOES alignment.

Do not merely assign a CRS to coordinates from another system. Use `to_crs(...)` for vector data and `rio.reproject(...)` or `rio.reproject_match(...)` for rasters.

### 2. Test one GOES scene

Use a known valid GOES file to confirm that:

- `LSA` and `DQF` can be opened;
- the geostationary projection is interpreted correctly;
- the AOI overlaps the raster;
- the result matches the MODIS reference grid; and
- output values are in expected albedo units.

### 3. Process the GOES archive

`clip_reproject_goes_data_loop_through_directories(...)` traverses:

```text
GOES_ROOT/YYYY/DDD/HH/files
```

The current implementation:

- filters years and Julian days to the requested date range;
- searches the 18 and 19 UTC folders;
- selects one file after the configured five-minute-file offset;
- reprojects GOES LSA to the MODIS target grid with bilinear resampling;
- reprojects DQF with nearest-neighbor resampling;
- divides LSA by `10000`;
- applies a scene-quality test; and
- writes a clipped GeoTIFF for accepted scenes.

The function returns and prints counts of accepted, rejected, and attempted files.

### 4. Clip and reproject MODIS

The notebook selects the following MCD43A3 subdatasets for shortwave albedo processing:

```text
Albedo_BSA_shortwave
Albedo_WSA_shortwave
BRDF_Albedo_Band_Mandatory_Quality_shortwave
```

MODIS data can first be clipped to Colorado, reprojected, and then clipped to the East River AOI in EPSG:32613. Ensure that black-sky and white-sky bands stay in a consistent order.

### 5. Screen missing data

The helper module includes functions for:

- examining DQF values;
- requiring a minimum valid-pixel fraction and count;
- finding dates with gaps;
- filtering MODIS scenes with excessive missing data; and
- plotting GOES and MODIS pairs for visual inspection.

Record rejected dates in a JSON file so the same dates can be removed during model preparation and postprocessing.

### 6. Mask GOES to the MODIS footprint

`mask_goes_to_match_modis(...)` identifies GOES and MODIS files from the same day and writes a GOES version with `NaN` wherever MODIS is missing.

This supports a controlled experiment in which predictor and target have the same observed-pixel footprint. The U-Net code can use either this masked directory or the unmasked GOES directory.

## Notebook 2: MODIS blue-sky albedo

Open `calculate_modis_blue_sky_albedo.ipynb` after the MCD43 rasters have been clipped and reprojected.

### Calculation sequence

1. Read band 1 as black-sky albedo.
2. Read band 2 as white-sky albedo.
3. Extract the date from each filename.
4. Obtain daily AOD from the configured raster source or CERES NetCDF.
5. Estimate solar zenith angle at local solar noon for each latitude.
6. Round the solar zenith angle to the lookup-table row.
7. convert AOD to the lookup-table column convention.
8. Retrieve the diffuse skylight ratio from `sw_lut.csv`.
9. Calculate blue-sky albedo.
10. Write one GeoTIFF per date using a reference raster's CRS, transform, and profile.

The helper uses:

```text
blue_sky = (1 - diffuse_ratio) * (black_sky * 0.001)
           + diffuse_ratio * (white_sky * 0.001)
```

The `0.001` factor converts MODIS scaled integers to albedo values.

### Output naming

The notebook writes files with a date prefix similar to:

```text
2021244_modis_blue_sky_albedo_new.tif
```

Later code expects the date to be the first seven filename characters or otherwise discoverable as `YYYYDDD`.

## `sw_lut.csv`

This file is a two-dimensional lookup table for the shortwave diffuse fraction. The first column represents solar zenith-angle rows and the remaining columns represent AOD values.

When changing or replacing it:

- keep numeric row labels compatible with rounded solar zenith angles;
- keep AOD column names formatted exactly as the conversion helper produces them;
- confirm the table covers the full AOD and zenith ranges in the data; and
- do not interpolate silently outside the supported range.

## Expected outputs

A complete run may produce:

```text
clipped_modis_data_colorado/
reprojected_colorado_modis_data_lat_lon/
blue_sky_albedo_colorado/
blue_sky_albedo_sail/
goes_output_data/
nan_data/
invalid_modis_dates.json
```

Actual names depend on the paths configured in the helper modules.

## Quality-control checklist

For several dates across the study period, verify:

- GOES and MODIS have the same CRS, transform, width, and height;
- the East River AOI overlaps every accepted scene;
- DQF is resampled with nearest-neighbor, not bilinear interpolation;
- GOES LSA was scaled once and only once;
- MODIS albedo was scaled once and only once;
- black-sky and white-sky bands were not reversed;
- masked GOES has missing pixels in the same locations as MODIS;
- the number of accepted dates agrees with the invalid-date list; and
- albedo values are physically plausible before model training.

Useful checks include:

```python
print(raster.rio.crs)
print(raster.rio.bounds())
print(raster.rio.resolution())
print(float(raster.min()), float(raster.max()))
```

## Common problems

### `NoDataInBounds`

The AOI does not overlap the raster after reprojection, or the AOI has an incorrectly assigned CRS. Print both bounds in the same CRS before clipping.

### GOES files are skipped unexpectedly

Review the selected hour directories, file offset, DQF rules, valid-pixel thresholds, and date range. NOAA directory ordering must be sorted for the offset to be deterministic.

### MODIS subdataset cannot be opened

Inspect the HDF subdataset names with Rasterio. Product versions and drivers may expose different strings. Use the strict raster-opening helper when working with HDF or NetCDF containers.

### Blue-sky lookup raises `KeyError`

The rounded solar zenith angle or converted AOD is not present in `sw_lut.csv`. Inspect the lookup index and columns, and decide explicitly how out-of-range values should be handled.

### Imports fail immediately

The helper modules open configured files at import time. Correct the paths before importing them.

## Relationship to the model

The first U-Net expects paired daily rasters on the common 500 m grid:

- predictor: GOES LSA, optionally masked to MODIS missing pixels;
- target: MODIS blue-sky albedo; and
- target mask: finite MODIS pixels used as per-pixel sample weights.

Continue with `../GOES-Modis-U-Net-Albedo-Code-main/README.md` after the paired rasters and invalid-date list are complete.