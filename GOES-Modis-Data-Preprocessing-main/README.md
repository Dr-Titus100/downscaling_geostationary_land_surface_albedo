# GOES-MODIS Data Preprocessing

This directory contains the notebooks and reference lookup table used to prepare GOES and MODIS inputs for the first U-Net stage.

## Contents

| Path | Description |
|---|---|
| `GOES_Modis_U-Net_Data_Preprocessing.ipynb` | Clips, quality-filters, reprojects, aligns, and date-matches GOES and MODIS rasters. |
| `calculate_modis_blue_sky_albedo.ipynb` | Calculates MODIS blue-sky albedo from black-sky and white-sky albedo. |
| `sw_lut.csv` | Shortwave diffuse-skylight lookup table indexed by solar zenith angle and aerosol optical depth. |
| `shapefile_colorado/` | Legacy Colorado boundary bundle used by existing notebook paths. |

The reusable implementations used by these notebooks are now located in:

```text
../goes_modis_downscaling/data_preprocessing.py
../goes_modis_downscaling/modis_bluesky_albedo.py
../goes_modis_downscaling/packages.py
```

## Required inputs

- GOES ABI Level-2 LSAC NetCDF files with `LSA`, `DQF`, and projection metadata;
- MODIS MCD43A3 black-sky, white-sky, and quality subdatasets;
- optional MODIS or CERES aerosol data;
- East River AOI geometry;
- Colorado boundary data for regional clipping; and
- writable output directories for intermediate GeoTIFFs.

## Configuration

Update all absolute paths in the notebooks and in `../goes_modis_downscaling/data_preprocessing.py` before importing it. That module reads the AOI, Colorado boundary, and a reference GOES file at import time.

Use this notebook setup:

```python
from pathlib import Path
import sys

repo_root = Path("/path/to/downscaling_geostationary_land_surface_albedo")
sys.path.append(str(repo_root / "goes_modis_downscaling"))

from data_preprocessing import *
from modis_bluesky_albedo import *
```

The project also includes current boundary files in `../shape_files/`. Existing code still points to the legacy boundary path under this directory, so update the configured path deliberately when switching sources.

## Main preprocessing sequence

1. Read the East River and Colorado boundaries.
2. Inspect a known GOES scene and construct its geostationary CRS.
3. Traverse the NOAA hierarchy `YYYY/DDD/HH/`.
4. select the preferred 18 UTC scene or configured 19 UTC fallback;
5. quality-screen GOES DQF values;
6. divide GOES LSA by `10000`;
7. reproject GOES to EPSG:32613 and match the MODIS grid;
8. clip and reproject MODIS shortwave BSA and WSA;
9. record invalid or incomplete dates; and
10. optionally mask GOES wherever MODIS is missing.

Continuous albedo data should use bilinear resampling. DQF and other categorical layers should use nearest-neighbor resampling.

## MODIS blue-sky calculation

`calculate_modis_blue_sky_albedo.ipynb` uses:

```text
blue_sky = (1 - diffuse_fraction) * black_sky
           + diffuse_fraction * white_sky
```

MODIS scaled integers are multiplied by `0.001`. The diffuse fraction is obtained from aerosol optical depth, solar zenith angle, and `sw_lut.csv`.

## Expected outputs

Typical outputs include:

- clipped and reprojected GOES LSA GeoTIFFs;
- clipped MODIS BSA and WSA rasters;
- daily MODIS blue-sky albedo GeoTIFFs;
- GOES rasters masked to the MODIS valid-data footprint; and
- an invalid-date JSON file.

## Validation

For representative dates, verify:

- identical CRS, transform, width, and height for paired rasters;
- expected East River coverage;
- GOES and MODIS scaling applied exactly once;
- DQF resampled with nearest-neighbor;
- BSA and WSA band order;
- physically plausible albedo values; and
- agreement between retained files and the invalid-date list.

## Next step

Continue with:

```text
../goes_modis_downscaling/README.md
```
