# Sentinel-2 Albedo and Cloud-Assessment Workflows

This directory contains the Sentinel-2 portion of the project. It supports retrieval and processing of Sentinel-2 L2A data, broadband albedo calculation at 20 m, snow and cloud screening, BRDF normalization, aerosol-based blue-sky albedo, optional topographic correction, matching with ARM Total Sky Imager observations, interpolation of sky-camera cloud masks, and visualization of outputs.

The generated shortwave albedo products also provide the targets for the second-stage MODIS-to-Sentinel-2 U-Net in `../s2_modis_downscaling/`.

## Directory contents

| Path | Purpose |
|---|---|
| `s2_20m_download_final.ipynb` | Interactive execution and inspection of the final Sentinel-2 20 m albedo workflow. |
| `s2_20m_download_pc.ipynb` | Earlier or alternate Planetary Computer-based Sentinel-2 retrieval workflow. |
| `s2_plots.ipynb` | Visualizes and compares Sentinel-2 albedo, quality, snow, or related output products. |
| `baseline_change.ipynb` | Investigates Sentinel-2 processing-baseline changes and reflectance harmonization. |
| `cloud_cover_fraction.ipynb` | Matches Sentinel-2 scenes with TSI observations and examines cloud-cover information. |
| `natural_neighbor_interpolation.ipynb` | Processes TSI decision images and interpolates sky classifications/cloud fractions. |
| `rgb_to_html.ipynb` | Builds HTML-oriented RGB or image visualizations from processed Sentinel-2 data. |
| `s2_functions/` | Reusable production scripts and helper functions. See its README for detailed APIs and configuration. |

## Primary production scripts

Two major Sentinel-2 processing routes are available in `s2_functions/`.

### `s2_20m_download_final.py`

This is the most comprehensive Sentinel-2 workflow in the repository. It includes:

- Planetary Computer Sentinel-2 L2A search;
- AOI clipping and a common 20 m grid;
- processing-baseline harmonization;
- surface-reflectance scaling;
- SCL cloud and invalid-pixel masking;
- NDSI- or SCL-based snow detection;
- hard and soft snow choices;
- BRDF c-factor normalization;
- SW, VIS, and NIR narrow-to-broadband albedo;
- black-sky, white-sky, and blue-sky calculations;
- diffuse-fraction lookup using aerosol information;
- per-date GeoTIFF outputs;
- quality-assurance layers;
- stack/NetCDF/CSV bookkeeping; and
- plotting or web-map support used by related notebooks.

### `s2_20m_albedo_topocorr_brdf_fusion.py`

This alternate workflow emphasizes:

- 20 m Sentinel-2 reflectance stacks;
- cloud/shadow masking while retaining snow;
- Copernicus DEM GLO-30 slope and aspect;
- SCS+C terrain-illumination correction;
- narrow-to-broadband shortwave albedo; and
- fusion with MODIS MCD43A3 black-sky and white-sky albedo through ratio sharpening.

Use this route when topographic correction and MODIS BRDF fusion are the main objective. Its output definitions differ from those in the final download script, so keep the workflows and metadata clearly separated.

## Prerequisites

Create the environment at the repository root:

```bash
mamba env create -f environment.yml
conda activate sail_env
python -m ipykernel install --user --name sail_env --display-name "Python (sail_env)"
jupyter lab
```

The supplied environment includes the Planetary Computer client, PySTAC Client, Rasterio, GeoPandas, Xarray, SciPy, Dask, Earthaccess, and most plotting dependencies.

Some exploratory notebooks also use packages such as OpenCV, Pillow, MetPy, or ImageIO. Install missing optional packages in `sail_env` only when the selected notebook requires them.

## Required data and services

Depending on the workflow, configure:

- East River AOI shapefile;
- Microsoft Planetary Computer STAC access;
- Sentinel-2 L2A assets;
- `GOES-Modis-Data-Preprocessing-main/sw_lut.csv`;
- ARM TSI sky-cover `.cdf` files;
- TSI decision-image `.png.tar` archives;
- optional Doppler-lidar/cloud-base-height NetCDF data;
- optional MODIS MCD43A3 files;
- optional CERES or MODIS aerosol data;
- optional Copernicus DEM GLO-30; and
- output directories with enough space for daily multiband and single-band rasters.

## Configure paths

The notebooks and scripts contain absolute paths under `/bsuhome/tnde/...` and `~/geoscience/...`. Update them before import or execution.

Recommended notebook setup:

```python
from pathlib import Path
import sys

repo_root = Path("/path/to/downscaling_geostationary_land_surface_albedo")
s2_root = repo_root / "sentinel_2"
sys.path.append(str(s2_root / "s2_functions"))

import matplotlib.pyplot as plt
plt.style.use(repo_root / "MNRAS.mplstyle")
```

Review at minimum:

- `shapefile_path`;
- `out_dir` or `OUT_DIR`;
- `time_of_interest` or `DATE_RANGE`;
- TSI `.cdf` directory;
- TSI image archive directory;
- diffuse-skylight lookup path;
- DEM/MODIS paths for the fusion workflow; and
- helper `function_path` values inside notebooks.

Several scripts read the AOI shapefile at import time, so the configured file must exist before importing the module.

## Sentinel-2 query and grid

The final workflow queries the Planetary Computer `sentinel-2-l2a` collection using the AOI geometry and configured date interval.

The current target resolution is:

```text
20 meters
```

This resolution accommodates B8A, B11, B12, and the 20 m scene-classification layer. Native 10 m bands are resampled to the target grid, normally with bilinear interpolation for continuous reflectance. Categorical masks must use nearest-neighbor resampling.

Always sign Planetary Computer item assets immediately before access. Signed asset URLs can expire.

## Processing-baseline harmonization

Sentinel-2 processing changes introduced a radiometric offset for newer products. The final script defines a cutoff of January 25, 2022 and a set of reflectance bands to harmonize.

The `baseline_change.ipynb` notebook is intended for checking the effect of that change and confirming that scenes before and after the cutoff are comparable.

When modifying harmonization:

- inspect the product baseline metadata;
- apply offsets only to affected surface-reflectance bands;
- do not alter SCL or quality layers as though they were reflectance; and
- document whether reflectance is stored as digital numbers or scaled values.

## Cloud and invalid-pixel masking

The final workflow uses Sentinel-2 Scene Classification Layer classes. The active mask excludes configured no-data, saturation, cloud-shadow, unclassified, cloud, and cirrus classes.

The alternative terrain-correction workflow excludes:

```text
0, 1, 2, 3, 6, 8, 9, 10
```

and intentionally retains snow/ice class 11.

These class lists are not identical. Record the active list with every output product.

Scene-level `eo:cloud_cover` can be used as a search pre-filter, but pixel-level masking remains necessary.

## Snow identification

The final script supports:

- NDSI-based snow detection; or
- SCL class 11.

The current NDSI configuration uses:

```text
NDSI threshold = 0.4
green reflectance minimum = 0.2
```

NDSI is calculated from green and SWIR reflectance. The workflow produces keep-all and snow-only variants and can maintain hard and soft snow-selection products.

Validate snow masks visually across autumn, winter, spring transition, forest, shadow, cloud, and bare-ground conditions. Thresholds suitable for East River may not transfer directly to another region.

## Broadband albedo

The final workflow includes separate coefficient sets for snow and snow-free surfaces from Li et al. (2018).

It calculates:

- shortwave albedo;
- visible albedo; and
- near-infrared albedo.

Reflectance inputs must be scaled to the `0-1` range before applying the linear coefficients. Check outputs for physically implausible values and decide whether clipping is scientifically justified for the intended analysis.

The alternate terrain-correction workflow uses a nine-band narrow-to-broadband weighted sum with B02, B03, B04, B05, B06, B07, B08, B11, and B12.

## BRDF and blue-sky albedo

The final workflow contains BRDF kernel coefficients for MODIS-equivalent spectral bands and a reference geometry of:

```text
solar zenith angle: 45 degrees
view zenith angle:   0 degrees
relative azimuth:    0 degrees
```

It can derive black-sky and white-sky albedo and then combine them with a diffuse fraction:

```text
blue_sky = (1 - diffuse_fraction) * black_sky
           + diffuse_fraction * white_sky
```

The diffuse fraction can use Sentinel-2 aerosol optical thickness and `sw_lut.csv`, or an alternate aerosol source according to the active configuration.

Verify that lookup-table AOD columns and solar-angle rows cover the observed values.

## Optional topographic correction

`s2_20m_albedo_topocorr_brdf_fusion.py`:

1. loads a projected DEM;
2. calculates slope and aspect;
3. derives local illumination from solar geometry;
4. applies SCS+C correction per band; and
5. calculates terrain-corrected shortwave albedo.

The current SCS+C helper estimates a correction term from scene statistics. Treat it as a research implementation and validate it against uncorrected reflectance, terrain aspect, solar geometry, and field observations.

## ARM TSI matching

`cloud_cover_fraction.ipynb`, `natural_neighbor_interpolation.ipynb`, and `s2_functions/tsi_functions.py` connect Sentinel-2 scenes to ARM Total Sky Imager data.

The matching workflow:

1. groups Sentinel-2 items by UTC date;
2. locates a TSI `.cdf` file containing the same calendar date;
3. removes TSI fill values, usually `-100`;
4. filters thin and opaque cloud percentages by configured thresholds;
5. finds the nearest TSI timestamp to each Sentinel-2 acquisition; and
6. returns aligned TSI times, Sentinel-2 items, and source filenames.

A 30-minute maximum separation is described in the helper documentation, but the enforcing check is commented in the current implementation. Reinstate and test the tolerance when strict temporal matching is required.

## TSI decision-image interpolation

`natural_neighbor_interpolation.ipynb` works with TSI decision images stored in compressed archives and uses image processing plus spatial interpolation to fill or analyze cloud classification.

The notebook includes dependencies and experiments involving:

- OpenCV and Pillow image handling;
- HSV or color-class processing;
- Delaunay triangulation;
- natural-neighbor interpolation through MetPy;
- SciPy linear/grid interpolation fallbacks;
- Qhull error handling; and
- cloud-fraction calculation over a defined circular field of view.

Large image archives should be processed incrementally. Verify that image timestamps, TSI timestamps, and Sentinel-2 UTC times are aligned before exporting a cloud-fraction CSV.

## Notebook guide

### `s2_20m_download_final.ipynb`

Use for interactive runs of the final albedo pipeline. Restart the kernel, configure paths and dates, run cells in order, and inspect a small number of scenes before processing the full interval.

### `s2_20m_download_pc.ipynb`

Use as a reference for Planetary Computer retrieval and earlier processing experiments. Compare active coefficients and masks with the final script before using its outputs in model training.

### `baseline_change.ipynb`

Use to diagnose reflectance differences around the Sentinel-2 processing-baseline cutoff and confirm the harmonization method.

### `cloud_cover_fraction.ipynb`

Use to inspect TSI thin/opaque cloud time series beside matched Sentinel-2 RGB images and to compare scene metadata with ground-based sky observations.

### `natural_neighbor_interpolation.ipynb`

Use to build cloud masks or cloud fractions from TSI decision imagery and produce matched cloud-screening information.

### `rgb_to_html.ipynb`

Use to prepare browser-viewable RGB imagery or HTML outputs. Confirm that generated HTML references accessible local or embedded assets before sharing it.

### `s2_plots.ipynb`

Use for final plotting and cross-product visual checks. Update all paths and keep output scale, CRS, color range, snow mask, and date in each figure caption.

## Expected outputs

Depending on the selected workflow, outputs may include:

- per-date 20 m SW, VIS, and NIR albedo GeoTIFFs;
- keep-all and snow-only products;
- hard and soft snow-choice products;
- black-sky, white-sky, and blue-sky albedo;
- terrain-corrected shortwave albedo;
- MODIS-fused 20 m BSA and WSA;
- NDSI, snow probability, snow mask, and processing-choice QA layers;
- time-indexed Xarray/NetCDF stacks;
- per-date metadata or summary CSV files;
- TSI/Sentinel-2 match tables;
- interpolated cloud-fraction CSV used by U-Net 2;
- RGB, diagnostic, time-series, and map figures; and
- HTML visualization products.

Output file patterns in the second U-Net currently expect:

```text
YYYY-MM-DD_S2_BLUE20m_SW_hard.tif
```

Do not change this pattern without updating `../s2_modis_downscaling/modis_s2_unet2.py`.

## Recommended execution sequence

```text
1. Configure AOI, output paths, date range, and credentials.
2. Query a short Sentinel-2 date interval.
3. Inspect baseline metadata, assets, CRS, and scale.
4. Produce one test scene at 20 m.
5. Validate SCL/cloud and snow masks.
6. Validate SW/VIS/NIR and BRDF outputs.
7. Match TSI data and create cloud-fraction diagnostics.
8. Process the full date interval.
9. Build summary stacks and visualizations.
10. Select model-ready scenes and record all exclusions.
11. Run the second U-Net in `../s2_modis_downscaling/`.
```

## Quality-control checklist

For every retained date, confirm:

- acquisition time and product baseline;
- asset signing and successful read;
- CRS, affine transform, width, height, and 20 m resolution;
- reflectance scaling and harmonization;
- cloud/SCL exclusion classes;
- NDSI and snow-choice behavior;
- aerosol and diffuse-fraction validity;
- finite-pixel percentage;
- albedo value range and spatial pattern;
- correct keep-all versus snow-only semantics;
- match to the correct TSI timestamp and file; and
- output filename date and target variable.

## Common problems

### Planetary Computer asset cannot be opened

Sign the item or asset again. Signed URLs expire, and unsigned HTTPS assets may return authorization errors.

### No Sentinel-2 items are returned

Check AOI geometry validity, WGS84 coordinate order, date interval, collection name, and optional cloud query.

### Output appears offset from the AOI

Verify the target transform and CRS. Reproject vectors and rasters rather than relabeling CRSs.

### Snow is confused with cloud

Inspect SCL, NDSI, green reflectance, SWIR reflectance, cloud classes, and TSI information together. Do not rely on a single scene-level cloud percentage.

### TSI matching returns unexpected times

Normalize both sources to UTC and enforce a maximum time difference. Check whether TSI timestamps are stored as timezone-naive UTC values.

### Natural-neighbor interpolation fails with Qhull errors

The valid sample points may be too few, duplicated, or nearly collinear. Remove duplicates and use the documented linear or nearest fallback when scientifically acceptable.

### Notebook import refers to a missing module

Some notebooks retain older names such as `s2_20m_download`. Import `s2_20m_download_final` or update the helper name deliberately after comparing behavior.

## Reproducibility

For every Sentinel-2 product set, save:

- STAC item IDs;
- acquisition and processing timestamps;
- processing baseline;
- active coefficient tables;
- reflectance scaling and harmonization rule;
- SCL exclusion classes;
- NDSI and snow thresholds;
- BRDF reference geometry;
- aerosol source and lookup table;
- terrain-correction settings;
- TSI temporal tolerance and cloud thresholds;
- AOI geometry and target grid; and
- repository commit.