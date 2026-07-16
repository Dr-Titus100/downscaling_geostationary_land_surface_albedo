# Sentinel-2 Albedo and Cloud-Assessment Workflows

This directory contains the Sentinel-2 portion of the project. It supports Sentinel-2 L2A retrieval, 20 m broadband albedo production, processing-baseline harmonization, cloud and snow screening, BRDF normalization, aerosol-based blue-sky albedo, optional terrain correction, ARM Total Sky Imager matching, image-based cloud-fraction analysis, and visualization.

The generated shortwave blue-sky products are targets for both:

- `../modis_s2_downscaling/`, which uses first-stage MODIS-scale predictions; and
- `../goes_s2_downscaling/`, which uses GOES predictors directly.

## Contents

| Path | Purpose |
|---|---|
| `s2_20m_download_final.ipynb` | Interactive execution and inspection of the principal 20 m albedo workflow. |
| `s2_20m_download_pc.ipynb` | Earlier or alternate Planetary Computer retrieval workflow. |
| `s2_plots.ipynb` | Albedo, QA, snow, cloud, and cross-product visualization. |
| `baseline_change.ipynb` | Investigation of Sentinel-2 processing-baseline changes and reflectance harmonization. |
| `cloud_cover_fraction.ipynb` | Sentinel-2 and TSI temporal matching and cloud-cover diagnostics. |
| `natural_neighbor_interpolation.ipynb` | TSI decision-image processing, interpolation, and spatial cloud-fraction estimation. |
| `rgb_to_html.ipynb` | Browser-oriented RGB and image visualization experiments. |
| `packages.py` | Shared imports used by notebooks and scripts in this directory. |
| `s2_functions/` | Reusable production and helper scripts. See its README. |

## Primary production scripts

### `s2_functions/s2_20m_download_final.py`

The principal workflow includes:

- Planetary Computer Sentinel-2 L2A search;
- AOI clipping and a common 20 m grid;
- reflectance scaling and processing-baseline harmonization;
- SCL and cloud masking;
- NDSI- or SCL-based snow identification;
- hard and soft snow choices;
- SW, VIS, and NIR narrow-to-broadband albedo;
- BRDF normalization;
- black-sky, white-sky, and blue-sky albedo;
- aerosol-based diffuse-fraction lookup;
- per-date rasters and QA products; and
- stack, table, plot, and web-map support.

### `s2_functions/s2_20m_albedo_topocorr_brdf_fusion.py`

This alternate route emphasizes:

- cloud/shadow masking while retaining snow;
- Copernicus DEM slope and aspect;
- SCS+C terrain correction;
- nine-band shortwave albedo; and
- ratio sharpening of MODIS MCD43A3 BSA and WSA to 20 m.

Keep outputs from the two routes clearly identified because their corrections and product definitions differ.

## Setup

From the repository root:

```bash
mamba env create -f environment.yml
conda activate sail_env
python -m ipykernel install --user --name sail_env --display-name "Python (sail_env)"
jupyter lab
```

For notebook imports:

```python
from pathlib import Path
import sys

repo_root = Path("/path/to/downscaling_geostationary_land_surface_albedo")
s2_root = repo_root / "sentinel_2"
sys.path.append(str(s2_root))
sys.path.append(str(s2_root / "s2_functions"))
```

Because both directories contain a `packages.py`, keep the intended directory order explicit and avoid unrelated modules with the same name on `sys.path`.

## Required inputs and services

Depending on the route, configure:

- East River AOI geometry;
- Planetary Computer STAC access;
- Sentinel-2 L2A assets;
- `../GOES-Modis-Data-Preprocessing-main/sw_lut.csv`;
- ARM TSI sky-cover `.cdf` files;
- TSI decision-image archives;
- optional MODIS MCD43A3 and aerosol data;
- optional Copernicus DEM GLO-30; and
- output directories for rasters, tables, stacks, images, and HTML products.

## Required configuration

The notebooks and scripts contain absolute `/bsuhome/tnde/...` and `~/geoscience/...` paths. Update at least:

- AOI or shapefile path;
- output directories;
- date interval;
- target CRS and resolution;
- TSI and image-archive paths;
- diffuse-skylight lookup path;
- DEM and MODIS paths for fusion; and
- notebook helper-import paths.

Some modules read the AOI during import.

## Sentinel-2 query and grid

The final workflow queries the `sentinel-2-l2a` collection and uses a 20 m target grid. Native 10 m continuous bands are resampled to the common grid with an appropriate continuous method. SCL and other categorical layers require nearest-neighbor resampling.

Sign Planetary Computer assets immediately before access because signed URLs expire.

## Processing-baseline harmonization

The workflow accounts for processing changes affecting newer Sentinel-2 reflectance products. `baseline_change.ipynb` helps compare scenes around the cutoff and verify the active harmonization.

Record the processing baseline and apply offsets only to affected reflectance bands, not to SCL or other categorical layers.

## Cloud and invalid-pixel screening

The active SCL exclusions vary by workflow. The terrain-correction route excludes:

```text
0, 1, 2, 3, 6, 8, 9, 10
```

and retains snow/ice class 11. Scene-level cloud metadata is only a pre-filter; use pixel-level screening for products and model targets.

## Snow identification

The final route supports NDSI-based snow or SCL class 11. Common NDSI settings in the project are:

```text
NDSI threshold = 0.4
green reflectance minimum = 0.2
```

Outputs can include keep-all, snow-only, hard-choice, soft-choice, NDSI, probability, and QA variants. Validate the snow mask across terrain, vegetation, cloud, shadow, and seasonal transitions.

## Broadband and BRDF products

The final workflow uses separate snow and snow-free coefficient sets for SW, VIS, and NIR albedo. Reflectance must be in the expected scale before applying coefficients.

It can produce BSA, WSA, and blue-sky albedo using:

```text
blue_sky = (1 - diffuse_fraction) * black_sky
           + diffuse_fraction * white_sky
```

The alternate terrain-correction route uses B02, B03, B04, B05, B06, B07, B08, B11, and B12 with configured narrow-to-broadband weights.

## ARM TSI matching

The TSI workflows:

1. locate `.cdf` files from the Sentinel-2 acquisition date;
2. remove fill values such as `-100`;
3. apply thin- and opaque-cloud thresholds;
4. find the nearest valid TSI timestamp; and
5. return aligned TSI and Sentinel-2 records.

The current helper contains a described 30-minute tolerance whose enforcement should be checked in the active calling workflow. Normalize timestamps to UTC.

## TSI decision-image interpolation

`natural_neighbor_interpolation.ipynb` uses image processing, triangulation, natural-neighbor or grid interpolation, fallback methods, and circular spatial masks to estimate cloud fractions from TSI decision images. Process large archives incrementally and verify timestamp alignment before exporting model-screening tables.

## Model-ready output convention

The two 20 m U-Net directories currently expect target filenames of the form:

```text
YYYY-MM-DD_S2_BLUE20m_SW_hard.tif
```

The cloud table is commonly named:

```text
tsi_cloud_fractions.csv
```

and the current model scripts filter on `cf_interp`. Update both model directories if these conventions change.

## Recommended sequence

1. Configure paths, AOI, date range, and credentials.
2. Query a short test interval.
3. inspect item metadata, assets, CRS, and reflectance scale.
4. Verify harmonization and masks.
5. Produce and inspect albedo and QA rasters.
6. Match TSI data and generate cloud diagnostics.
7. Process the full interval.
8. Record retained and excluded dates.
9. Use targets in `../modis_s2_downscaling/` or `../goes_s2_downscaling/`.

## Expected outputs

Outputs may include:

- daily 20 m SW, VIS, and NIR albedo;
- keep-all and snow-only products;
- hard and soft snow-choice products;
- BSA, WSA, and blue-sky albedo;
- terrain-corrected and MODIS-fused products;
- NDSI, masks, probability, and QA layers;
- NetCDF or Xarray stacks;
- TSI match and cloud-fraction tables;
- RGB, map, diagnostic, and time-series figures; and
- browser-viewable HTML products.

## Quality control

For every retained date, verify:

- acquisition time and processing baseline;
- successful signed-asset access;
- CRS, transform, shape, and 20 m resolution;
- reflectance scaling and harmonization;
- SCL and cloud exclusions;
- snow-choice behavior;
- coefficient and band order;
- aerosol lookup coverage;
- finite-pixel fraction;
- target filename and variable; and
- TSI timestamp and temporal separation.

## Common problems

### Asset authorization fails

Re-sign the Planetary Computer assets.

### No items are returned

Check AOI validity, longitude/latitude order, collection name, date range, and cloud query.

### Output is spatially offset

Reproject rather than relabeling the CRS, and compare transforms and bounds.

### Snow and cloud are confused

Inspect SCL, NDSI, visible and SWIR reflectance, and TSI evidence together.

### TSI times are unexpected

Normalize to UTC and enforce an explicit maximum time difference.

### Interpolation raises Qhull errors

Remove duplicate or degenerate points and use a documented fallback only when scientifically acceptable.

## Reproducibility

Record item IDs, acquisition and processing timestamps, processing baseline, coefficient sets, masks, snow thresholds, BRDF geometry, aerosol source, terrain settings, TSI thresholds, AOI, target grid, exclusions, and repository commit.
