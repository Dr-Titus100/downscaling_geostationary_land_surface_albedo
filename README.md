# A project aimed at creating high-resolution spatio-temporal land surface albedo maps to enhance the modeling of snowmelt dynamics and water resource forecasting.

## Title: Using Low-Earth Orbit Instruments and Neural Networks to Downscale Geostationary Surface Albedo Products

## Abstract:
In the Upper Colorado River Basin (UCRB), USA, the seasonal mountain snowpack provides critical water resources for downstream agricultural and municipal communities. A source of uncertainty in estimates of snow water supply timing and amount can arise from poor constraints on how snow albedo varies in space and time, especially over the spring snowmelt period. Existing methods to observe snow albedo from remote sensing platforms are limited due to the trade-off between high spatial and temporal resolution. To circumvent this observational trade-off gap, we conducted a case study in the East River Basin of the UCRB  to produce a downscaled remotely sensed land surface albedo (LSA) and snow albedo product with simultaneously high spatial and temporal resolution. This product was derived by downscaling geostationary albedo to create high-spatio-temporal LSA and snow albedo maps. We utilize a U-Net neural network model to downscale spatiotemporal observations of surface albedo from 2 km GOES resolution images to 500 m MODIS resolution images. We then leverage the DOE ARM Surface Atmosphere Integrated Field Laboratory (SAIL) data for ground truthing. The final downscaled product achieves both high spatial and high temporal resolution and, therefore, enables detailed insights into snow albedo evolution during periods of rapid transition.  These albedo outputs have promising applications for assimilation into process-based land surface hydrology models, with the potential to improve near-real-time estimates of snow water storage in mountain regions 

## Project overview

This repository contains the data acquisition, geospatial preprocessing, neural-network training, evaluation, visualization, and Sentinel-2 processing workflows used to create high-resolution land-surface and snow-albedo products for the East River Basin in Colorado.

The repository currently supports three model routes:

1. **GOES to MODIS:** GOES-R ABI Level-2 Land Surface Albedo is aligned to a common 500 m grid and used to predict MODIS blue-sky albedo.
2. **MODIS to Sentinel-2:** first-stage MODIS-scale predictions are reprojected to the Sentinel-2 grid and used to predict 20 m Sentinel-2 shortwave blue-sky albedo.
3. **GOES to Sentinel-2:** a direct model reprojects GOES albedo to the Sentinel-2 grid and predicts the same 20 m Sentinel-2 target.

The Sentinel-2 workflows can also independently produce broadband SW, VIS, and NIR albedo, snow masks, BRDF-normalized products, terrain-corrected products, cloud diagnostics, and MODIS-fused BSA and WSA products.

## Processing routes

```text
NOAA GOES ABI L2 LSAC              NASA MODIS MCD43 products
           |                                   |
           v                                   v
   Download and organize             Download, clip, reproject,
   YYYY/day-of-year/hour              and calculate blue-sky albedo
           |                                   |
           +-------------------+---------------+
                               |
                               v
                  GOES-to-MODIS U-Net at 500 m
                               |
                 +-------------+-------------+
                 |                           |
                 v                           v
      MODIS-to-Sentinel-2 U-Net       Postprocessing, metrics,
          on the 20 m S2 grid         rasters, and figures

GOES 500 m-aligned rasters + Sentinel-2 20 m targets
                         |
                         v
              Direct GOES-to-Sentinel-2 U-Net
```

## Repository structure

| Path | Purpose |
|---|---|
| [`GOES-Modis-Data-Preprocessing-main/`](GOES-Modis-Data-Preprocessing-main/) | Interactive GOES and MODIS preprocessing, common-grid construction, quality screening, and MODIS blue-sky albedo calculation. |
| [`goes_modis_downscaling/`](goes_modis_downscaling/) | First-stage GOES-to-MODIS U-Net, shared preprocessing/model/plotting modules, and notebook driver. |
| [`GOES-Modis-Albedo-Postprocessing-main/`](GOES-Modis-Albedo-Postprocessing-main/) | Prediction rasterization, evaluation, field comparison, and final visualization notebook. |
| [`goes_s2_downscaling/`](goes_s2_downscaling/) | Direct GOES-to-Sentinel-2 U-Net workflow. |
| [`modis_s2_downscaling/`](modis_s2_downscaling/) | MODIS-prediction-to-Sentinel-2 U-Net workflow. |
| [`goes_data_download/`](goes_data_download/) | NOAA GOES ABI LSAC downloader. |
| [`modis_data_download/`](modis_data_download/) | NASA Earthdata MODIS downloader using `earthaccess`. |
| [`sentinel_2/`](sentinel_2/) | Sentinel-2 retrieval, harmonization, cloud/TSI analysis, albedo production, and visualization notebooks. |
| [`sentinel_2/s2_functions/`](sentinel_2/s2_functions/) | Reusable Sentinel-2, TSI, CDF, terrain-correction, and BRDF-fusion scripts. |
| [`shape_files/`](shape_files/) | East River and Colorado boundaries plus the boundary-visualization notebook. |
| [`MNRAS.mplstyle`](MNRAS.mplstyle) | Matplotlib style used by project notebooks. |
| [`environment.yml`](environment.yml) | Conda environment for geospatial processing, machine learning, notebooks, and visualization. |
| [`LICENSE`](LICENSE) | MIT license for this repository. |

Each major code or data-support directory has a README that describes its current files, inputs, configuration, outputs, and dependencies.

## Data sources

Large source datasets and generated outputs are not stored in the repository.

| Dataset | Role |
|---|---|
| NOAA GOES-R ABI Level-2 LSAC | High-temporal-resolution land-surface albedo predictor. |
| MODIS MCD43A3 v061 | Black-sky and white-sky albedo used to derive the first-stage target. |
| MODIS or CERES aerosol data | Diffuse-fraction support for blue-sky albedo. |
| Sentinel-2 L2A | Surface reflectance, scene classification, aerosol information, and 20 m target products. |
| Copernicus DEM GLO-30 | Optional slope, aspect, and topographic correction. |
| ARM SAIL Total Sky Imager products | Thin-cloud, opaque-cloud, and image-based cloud diagnostics. |
| East River and Colorado boundaries | Clipping, mapping, and common-grid construction. |
| SAIL field albedo observations | Point-scale comparison and ground-truth diagnostics. |

Users are responsible for following each provider's access, attribution, and redistribution requirements.

## Environment setup

From the repository root:

```bash
mamba env create -f environment.yml
conda activate sail_env
python -m ipykernel install --user --name sail_env --display-name "Python (sail_env)"
jupyter lab
```

Conda can be used in place of Mamba.

A CUDA-capable GPU is recommended for U-Net training. Preprocessing and small tests can run on a CPU.

## Required configuration

The repository contains research scripts rather than a packaged command-line application. Many files include absolute Boise State or NERSC paths. Update them before importing modules or running notebooks.

Locate hard-coded paths with:

```bash
grep -R "/bsuhome/tnde\|/global/cfs\|~/geoscience" -n \
  --include="*.py" --include="*.ipynb" .
```

At minimum, configure:

1. East River or replacement AOI geometry.
2. GOES raw, clipped, aligned, masked, and prediction directories.
3. MODIS raw, clipped, blue-sky, prediction, and invalid-date paths.
4. Sentinel-2 output, cloud-fraction, TSI, DEM, and ancillary paths.
5. Model, checkpoint, history, NumPy, and GeoTIFF output directories.
6. The lookup table at `GOES-Modis-Data-Preprocessing-main/sw_lut.csv`.
7. Notebook import paths for the relevant sibling module directory.
8. The included `MNRAS.mplstyle`, or remove the custom style call.

Some modules open shapefiles or reference rasters during import. Their configured paths must exist before importing them.

## End-to-end guide

### 1. Download GOES

Configure and run:

```bash
python goes_data_download/download_goes_data.py
```

The script preserves the NOAA hierarchy `ABI-L2-LSAC/YYYY/DDD/HH/`.

### 2. Download MODIS

Configure the product, dates, bounding box, and destination in `modis_data_download/download_modis_data.py`, then run:

```bash
python modis_data_download/download_modis_data.py
```

Use `MCD43A3` for the main MODIS albedo workflow.

### 3. Preprocess GOES and MODIS

Open:

```text
GOES-Modis-Data-Preprocessing-main/GOES_Modis_U-Net_Data_Preprocessing.ipynb
```

The notebook clips, quality-filters, reprojects, aligns, and date-matches GOES and MODIS data. It imports helpers from `goes_modis_downscaling/`.

### 4. Calculate MODIS blue-sky albedo

Open:

```text
GOES-Modis-Data-Preprocessing-main/calculate_modis_blue_sky_albedo.ipynb
```

The calculation combines MODIS black-sky and white-sky albedo using a diffuse-skylight fraction.

### 5. Train the GOES-to-MODIS model

Use the notebook:

```text
goes_modis_downscaling/run_goes_modis_unet.ipynb
```

or the script:

```bash
python goes_modis_downscaling/goes_modis_unet.py
```

Review `goes_modis_downscaling/README.md` before running.

### 6. Evaluate first-stage predictions

Open:

```text
GOES-Modis-Albedo-Postprocessing-main/Final-Visualizations.ipynb
```

This notebook creates georeferenced prediction rasters, computes masked metrics, compares products, samples the SAIL location, and prepares figures.

### 7. Produce Sentinel-2 target products

Review:

```text
sentinel_2/README.md
sentinel_2/s2_functions/README.md
```

The principal production script is:

```text
sentinel_2/s2_functions/s2_20m_download_final.py
```

A terrain-correction and MODIS BRDF-fusion alternative is also available.

### 8. Train a Sentinel-2 refinement model

For the two-stage route:

```bash
python modis_s2_downscaling/modis_s2_unet2.py
```

For the direct route:

```bash
python goes_s2_downscaling/goes_s2_unet.py
```

Both workflows use the Sentinel-2 filename pattern:

```text
YYYY-MM-DD_S2_BLUE20m_SW_hard.tif
```

and use finite Sentinel-2 pixels as per-pixel training weights.

## Expected outputs

Depending on the selected route, outputs may include:

- clipped and aligned GOES LSA GeoTIFFs;
- MODIS black-sky, white-sky, and blue-sky albedo rasters;
- Keras models, checkpoint weights, and training histories;
- NumPy prediction arrays;
- georeferenced 500 m and 20 m prediction rasters;
- Sentinel-2 SW, VIS, NIR, BSA, WSA, and blue-sky products;
- snow masks and processing-choice QA layers;
- TSI/Sentinel-2 cloud-fraction tables;
- masked R-squared and RMSE summaries; and
- maps, time series, web visualizations, and comparison figures.

## Important assumptions

- Many workflows use EPSG:32613 for the East River study area.
- GOES LSA is divided by `10000` during preprocessing.
- MODIS albedo integers are multiplied by `0.001`.
- File matching depends on date tokens embedded in filenames.
- The first model uses fixed padding for the current approximately `21 x 19` East River grid.
- The Sentinel-2 models reproject predictors to the exact target grid and pad spatial dimensions for U-Net pooling.
- Missing target pixels are filled only for tensor computation and receive zero sample weight.
- Scene-level cloud cover is a pre-filter, not a substitute for pixel-level screening.
- Notebooks are stateful and should be executed in order after restarting the kernel.

## Reproducibility checklist

Record the following for each reported run:

- repository commit;
- environment specification;
- data product and collection versions;
- AOI, date range, CRS, resolution, and resampling;
- file-matching and invalid-date rules;
- cloud and snow-screening thresholds;
- train, validation, and test split;
- model configuration, random seed, and checkpoint;
- predictor and target filename patterns; and
- output naming and any changed coefficients or thresholds.

## Troubleshooting

### Imports fail

Update module-level paths before importing. Use sibling imports from the current directories, such as `goes_modis_downscaling/`, `goes_s2_downscaling/`, or `modis_s2_downscaling/`.

### No paired dates are found

Inspect the active globs and date parsers. The three model routes use different GOES and MODIS filename conventions.

### Predictor and target grids differ

Compare CRS, transform, width, height, and bounds. The Sentinel-2 models use `rio.reproject_match` to align predictors to the target.

### Metrics are poor or negative

Check temporal pairing, scaling, spatial alignment, target variance, masks, cloud filtering, and train/test distribution before changing the architecture.

### Planetary Computer assets fail

Sign or re-sign the item assets. Signed URLs expire.

## Credits

The original U-Net workflow acknowledges Dr. Utkarsh Mital for U-Net support, Dr. Daniel Feldman for Earth-science workflow references, Dr. William Rudisill for coding and Earth-science expertise, and Lawrence Berkeley National Laboratory for computational infrastructure.

Portions of the U-Net encoder, decoder, and model-construction code are identified in the source as modifications of an MIT-licensed implementation by Vidushi Bhatia. Preserve the attribution and embedded license notice when redistributing those portions.

## Citation

When this repository supports a publication, thesis, presentation, or derived dataset, cite the associated project paper or data release and record the repository commit used.

## License

This repository is released under the [MIT License](LICENSE). Individual source sections may also contain attribution or license notices that must be retained.
