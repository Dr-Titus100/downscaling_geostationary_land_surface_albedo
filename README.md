# A project aimed at creating high-resolution spatio-temporal land surface albedo maps to enhance the modeling of snowmelt dynamics and water resource forecasting.

## Title: Using Low-Earth Orbit Instruments and Neural Networks to Downscale Geostationary Surface Albedo Products

## Abstract:
In the Upper Colorado River Basin (UCRB), USA, the seasonal mountain snowpack provides critical water resources for downstream agricultural and municipal communities. A source of uncertainty in estimates of snow water supply timing and amount can arise from poor constraints on how snow albedo varies in space and time, especially over the spring snowmelt period. Existing methods to observe snow albedo from remote sensing platforms are limited due to the trade-off between high spatial and temporal resolution. To circumvent this observational trade-off gap, we conducted a case study in the East River Basin of the UCRB  to produce a downscaled remotely sensed land surface albedo (LSA) and snow albedo product with simultaneously high spatial and temporal resolution. This product was derived by downscaling geostationary albedo to create high-spatio-temporal LSA and snow albedo maps. We utilize a U-Net neural network model to downscale spatiotemporal observations of surface albedo from 2 km GOES resolution images to 500 m MODIS resolution images. We then leverage the DOE ARM Surface Atmosphere Integrated Field Laboratory (SAIL) data for ground truthing. The final downscaled product achieves both high spatial and high temporal resolution and, therefore, enables detailed insights into snow albedo evolution during periods of rapid transition.  These albedo outputs have promising applications for assimilation into process-based land surface hydrology models, with the potential to improve near-real-time estimates of snow water storage in mountain regions 

## Project overview

This repository contains the data acquisition, geospatial preprocessing, neural-network training, evaluation, visualization, and Sentinel-2 refinement workflows used to build high-resolution land-surface and snow-albedo products for the East River Basin in Colorado.

The code supports two linked downscaling stages:

1. **GOES to MODIS:** GOES-R ABI Level-2 Land Surface Albedo is aligned with MODIS blue-sky albedo and downscaled from approximately 2 km to 500 m with a TensorFlow U-Net.
2. **MODIS to Sentinel-2:** predicted MODIS-scale albedo is paired with Sentinel-2-derived 20 m albedo and used in a second U-Net workflow to create finer-resolution predictions.

The repository also includes workflows for Sentinel-2 broadband albedo, snow masking, BRDF normalization, terrain correction, cloud assessment with ARM Total Sky Imager data, model evaluation, and publication-quality visualization.

## Processing workflow

```text
GOES ABI L2 LSAC files                    MODIS MCD43 products
          |                                        |
          v                                        v
Download and organize data               Clip, reproject, and calculate
by YYYY/day-of-year/hour                  daily blue-sky albedo
          |                                        |
          +------------------+---------------------+
                             |
                             v
             Align GOES and MODIS on a common
               500 m East River raster grid
                             |
                             v
                 Train/evaluate GOES-MODIS U-Net
                             |
                             v
              Rasterize predictions and calculate
                R-squared, RMSE, and diagnostics
                             |
                             v
       Pair predicted MODIS rasters with cloud-screened
          Sentinel-2 20 m albedo products by date
                             |
                             v
                Train/evaluate MODIS-Sentinel-2 U-Net
```

The Sentinel-2 directory also supports an independent physical-processing route that computes 20 m SW, VIS, NIR, black-sky, white-sky, and blue-sky albedo products using scene classification, NDSI snow masks, BRDF normalization, aerosol information, and optional topographic correction.

## Repository structure

| Path | Purpose |
|---|---|
| [`GOES-Modis-Data-Preprocessing-main/`](GOES-Modis-Data-Preprocessing-main/) | Clips, reprojects, quality-filters, and aligns GOES and MODIS data; calculates MODIS blue-sky albedo. |
| [`GOES-Modis-U-Net-Albedo-Code-main/`](GOES-Modis-U-Net-Albedo-Code-main/) | Notebook entry point for the GOES-to-MODIS U-Net workflow. |
| [`GOES-Modis-Albedo-Postprocessing-main/`](GOES-Modis-Albedo-Postprocessing-main/) | Converts predictions to rasters, evaluates them, and produces comparison figures. |
| [`functions/`](functions/) | Shared preprocessing, modeling, albedo-calculation, and plotting functions. |
| [`goes_data_download/`](goes_data_download/) | Downloads NOAA GOES ABI Level-2 LSAC files from the public S3 archive. |
| [`modis_data_download/`](modis_data_download/) | Searches and downloads NASA MODIS products through `earthaccess`. |
| [`sentinel_2/`](sentinel_2/) | Sentinel-2 download, harmonization, cloud/TSI analysis, albedo production, and visualization notebooks. |
| [`sentinel_2/s2_functions/`](sentinel_2/s2_functions/) | Reusable Sentinel-2, TSI, CDF, albedo, BRDF, and terrain-correction scripts. |
| [`s2_modis_downscaling/`](s2_modis_downscaling/) | Second U-Net stage for downscaling MODIS-scale predictions to a Sentinel-2 target grid. |
| [`MNRAS.mplstyle`](MNRAS.mplstyle) | Matplotlib style configuration used by several notebooks. |
| [`environment.yml`](environment.yml) | Conda environment for geospatial processing, notebooks, machine learning, and visualization. |

Each major directory contains its own README with inputs, configuration points, execution order, outputs, and known assumptions.

## Data sources

The workflows expect externally downloaded data. Large source and output files are intentionally not stored in the repository.

| Dataset | Typical role in this project |
|---|---|
| NOAA GOES-R ABI Level-2 LSAC | High-temporal-resolution land-surface albedo predictor. |
| MODIS MCD43A3 v061 | Daily black-sky and white-sky shortwave albedo used to derive blue-sky albedo and train the first U-Net. |
| MODIS/CERES aerosol data | Aerosol optical depth used when estimating the diffuse skylight fraction. |
| Sentinel-2 L2A | Surface reflectance, scene classification, aerosol optical thickness, and snow information at 10-20 m resolution. |
| Copernicus DEM GLO-30 | Slope and aspect for optional SCS+C topographic correction. |
| ARM SAIL Total Sky Imager products | Thin-cloud and opaque-cloud fractions for matching and evaluating Sentinel-2 scenes. |
| East River AOI shapefile | Spatial extent for clipping and common-grid construction. |
| Colorado state boundary | Supporting geometry for regional MODIS preprocessing and mapping. |
| SAIL field albedo observations | Point-scale comparison and ground-truth visualization. |

Users are responsible for complying with each data provider's access, attribution, and redistribution terms.

## Software requirements

The supplied environment uses Python 3.10 and packages from `conda-forge`. It includes TensorFlow, PyTorch, scikit-learn, JupyterLab, Rasterio, Rioxarray, Xarray, GDAL, GeoPandas, Earthaccess, Planetary Computer clients, Dask, NetCDF/HDF support, and plotting libraries.

A CUDA-capable GPU is recommended for U-Net training, but preprocessing and small test runs can be completed on a CPU. Memory and storage requirements depend strongly on the date range, number of satellite scenes, and whether intermediate rasters are retained.

### Create the environment

From the repository root:

```bash
mamba env create -f environment.yml
conda activate sail_env
```

Conda can be used instead of Mamba:

```bash
conda env create -f environment.yml
conda activate sail_env
```

Register the environment as a Jupyter kernel if needed:

```bash
python -m ipykernel install --user --name sail_env --display-name "Python (sail_env)"
```

Start JupyterLab with:

```bash
jupyter lab
```

### Optional packages

`auto_open_cdf.py` can inspect NASA CDF files through `spacepy.pycdf`. Install SpacePy separately when NASA CDF support is required. NetCDF-formatted `.cdf` files can be opened with the packages already listed in `environment.yml`.

## Required configuration before running

The current research code is not a packaged command-line application. Several scripts and notebooks contain absolute paths from Boise State and NERSC environments. Before importing the helper modules or executing notebook cells, update all path constants for your system.

Search the repository for these path prefixes:

```bash
grep -R "/bsuhome/tnde\|/global/cfs\|~/geoscience" -n \
  --include="*.py" --include="*.ipynb" .
```

At minimum, configure the following:

1. **AOI geometry:** path to the East River or replacement study-area shapefile.
2. **GOES directories:** raw LSAC archive, clipped output, masked output, and prediction output directories.
3. **MODIS directories:** raw MCD43 files, clipped products, blue-sky albedo products, and invalid-date JSON file.
4. **Sentinel-2 directories:** output rasters, TSI `.cdf` files, cloud-fraction CSV, and optional DEM or ancillary products.
5. **Model directories:** checkpoints, saved Keras models, NumPy predictions, training histories, and GeoTIFF output folders.
6. **Lookup tables:** verify that `GOES-Modis-Data-Preprocessing-main/sw_lut.csv` is reachable from the configured working directory.
7. **Helper imports:** replace notebook `function_path` values with the repository's local `functions/` or `sentinel_2/s2_functions/` path.
8. **Plot style:** point `plt.style.use(...)` to the included `MNRAS.mplstyle`, or remove that line.

Some helper modules read example rasters and shapefiles while they are being imported. Those configured paths must therefore exist before an import such as `from data_preprocessing import *` or `from plot_fxns import *` succeeds.

## Recommended data layout

The scripts can be adapted to another layout, but the following organization keeps inputs and generated files separate from source code:

```text
project_workspace/
├── repository/                         # this Git repository
├── data/
│   ├── boundaries/
│   │   ├── east_river/
│   │   └── colorado/
│   ├── goes/
│   │   ├── raw/ABI-L2-LSAC/YYYY/DDD/HH/
│   │   ├── clipped_500m/
│   │   └── masked_500m/
│   ├── modis/
│   │   ├── raw/
│   │   ├── clipped/
│   │   ├── blue_sky_500m/
│   │   └── interpolated_500m/
│   ├── sentinel2/
│   │   ├── albedo_20m/
│   │   └── tsi/
│   └── field/
└── results/
    ├── unet_goes_modis/
    ├── unet_modis_s2/
    ├── predictions/
    └── figures/
```

The root `.gitignore` excludes common raw-data, raster, NetCDF, image, archive, and web-map outputs. Confirm that important generated results are backed up outside the repository.

## End-to-end execution guide

### 1. Download GOES files

Edit the date range, satellite bucket, product prefix, and output directory in `goes_data_download/download_goes_data.py`, then run:

```bash
python goes_data_download/download_goes_data.py
```

The downloader preserves the NOAA S3 hierarchy `product/YYYY/DDD/HH/file` below the configured local directory.

### 2. Download MODIS files

Authenticate with NASA Earthdata and configure the product, temporal range, bounding box, and output directory in `modis_data_download/download_modis_data.py`:

```bash
python modis_data_download/download_modis_data.py
```

For the first-stage U-Net, the primary albedo product is MCD43A3. Other short names in the script support AOD, BRDF, quality, and surface-reflectance experiments.

### 3. Preprocess GOES and MODIS

Open:

```text
GOES-Modis-Data-Preprocessing-main/GOES_Modis_U-Net_Data_Preprocessing.ipynb
```

Run the notebook after updating all paths. Its principal operations are:

- construct AOI bounds in the required CRSs;
- clip and reproject GOES LSA and DQF data;
- align GOES rasters to the MODIS grid using bilinear resampling for LSA and nearest-neighbor resampling for DQF;
- filter invalid scenes;
- clip and reproject MODIS rasters;
- create GOES rasters with MODIS-matched missing-data masks; and
- inspect spatial alignment and missing dates.

The GOES input directory is expected to follow `YYYY/DDD/HH`. The workflow currently selects observations from the 18 and 19 UTC folders, using an index offset to approximate the desired overpass time.

### 4. Calculate MODIS blue-sky albedo

Open:

```text
GOES-Modis-Data-Preprocessing-main/calculate_modis_blue_sky_albedo.ipynb
```

The notebook reads MCD43 black-sky and white-sky shortwave albedo, estimates a diffuse skylight ratio from aerosol optical depth and solar zenith angle, and writes daily blue-sky albedo GeoTIFFs.

The calculation is:

```text
blue_sky = (1 - diffuse_fraction) * black_sky
           + diffuse_fraction * white_sky
```

MODIS scaled integer albedo values are multiplied by `0.001` in the helper functions.

### 5. Train the GOES-to-MODIS U-Net

Use either:

- `GOES-Modis-U-Net-Albedo-Code-main/run_unet.ipynb` for an interactive run, or
- `python functions/unet1_main.py` for the scripted pipeline.

Before running, set:

- training, validation, and test date ranges;
- masked or unmasked GOES input selection;
- invalid dates;
- model/checkpoint/result paths; and
- `load_weights_bool` and `load_model` behavior.

The current East River rasters are approximately `21 x 19` pixels and are padded to `24 x 24` for the network. If the AOI or target grid changes, revise the padding and cropping logic together. Inputs are stacked as `(samples, height, width, channels)`, while MODIS validity masks are used as per-pixel sample weights.

Example scripted run:

```bash
python functions/unet1_main.py
```

### 6. Evaluate and visualize first-stage predictions

Open:

```text
GOES-Modis-Albedo-Postprocessing-main/Final-Visualizations.ipynb
```

This notebook can:

- convert `.npy` predictions to georeferenced GeoTIFFs;
- generate interpolated MODIS comparison rasters;
- calculate masked R-squared and RMSE;
- plot prediction and target rasters by date;
- examine MODIS missing-pixel percentages;
- compare selected rasters with SAIL field observations; and
- prepare multi-source figures.

Use the exact saved test-date list and invalid-date configuration from model training so that array indices remain aligned with calendar dates.

### 7. Produce Sentinel-2 20 m albedo

See `sentinel_2/README.md` and `sentinel_2/s2_functions/README.md` before running this stage. The principal production workflow is implemented in:

```text
sentinel_2/s2_functions/s2_20m_download_final.py
```

An alternate terrain-correction and MODIS BRDF-fusion workflow is provided in:

```text
sentinel_2/s2_functions/s2_20m_albedo_topocorr_brdf_fusion.py
```

These workflows query Sentinel-2 L2A through the Microsoft Planetary Computer, harmonize affected processing baselines, mask clouds and invalid classes, identify snow, normalize directional reflectance, and write daily 20 m albedo and quality-assurance products.

### 8. Train the MODIS-to-Sentinel-2 U-Net

Configure the file globs, cloud-fraction CSV, excluded dates, model paths, and output directories in:

```text
s2_modis_downscaling/modis_s2_unet2.py
```

Then run:

```bash
python s2_modis_downscaling/modis_s2_unet2.py
```

The script matches predicted MODIS files and Sentinel-2 albedo files by date, filters scenes using interpolated cloud fraction, reprojects MODIS to the exact Sentinel-2 grid, trains with a target-validity mask, and writes predictions using the Sentinel-2 georeferencing.

## Expected outputs

Depending on the stages run, outputs include:

- clipped and reprojected GOES LSA GeoTIFFs;
- MODIS black-sky, white-sky, and calculated blue-sky albedo rasters;
- GOES rasters masked to the corresponding MODIS valid-data footprint;
- Keras models, checkpoint weights, training-history JSON, and test-date JSON;
- NumPy arrays of first- and second-stage predictions;
- georeferenced GOES-to-MODIS and MODIS-to-Sentinel prediction rasters;
- daily Sentinel-2 SW, VIS, NIR, BSA, WSA, and blue-sky albedo rasters;
- hard and soft snow-mask variants and quality-assurance layers;
- cloud-fraction tables and TSI/Sentinel-2 matching diagnostics; and
- R-squared, RMSE, time-series, map, and comparison figures.

## Important assumptions

- **Study CRS:** many workflows target EPSG:32613, which is appropriate for the current East River AOI. Select a suitable projected CRS for another study area.
- **GOES scale factor:** GOES LSA is divided by `10000` during preprocessing.
- **MODIS scale factor:** MODIS albedo values are multiplied by `0.001` when blue-sky albedo is calculated.
- **Filename conventions:** matching depends on GOES timestamps and MODIS/Sentinel dates embedded in filenames. Preserve or update the parser logic when renaming files.
- **Observation time:** GOES preprocessing currently favors an 18 UTC scene and uses a 19 UTC fallback for selected dates.
- **Raster dimensions:** the first U-Net's fixed padding assumes a `21 x 19` native grid. The second model can pad batches to a multiple of eight, but still expects paired rasters with identical target grids.
- **Missing data:** MODIS and Sentinel-2 target masks determine which pixels contribute to the loss and metrics.
- **Cloud screening:** scene-level cloud percentages are only a pre-filter. Pixel-level SCL, NDSI, TSI, or cloud-fraction rules are applied elsewhere in the workflow.
- **Notebooks are stateful:** execute cells in order after restarting the kernel, especially when paths, global arrays, or matched file lists are changed.

## Reproducibility checklist

Before reporting a run, record:

- repository commit;
- environment export or `environment.yml` checksum;
- data product names and collection versions;
- acquisition date range and AOI geometry;
- CRS, target resolution, and resampling methods;
- invalid-date and cloud-screening rules;
- train/validation/test date split;
- masked versus unmasked GOES selection;
- model configuration, random seed, and checkpoint used;
- output naming convention; and
- any edits made to hard-coded thresholds or coefficients.

## Troubleshooting

### Import fails before a function is called

`data_preprocessing.py`, `plot_fxns.py`, and some Sentinel-2 modules initialize shapefiles or example datasets at import time. Update their path constants first and verify the files exist.

### `NoDataInBounds` or empty clips

Check that the AOI and raster overlap after reprojection. Print both bounds and CRSs before clipping. Do not assign a CRS without transforming coordinates.

### Prediction and target shapes differ

Confirm that both rasters use the same CRS, affine transform, width, height, and pixel ordering. Use `rio.reproject_match` for alignment. For the first U-Net, also verify that the native array is `21 x 19` before fixed padding.

### No paired dates are found

Check filename patterns and date parsers. GOES uses a timestamp token beginning with `_sYYYYDDDHHMM...`; MODIS files are generally matched with a seven-digit `YYYYDDD`; Sentinel-2 products are matched with calendar dates such as `YYYY-MM-DD`.

### TensorFlow runs out of memory

Use fewer scenes, reduce batch size in the model call, close unused datasets, or run on a GPU node with more memory. The current scripts do not automatically choose a batch size based on available resources.

### Planetary Computer assets return authorization errors

Open the STAC client with the Planetary Computer signing modifier and sign item assets before Rasterio access. Recreate stale signed URLs rather than reusing them after they expire.

## Credits

The original U-Net workflow acknowledges Dr. Utkarsh Mital for U-Net support, Dr. Daniel Feldman for Earth-science workflow references, Dr. William Rudisill for coding and Earth-science expertise, and Lawrence Berkeley National Laboratory for computational infrastructure.

Portions of the U-Net encoder, decoder, and model-construction code are identified in the source as modifications of an MIT-licensed implementation by Vidushi Bhatia. Preserve the attribution and embedded license notice when redistributing those portions.

## Citation

When this repository supports a publication, thesis, presentation, or derived dataset, cite the associated project paper or data release and record the repository commit used. A formal `CITATION.cff` file is not currently included.

## License

A repository-level license file is not currently included. Unless a separate agreement applies, do not assume permission to redistribute or reuse the repository as a whole. Individual source sections may contain their own attribution or license notices, which must be retained.