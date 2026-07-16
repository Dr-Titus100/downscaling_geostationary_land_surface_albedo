# MODIS Data Download

This directory contains the NASA Earthdata download script used to search and retrieve MODIS products for the albedo-processing workflow.

## Contents

| File | Description |
|---|---|
| `download_modis_data.py` | Authenticates with Earthdata through `earthaccess`, searches a temporal and spatial range, and downloads all matching granules. |

## Products referenced in the script

| Short name | Intended use in the project |
|---|---|
| `MCD43A3` | MODIS BRDF/albedo product containing black-sky and white-sky albedo. This is the primary target source for the first U-Net. |
| `MCD19A2` | Aerosol optical depth experiments. |
| `MCD43A1` | BRDF model parameters. |
| `MCD43A2` | BRDF/albedo quality information. |
| `MOD09GA` | Daily surface reflectance. The current executable line requests this product. |

Only one product is downloaded by the final function call unless the script is edited to call the downloader more than once.

## Requirements

1. Create a NASA Earthdata Login account.
2. Activate the project environment:

```bash
conda activate sail_env
```

3. Ensure `earthaccess`, GeoPandas, Shapely, and Matplotlib are available. They are included in `environment.yml`.

## Authentication

The script uses:

```python
earthaccess.login(strategy="interactive", persist=True)
```

The first run prompts for Earthdata credentials or browser authorization. Persisted credentials are stored by `earthaccess` outside the repository. Never commit usernames, passwords, tokens, `.netrc` content, or credential files.

On an HPC login node, interactive authentication may require a browser on another machine or a previously configured Earthdata credential method.

## Current defaults

```text
search start: 09/01/2021
search end:   09/02/2021
bounding box: Colorado state bounds
active product: MOD09GA
active output: /surface-reflectance-data/
```

The Colorado bounding box is:

```text
minimum longitude: -109.06025710058428
minimum latitude:    36.99242725707592
maximum longitude: -102.04152713693466
maximum latitude:    41.003445924840214
```

Review whether Earthaccess treats the temporal endpoint as inclusive for the selected collection and confirm the actual returned acquisition times.

## Configuration

Edit the variables at the bottom of `download_modis_data.py`.

### Dates

The script uses `MM/DD/YYYY`:

```python
search_start_date = "09/01/2021"
search_end_date = "06/15/2023"
```

### Product

For the first-stage albedo target:

```python
product_shortname = "MCD43A3"
```

For other products, use the corresponding short name and verify the expected variables, scale factors, spatial resolution, and quality layers.

### Output directory

Use an absolute or project-relative path with sufficient space:

```python
output_dir = "/path/to/data/modis/raw/MCD43A3"
```

The script passes this directory to `earthaccess.download(...)`.

### Spatial extent

`download_modis_data(...)` currently contains a hard-coded Colorado bounding box. Replace it with the study area or another regional extent as needed:

```python
bounding_box=(min_lon, min_lat, max_lon, max_lat)
```

The coordinates must be longitude/latitude in WGS84 order.

## Optional boundary inspection

`print_modis_shapefile_details()` reads a Colorado boundary shapefile, reprojects it to EPSG:4326, and prints its total bounds.

Before calling it, update:

```python
colorado_shapefile_path = "/path/to/Colorado_State_Boundary.shp"
```

A copy of the Colorado boundary is available under:

```text
../GOES-Modis-Data-Preprocessing-main/shapefile_colorado/
```

## Run the downloader

From the repository root:

```bash
python modis_data_download/download_modis_data.py
```

The function:

1. authenticates with Earthdata;
2. searches by product short name;
3. applies the bounding box and temporal interval;
4. requests all matching results with `count=-1`; and
5. downloads the returned granules.

## Downloading multiple products

Call the function separately for each required product and keep products in separate directories:

```python
download_modis_data(
    search_start_date,
    search_end_date,
    "MCD43A3",
    "/path/to/data/modis/raw/MCD43A3",
)

download_modis_data(
    search_start_date,
    search_end_date,
    "MCD43A2",
    "/path/to/data/modis/raw/MCD43A2",
)
```

Separate directories reduce the risk of selecting the wrong product during preprocessing.

## Expected files

MODIS filenames normally encode:

```text
product.AYYYYDDD.tile.collection.production_timestamp.extension
```

Example pattern:

```text
MCD43A3.A2021244.h09v05.061.<timestamp>.hdf
```

The preprocessing code uses the `AYYYYDDD` acquisition token and product subdataset names. Preserve the original filenames.

## Validation

After download, inspect file counts and one sample granule.

With Rasterio:

```python
import rasterio as rio

path = "/path/to/a/MCD43A3...hdf"
with rio.open(path) as src:
    print(src.subdatasets)
```

For MCD43A3, verify that the required subdatasets include equivalents of:

```text
Albedo_BSA_shortwave
Albedo_WSA_shortwave
BRDF_Albedo_Band_Mandatory_Quality_shortwave
```

Also confirm:

- the expected tile covers the AOI;
- the acquisition date is within the requested period;
- all expected days or composites are present;
- downloads are not HTML error pages or zero-byte files; and
- the GDAL environment can read the HDF format.

## Common problems

### Interactive login does not open

Configure Earthdata credentials in a supported non-browser method or authenticate from a machine with browser access before running on the cluster.

### Search returns no results

Check the short name, temporal format, bounding-box coordinate order, product collection availability, and Earthaccess authentication status.

### Wrong product is downloaded

The variables define several product names, but the last executable line determines the actual request. Read that line before starting a long run.

### HDF file cannot be read

The active GDAL build may lack the required HDF driver. Use the Conda environment supplied by the project and inspect `rio.env.Env().drivers()` or `gdalinfo --formats`.

### Too many results or excessive storage use

Test a short interval first. Narrow the bounding box or date range, and place raw files outside the Git repository.

## Next step

For MCD43A3, continue with:

```text
../GOES-Modis-Data-Preprocessing-main/GOES_Modis_U-Net_Data_Preprocessing.ipynb
```

That workflow selects shortwave black-sky/white-sky bands, clips and reprojects them, and prepares them for blue-sky albedo calculation.