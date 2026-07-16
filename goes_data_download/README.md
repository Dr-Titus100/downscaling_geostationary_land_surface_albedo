# GOES ABI LSAC Data Download

This directory contains the script used to download NOAA GOES Advanced Baseline Imager Level-2 Land Surface Albedo files from the public AWS S3 archive.

## Contents

| File | Description |
|---|---|
| `download_goes_data.py` | Lists and downloads every object under the selected product, date, and hour prefixes. |

## Current defaults

The script is currently configured with:

```text
start date:      04-01-2023
end date:        06-16-2023
bucket:          noaa-goes17
product prefix:  ABI-L2-LSAC
local directory: /global/cfs/cdirs/m3779/felix/GOES/data
```

The start date is inclusive and the end date is non-inclusive. With the defaults, the final date considered is June 15, 2023.

## Requirements

Activate the project environment or install Boto3:

```bash
conda activate sail_env
```

The NOAA public archive can normally be listed and downloaded without AWS credentials. If the local AWS configuration forces authenticated access, use an unsigned S3 client or remove the conflicting profile settings.

## Configuration

Edit `main()` and `download_files_range(...)` before running.

### Date range

Dates use `MM-DD-YYYY`:

```python
start_date = "09-01-2021"
end_date = "06-16-2023"  # non-inclusive
```

### Satellite bucket

Select the bucket that contains the required mission and period:

```python
bucket_name = "noaa-goes17"
```

Do not assume that every product and date exists in every GOES bucket. Confirm the satellite, coverage region, and operational period for the intended study.

### Product

The current prefix is:

```python
base_s3_path = "ABI-L2-LSAC"
```

Changing the product requires checking variable names, scale factors, DQF definitions, and preprocessing assumptions elsewhere in the repository.

### Local output

Set a writable directory with enough storage:

```python
local_path_dir = "/path/to/data/goes/raw"
```

The script appends each full S3 object key to this path.

## Output structure

For each day and all 24 hours, the script requests a prefix of the form:

```text
ABI-L2-LSAC/YYYY/DDD/HH/
```

Files are saved while preserving that hierarchy:

```text
/path/to/data/goes/raw/
└── ABI-L2-LSAC/
    └── 2023/
        └── 091/
            ├── 00/
            ├── 01/
            └── ...
```

This structure is compatible with the directory traversal in `../functions/data_preprocessing.py`.

## Run the downloader

From the repository root:

```bash
python goes_data_download/download_goes_data.py
```

The script prints each requested prefix and a confirmation for each downloaded file.

## How the script works

1. Creates an S3 client.
2. Converts start and end strings to `datetime` objects.
3. Iterates through each date in the half-open interval.
4. Converts the date to year and zero-padded day of year.
5. Iterates through hours `00` to `23`.
6. Lists objects with the S3 `list_objects_v2` paginator.
7. Creates the corresponding local directory.
8. Downloads every listed object.

Pagination prevents a single list request from being limited to the first page of results.

## Storage and runtime considerations

The script downloads all LSAC files in every hour, while the preprocessing workflow later uses only selected files near the MODIS comparison time. A multi-year request can therefore be large.

Before starting a full download:

- run one day as a test;
- estimate the number and size of objects;
- confirm quota and filesystem performance;
- consider downloading only the required hours; and
- avoid running duplicate jobs against the same output tree.

A simple reduced-hour modification is:

```python
for hour in (18, 19):
    ...
```

Only make this change after confirming the preprocessing time-selection logic.

## Resume behavior

The current script does not explicitly skip files that already exist. Re-running it may overwrite or re-download existing files depending on Boto3 behavior.

For a resumable workflow, add an existence and size check before `download_file(...)`. Keep any such code change separate from the documented baseline so the download provenance remains clear.

## Error handling

`download_file(...)` catches `S3TransferFailedError` and prints the failed object key. Other errors, including network failures, permission errors, or local disk errors, may stop the program.

After a run, search logs for failures and compare local file counts with S3 listings.

## Validation

Inspect at least one downloaded file:

```python
import xarray as xr

path = "/path/to/a/downloaded/OR_ABI-L2-LSAC-....nc"
ds = xr.open_dataset(path)
print(ds)
print(ds["LSA"])
print(ds["DQF"])
print(ds["goes_imager_projection"])
```

Confirm that:

- the file opens without corruption;
- `LSA` and `DQF` are present;
- the projection metadata is present;
- the timestamp matches the parent directory; and
- the AOI is within the satellite's coverage.

## Common problems

### No files are listed

Check the satellite bucket, product name, date, day-of-year calculation, and zero-padded hour. The product may not exist for that mission or period.

### Credential or access error

The archive is public, but the client may be using a restrictive AWS profile. Configure unsigned public access if required.

### Download fills the filesystem

Stop the process, check remaining space, and narrow the date range or hour list. Raw satellite archives should be stored outside the Git repository.

### Preprocessing does not find the files

Verify that `goes_data_dir` in `../functions/data_preprocessing.py` points to the `ABI-L2-LSAC` directory that directly contains the year folders.

## Next step

After validation, continue with:

```text
../GOES-Modis-Data-Preprocessing-main/GOES_Modis_U-Net_Data_Preprocessing.ipynb
```

That notebook selects the required observation times, applies DQF checks, clips the AOI, scales LSA, and aligns each accepted scene to the MODIS grid.