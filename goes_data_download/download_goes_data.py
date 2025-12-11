import boto3
import pdb
import os
from datetime import datetime, timedelta

HOURS_PER_DAY = 24


def main():
    s3_client = boto3.client('s3')
    start_date = '04-01-2023'
    end_date = '06-16-2023'

    download_files_range(start_date, end_date, s3_client)


def download_file(bucket_name, object_to_download, local_path, s3_client):
    # Make directory to download to
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    try:
        s3_client.download_file(bucket_name, object_to_download, local_path)
        print(f'Downloaded {object_to_download} to {local_path}')
    except boto3.exceptions.S3TransferFailedError as e:
        print(f"Failed to download {object_to_download}: {e}")


def download_files_range(start_date, end_date, s3_client):
    # start date inclusive, end date non-inclusive
    start_date_dt = datetime.strptime(start_date, '%m-%d-%Y')
    end_date_dt = datetime.strptime(end_date, '%m-%d-%Y')
    total_days = (end_date_dt - start_date_dt).days

    bucket_name = 'noaa-goes17'
    local_path_dir = '/global/cfs/cdirs/m3779/felix/GOES/data'
    base_s3_path = 'ABI-L2-LSAC'

    os.makedirs(local_path_dir, exist_ok=True)
    # For every date within the range, download the data
    for single_date in (start_date_dt + timedelta(n) for n in range(total_days)):
        year = single_date.year
        day_of_year = single_date.timetuple().tm_yday
        for hour in range(HOURS_PER_DAY):
            # Pad single digits to have 0 to conform to aws bucket naming convention
            padded_hour = f'{hour:02}'
            prefix = f'{base_s3_path}/{year}/{day_of_year:03d}/{padded_hour}/'

            # List files in the bucket + prefix
            files = list_files_in_prefix(bucket_name, prefix, s3_client)

            # Download each file
            for file_name in files:
                # Specify a location path to download the data
                local_path = os.path.join(
                    local_path_dir, file_name)
                download_file(bucket_name, file_name, local_path, s3_client)


def list_files_in_prefix(bucket_name, prefix, s3_client):
    # Paginator spreads requests over time to prevent throttling
    # Paginator contains metadata instead of all the actual data
    paginator = s3_client.get_paginator('list_objects_v2')
    print(prefix)
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    files = []
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                files.append(obj['Key'])
    return files


if __name__ == "__main__":
    main()
