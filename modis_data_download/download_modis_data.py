import matplotlib.pyplot as plt
from shapely.geometry import box
import geopandas as gpd
import earthaccess
import datetime
from pathlib import Path


def print_modis_shapefile_details():
    colorado_shapefile_path = '/global/homes/f/feyu39/MODIS/Colorado_State_Boundary.shp'

    Boundary = gpd.read_file(colorado_shapefile_path)
    Boundary_reproj = Boundary.to_crs('EPSG:4326')
    Boundary_box = [box(*Boundary_reproj.total_bounds)]

    bounds = Boundary_reproj.total_bounds  # Returns (minx, miny, maxx, maxy)

    # Print the bounding box
    print(f"Bounding Box Coordinates:")
    print(f"Min Longitude: {bounds[0]}")
    print(f"Min Latitude: {bounds[1]}")
    print(f"Max Longitude: {bounds[2]}")
    print(f"Max Latitude: {bounds[3]}")


def download_modis_data(start_date, end_date, product_shortname, output_dir):
    auth = earthaccess.login(strategy="interactive", persist=True)

    # Download MODIS Aqua Terra 500m BRDF Albedo Data
    # Bounding_Box: (lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat)
    results = earthaccess.search_data(
        short_name=product_shortname,
        bounding_box=(-109.06025710058428, 36.99242725707592,
                      -102.04152713693466, 41.003445924840214),
        temporal=(start_date, end_date),
        count=-1
    )
    # Generate local directories for the files

    files = earthaccess.download(results, output_dir)


search_start_date = "09/01/2021"
search_end_date = "09/02/2021"
brdf_product_output_dir = "./modis/brdf-product-data/"
surface_reflectance_output_dir = "/surface-reflectance-data/"

albedo_product_shortname = "MCD43A3"
aod_product_shortname = "MCD19A2"
brdf_product_shortname = "MCD43A1"
brdf_quality_shortname = "MCD43A2"
surface_reflectance_shortname = "MOD09GA"
download_modis_data(search_start_date, search_end_date, surface_reflectance_shortname, surface_reflectance_output_dir)
# print_modis_shapefile_details()