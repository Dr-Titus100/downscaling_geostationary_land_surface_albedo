# Colorado State Boundary Data

This directory stores the Colorado state boundary in several common geospatial formats. The preprocessing notebook uses the boundary to clip or inspect MODIS data at the state scale before creating East River products.

## Contents

| Path | Format and use |
|---|---|
| `Colorado_State_Boundary.csv` | Lightweight tabular export associated with the boundary dataset. Inspect its fields before assuming it contains complete polygon geometry. |
| `Colorado_State_Boundary.geojson` | Self-contained GeoJSON representation suitable for GeoPandas, web maps, and GIS software. |
| `Colorado_State_Boundary.kml` | KML representation for Google Earth and other KML-compatible tools. |
| `Colorado_State_Boundary/` | ESRI Shapefile bundle containing geometry, attributes, index, projection, and encoding files. |

The same boundary is supplied in multiple formats for compatibility. Choose one format for a workflow rather than combining them.

## Recommended use in Python

### GeoJSON

```python
import geopandas as gpd

colorado = gpd.read_file(
    "GOES-Modis-Data-Preprocessing-main/shapefile_colorado/Colorado_State_Boundary.geojson"
)
print(colorado.crs)
print(colorado.total_bounds)
```

### KML

KML driver availability depends on the GDAL/Fiona installation:

```python
import geopandas as gpd

colorado = gpd.read_file(
    "GOES-Modis-Data-Preprocessing-main/shapefile_colorado/Colorado_State_Boundary.kml",
    driver="KML",
)
```

### ESRI Shapefile

```python
import geopandas as gpd

colorado = gpd.read_file(
    "GOES-Modis-Data-Preprocessing-main/shapefile_colorado/Colorado_State_Boundary/Colorado_State_Boundary.shp"
)
```

See the nested README for the role of each shapefile component.

## Use in this project

The GOES-MODIS preprocessing notebook reprojects this boundary to the native MODIS CRS and EPSG:32613, constructs a bounding geometry, and uses it for regional clipping or visualization.

Typical sequence:

```python
colorado = gpd.read_file(path_to_boundary)
colorado_on_modis = colorado.to_crs(modis_raster.rio.crs)
colorado_utm13 = colorado.to_crs(epsg=32613)
```

The East River AOI, not the entire Colorado polygon, is the final spatial target for the main downscaling model.

## CRS and geometry checks

Before clipping a raster:

```python
print(colorado.crs)
print(colorado.total_bounds)
print(colorado.is_valid.all())
```

Then reproject the vector to the raster CRS:

```python
colorado_for_clip = colorado.to_crs(raster.rio.crs)
```

Do not use `set_crs(...)` to relabel coordinates that are actually in another CRS.

## Maintenance rules

- Keep equivalent formats synchronized if the boundary is replaced.
- Record the source, download date, original CRS, and any geometry simplification in project metadata.
- Preserve feature attributes unless there is a documented reason to remove them.
- Do not rename only one member of the shapefile bundle.
- Avoid committing derived rasters or maps to this directory.

## Choosing a format

- Use **GeoJSON** for portability and web-based workflows.
- Use **KML** for Google Earth visualization.
- Use the **Shapefile bundle** for compatibility with older GIS tools and the paths already used in the preprocessing code.
- Use the **CSV** only after confirming that its fields contain the information required by the intended workflow.