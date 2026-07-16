# Project Boundary and Mapping Files

This directory contains the current East River and Colorado boundary files used for study-area visualization, regional context maps, and path migration away from older boundary locations.

## Contents

| File | Description |
|---|---|
| `East_River.kml` | East River study-area boundary with multiple KML layers used by the mapping notebook. |
| `Colorado_State_Boundary.geojson` | Colorado boundary in a portable single-file format suitable for GeoPandas and web mapping. |
| `Colorado_State_Boundary.kml` | Colorado boundary for KML-compatible GIS software and Google Earth. |
| `Colorado_State_Boundary.csv` | Tabular export associated with the Colorado boundary; inspect its fields before treating it as polygon geometry. |
| `east_river_boundary.ipynb` | Notebook for inspecting KML layers and producing East River, Colorado, surrounding-state, inset, compass, scale-bar, and contextual maps. |

## Relationship to legacy boundary files

Some existing code still points to:

```text
GOES-Modis-Data-Preprocessing-main/shapefile_colorado/
```

That legacy directory contains a shapefile bundle and its own documentation. The files here are the current top-level mapping resources. Update code paths deliberately rather than assuming the old and new files have identical attributes, layer names, or geometry detail.

## East River KML

`East_River.kml` is read with GeoPandas. The notebook checks available layers and currently works with layer names such as:

```text
East_River.kml
East_River
```

Layer names may differ after re-exporting the KML. Inspect them before plotting:

```python
import geopandas as gpd

kml_path = "shape_files/East_River.kml"
print(gpd.list_layers(kml_path))
```

Read a selected layer with:

```python
study_area = gpd.read_file(kml_path, layer="East_River")
print(study_area.crs)
print(study_area.total_bounds)
```

KML coordinates are normally longitude and latitude in WGS84. Only use `set_crs("EPSG:4326")` when the coordinates are already WGS84 but the CRS metadata is missing. Use `to_crs(...)` when transforming coordinates.

## Colorado boundary formats

The Colorado boundary is supplied in several formats for compatibility:

- use **GeoJSON** for GeoPandas, portable analysis, and web maps;
- use **KML** for Google Earth and KML-compatible software; and
- use **CSV** only after confirming that its fields provide the geometry or attributes needed by the workflow.

Do not combine duplicate representations as separate boundary features in the same analysis.

## `east_river_boundary.ipynb`

The notebook includes workflows for:

- reading and listing KML layers;
- plotting polygon, line, point, and other geometry types;
- comparing outer and inner East River boundaries;
- displaying Colorado and surrounding states;
- adding Texas to the regional context when required;
- marking the East River study location;
- drawing leader lines between a regional marker and a zoomed inset;
- formatting longitude and latitude axes;
- adding grid lines, north arrows, and map scale bars;
- positioning compasses on main and inset maps;
- using Contextily basemaps; and
- creating Folium or static Matplotlib mapping products.

The notebook imports packages including GeoPandas, Matplotlib, Folium, Contextily, PyProj, Pandas, and Shapely.

## Recommended setup

Run from the repository root:

```bash
conda activate sail_env
jupyter lab shape_files/east_river_boundary.ipynb
```

Update the hard-coded KML path in the notebook to a portable repository-relative path:

```python
from pathlib import Path

repo_root = Path("/path/to/downscaling_geostationary_land_surface_albedo")
kml_path = repo_root / "shape_files" / "East_River.kml"
```

Do not commit notebook-generated map tiles, HTML files, or raster outputs unless they are intentionally part of the source repository.

## CRS and geometry checks

Before using a boundary, verify:

```python
print(boundary.crs)
print(boundary.total_bounds)
print(boundary.geometry.geom_type.value_counts())
print(boundary.is_valid.all())
```

For projected distance, scale, buffering, or area calculations, reproject from WGS84 to an appropriate projected CRS. The East River workflows commonly use EPSG:32613.

## Reading the Colorado GeoJSON

```python
import geopandas as gpd

colorado = gpd.read_file("shape_files/Colorado_State_Boundary.geojson")
print(colorado.crs)
print(colorado.total_bounds)
```

Reproject to the target raster before clipping:

```python
colorado_for_raster = colorado.to_crs(raster.rio.crs)
```

Do not relabel coordinates with `set_crs(...)` when a true coordinate transformation is required.

## Maintenance rules

- Keep equivalent Colorado formats synchronized when replacing the boundary.
- Record the source, download date, original CRS, and processing history.
- Preserve KML layer names or update the notebook after re-export.
- Validate geometries after conversion between formats.
- Do not edit binary or structured geospatial formats as ordinary text unless the format and consequences are understood.
- Update all README and code references when moving boundary files.

## Validation checklist

Confirm that:

- every boundary opens without errors;
- CRS metadata is present or correctly assigned;
- bounds correspond to Colorado or East River as expected;
- KML layers are not empty;
- regional and inset maps use compatible coordinate systems;
- map scales are calculated in projected units;
- axes are labeled according to the actual CRS; and
- context basemaps are added only after transforming to their required projection.
