# Colorado State Boundary Shapefile Bundle

This directory contains the component files that together form the ESRI Shapefile version of the Colorado state boundary.

## Files

| File | Purpose |
|---|---|
| `Colorado_State_Boundary.shp` | Main binary geometry file. This is the file passed to GeoPandas or GIS software. |
| `Colorado_State_Boundary.shx` | Positional index linking features to geometry records. |
| `Colorado_State_Boundary.dbf` | Attribute table associated with the polygon feature or features. |
| `Colorado_State_Boundary.prj` | Coordinate reference system definition in Well-Known Text form. |
| `Colorado_State_Boundary.cpg` | Character-encoding declaration for text stored in the DBF table. |

A shapefile is a multi-file dataset. All five files should remain together and retain the same base name.

## Reading the boundary

```python
import geopandas as gpd

boundary = gpd.read_file(
    "GOES-Modis-Data-Preprocessing-main/shapefile_colorado/"
    "Colorado_State_Boundary/Colorado_State_Boundary.shp"
)

print(boundary)
print(boundary.crs)
print(boundary.total_bounds)
```

## Project use

The preprocessing workflow uses this boundary for Colorado-scale MODIS clipping and visualization. It is reprojected to match the raster being processed before clipping:

```python
boundary_for_raster = boundary.to_crs(raster.rio.crs)
clipped = raster.rio.clip(boundary_for_raster.geometry, all_touched=True)
```

The project's final East River model grid is created from a separate East River AOI shapefile.

## Integrity checks

After copying or downloading the repository, verify:

```python
assert boundary.crs is not None
assert not boundary.empty
assert boundary.geometry.notna().all()
assert boundary.is_valid.all()
```

If `.shx`, `.dbf`, or `.prj` is missing, the `.shp` file may fail to load or may load without required attributes or CRS information.

## Maintenance rules

- Do not move, rename, or replace only one sidecar file.
- Do not edit the binary components with a text editor.
- If the boundary is regenerated, export the complete bundle together.
- Record the source and processing history in the parent directory README or project metadata.
- If a more portable single-file format is preferred, use the GeoJSON copy in the parent directory.