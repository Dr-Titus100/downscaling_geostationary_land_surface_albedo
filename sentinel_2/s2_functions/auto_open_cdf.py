#!/usr/bin/env python3
"""
auto_open_cdf.py  —  Detect and open .cdf files (NetCDF-3/4 vs NASA CDF)

Usage:
  python auto_open_cdf.py path/to/file.cdf

What it does:
- Detects file type via magic bytes (HDF5, classic NetCDF) and then by trying libraries.
- Opens with:
    * netCDF4/xarray for NetCDF-3 and NetCDF-4 (HDF5-backed)
    * spacepy.pycdf for NASA CDF
- Prints a quick inventory of variables and shapes.
"""

from pathlib import Path
import sys

HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"   # NetCDF-4 uses HDF5 container
NETCDF_CLASSIC_MAGICS = {b"CDF\x01", b"CDF\x02"}  # NetCDF-3 classic/64-bit
# Note: NASA CDF headers also start with "CDF", so we use open-try fallbacks.

def sniff_header(path: Path) -> bytes:
    with path.open("rb") as f:
        return f.read(8)

def detect_format(path: Path) -> str:
    """
    Returns one of: 'netcdf4', 'netcdf3', 'maybe_nasa', or 'unknown'
    """
    header = sniff_header(path)
    if header.startswith(HDF5_MAGIC):
        return "netcdf4"
    if header[:4] in NETCDF_CLASSIC_MAGICS:
        # Could be NetCDF-3 OR NASA CDF. We'll call it 'maybe_nasa' and resolve by trying to open.
        return "maybe_nasa"
    return "unknown"

def try_open_netcdf(path: Path):
    try:
        import xarray as xr
        ds = xr.open_dataset(path.as_posix())
        # Trigger a light read of metadata to ensure it's valid
        _ = list(ds.variables)
        return ("netcdf", ds)
    except Exception:
        # Try netCDF4 directly for a second opinion
        try:
            from netCDF4 import Dataset
            ds = Dataset(path.as_posix(), "r")
            _ = list(ds.variables.keys())
            return ("netcdf4_raw", ds)
        except Exception:
            return (None, None)

def try_open_nasa_cdf(path: Path):
    try:
        import spacepy.pycdf as cdf
        f = cdf.CDF(path.as_posix())
        _ = list(f.keys())
        return ("nasa_cdf", f)
    except Exception:
        return (None, None)

def show_inventory(kind: str, handle):
    print(f"\nDetected and opened as: {kind}")
    try:
        if kind in ("netcdf", "netcdf4_raw"):
            if kind == "netcdf":
                ds = handle  # xarray.Dataset
                # print("Dimensions:", dict(ds.dims))
                print("Dimensions:", dict(ds.sizes))
                print("Variables:")
                for name, da in ds.data_vars.items():
                    shape = tuple(da.shape)
                    dtype = str(da.dtype)
                    print(f"  - {name}: shape={shape}, dtype={dtype}")
                # Small preview:
                for name, da in list(ds.data_vars.items())[:1]:
                    print(f"\nPreview of '{name}':")
                    try:
                        vals = da.values
                        print(vals.ravel()[:10])
                    except Exception as e:
                        print(f"  (preview unavailable: {e})")
            else:
                ds = handle  # netCDF4.Dataset
                print("Dimensions:", {d: len(ds.dimensions[d]) for d in ds.dimensions})
                print("Variables:")
                for name, var in ds.variables.items():
                    shape = var.shape
                    dtype = str(var.dtype)
                    print(f"  - {name}: shape={shape}, dtype={dtype}")
                # Small preview:
                for name in list(ds.variables.keys())[:1]:
                    print(f"\nPreview of '{name}':")
                    try:
                        import numpy as np
                        arr = ds.variables[name][...]
                        print(np.ravel(arr)[:10])
                    except Exception as e:
                        print(f"  (preview unavailable: {e})")
        elif kind == "nasa_cdf":
            f = handle  # spacepy.pycdf.CDF
            print("Variables:")
            for k in f.keys():
                try:
                    shape = f[k].shape
                except Exception:
                    shape = "unknown"
                print(f"  - {k}: shape={shape}, type={type(f[k]).__name__}")
            # Small preview:
            keys = list(f.keys())
            if keys:
                k0 = keys[0]
                print(f"\nPreview of '{k0}':")
                try:
                    import numpy as np
                    vals = f[k0][...]
                    print(np.ravel(vals)[:10])
                except Exception as e:
                    print(f"  (preview unavailable: {e})")
        else:
            print("Opened, but unrecognized kind for inventory.")
    except Exception as e:
        print(f"(Inventory failed: {e})")

def main():
    if len(sys.argv) != 2:
        print("Usage: python auto_open_cdf.py path/to/file.cdf")
        sys.exit(2)

    path = Path(sys.argv[1]).expanduser().resolve()
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    guess = detect_format(path)
    # Strategy:
    #  - If HDF5 magic -> try NetCDF (xarray/netCDF4)
    #  - If classic magic -> try NetCDF first; if that fails -> try NASA CDF
    #  - Unknown -> try NetCDF, then NASA CDF

    if guess == "netcdf4":
        kind, handle = try_open_netcdf(path)
        if kind:
            show_inventory(kind, handle)
            sys.exit(0)
        # Fallback to NASA CDF just in case
        kind, handle = try_open_nasa_cdf(path)
        if kind:
            show_inventory(kind, handle)
            sys.exit(0)
        print("Could not open as NetCDF-4 or NASA CDF. Is the file valid?")
        sys.exit(1)

    elif guess == "maybe_nasa":
        # Try NetCDF first; if it fails, likely NASA CDF
        kind, handle = try_open_netcdf(path)
        if kind:
            show_inventory(kind, handle)
            sys.exit(0)
        kind, handle = try_open_nasa_cdf(path)
        if kind:
            show_inventory(kind, handle)
            sys.exit(0)
        print("Header looked like classic NetCDF or NASA CDF, but neither library could open it.")
        sys.exit(1)

    else:  # 'unknown'
        # Blind try NetCDF -> NASA CDF
        kind, handle = try_open_netcdf(path)
        if kind:
            show_inventory(kind, handle)
            sys.exit(0)
        kind, handle = try_open_nasa_cdf(path)
        if kind:
            show_inventory(kind, handle)
            sys.exit(0)
        print("Unknown format and neither NetCDF nor NASA CDF handlers worked.")
        sys.exit(1)

if __name__ == "__main__":
    main()
