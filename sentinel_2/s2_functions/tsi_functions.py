# Packages
import os
import re
import io
import rasterio
import requests
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
from pathlib import Path
from datetime import timezone
from pyproj import Transformer
import matplotlib.pyplot as plt
import planetary_computer as pc
from contextlib import nullcontext
from rasterio.io import MemoryFile
from rasterio.plot import reshape_as_image
from shapely.geometry import Point, mapping
from typing import Optional, Tuple, Sequence
from rasterio import windows, features, warp
from shapely.ops import transform as shp_transform

######################################################################################
# Shape file
shapefile_path = "/bsuhome/tnde/scratch/felix/modis/East_River_SHP/ER_bbox.shp"
# AOI from shapefile (WGS84 for STAC)
Boundary = gpd.read_file(shapefile_path)
Boundary_wgs84 = Boundary.to_crs(epsg=4326)

# bounds tuple (minlon, minlat, maxlon, maxlat)
aoi_bounds_ll = tuple(Boundary_wgs84.total_bounds)
######################################################################################
######################################################################################
def dates_with_clear_sky(cdf_dir, thin_threshold = 5, opaque_threshold = 5, calendar_date = True):
    """
    Scan all .cdf files in `cdf_dir`. For each file, if there exists at least
    one time where percent_thin < threshold AND percent_opaque < threshold,
    add that day's Julian date (YYYY-DOY) to a set (unique) and return a sorted list.
    """
    cdf_dir = Path(cdf_dir)
    # print(cdf_dir)
    date_list = set()

    for fp in sorted(cdf_dir.glob("*.cdf")):
        try:
            ds = xr.open_dataset(fp)

            # Ensure required variables exist
            for v in ("time", "percent_thin", "percent_opaque"):
                if v not in ds:
                    raise KeyError(f"Missing variable '{v}' in {fp.name}")

            thin = ds["percent_thin"]
            opaque = ds["percent_opaque"]

            # Build mask: both thin and opaque <= 5%
            mask1 = (thin <= thin_threshold) & (opaque <= opaque_threshold)
            mask2 = (thin != -100) & (opaque != -100)
            mask = mask1 & mask2

            # Keep only times that meet the condition
            good_times = ds["time"].where(mask, drop=True)

            if good_times.size > 0:
                # Use the first valid time to derive the date (all times are same day)
                t0 = pd.to_datetime(good_times.values[0])
                
                if calendar_date:
                    # Calendar date
                    date_str = t0.date().isoformat()
                    date_list.add(date_str)
                else:
                    # Julian day-of-year
                    doy = f"{t0.year}-{t0.timetuple().tm_yday:03d}"
                    date_list.add(doy)
            else:
                continue
            ds.close()

        except Exception as e:
            # You can log or print errors if needed; skipping problematic files
            print(f"Skipping {fp.name}: {e}")

    # Return a sorted list (chronological by string format YYYY-DOY)
    return sorted(date_list)

######################################################################################
######################################################################################

def _to_utc(ts):
    """Return a tz-aware UTC pandas.Timestamp without changing wall time."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        # times are stored as UTC-naive; interpret as UTC
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")

def match_tsi_to_s2(
    cdf_dir,
    items,
    thin_threshold=100, #5
    opaque_threshold=100, #5
    fill_value=-100,
):
    """
    For each Sentinel-2 item in `items`, find the TSI (.cdf) file for the same date,
    filter TSI times where percent_thin <= thin_threshold and percent_opaque <= opaque_threshold
    (and not equal to `fill_value`), then select the TSI time closest to the Sentinel-2 sensing time.
    
    Hence, for each Sentinel-2 item, find TSI file for same UTC date, filter by thresholds,
    and pick the TSI time closest to the Sentinel-2 sensing time. Returns aligned lists.

    Parameters
    ----------
    cdf_dir : str or Path
        Directory containing TSI .cdf files (filenames include YYYYMMDD).
    items : Iterable
        Iterable of Sentinel-2 items, each with `.datetime` (datetime) and `.id`.
    thin_threshold : float
        Threshold for percent_thin (fraction; 0.05 == 5%).
    opaque_threshold : float
        Threshold for percent_opaque (fraction; 0.05 == 5%).
    fill_value : numeric
        Invalid/fill value to exclude (e.g., -100).

    Returns
    -------
    tsi_times : list of pandas.Timestamp
        The closest TSI timestamps (UTC) for each matched Sentinel-2 item.
    s2_items  : list
        The corresponding Sentinel-2 items (same order/length as `tsi_times`).
    tsi_files  : list[str]
        TSI file paths (or names) used for each match

    Notes
    -----
    - If a date has no valid TSI times, that Sentinel-2 item is skipped. This will most likely never happen anyway.
    - Filenames are assumed to contain an 8-digit date (YYYYMMDD).
    - We ensure that the TSI to be within ±30 minutes of the Sentinel-2 time.
    
    Steps
    -----
    - takes the Sentinel-2 'items' and a TSI '.cdf' directory,
    - keeps only dates that appear in the Sentinel-2 list,
    - for each Sentinel-2 item, opens the matching TSI file for that date, filters on 'percent_thin <= 0.05' and 'percent_opaque <= 0.05' (ignoring '-100' fill values), and
    - picks the TSI timestamp closest to the Sentinel-2 sensing time (same date),
    - returns two aligned lists: 'tsi_times' (closest timestamps) and 's2_items' (the corresponding Sentinel-2 items).

    The funtion parses the date straight from the TSI filename (e.g., '...20230105...') so it does not have to open every file.
    """
    
    cdf_dir = Path(cdf_dir)

    # S2 items grouped by UTC date (YYYY-MM-DD)
    s2_by_date = {}
    for it in items:
        s2_dt = _to_utc(it.datetime)         # <-- enforce UTC-awareness
        date_str = s2_dt.date().isoformat()
        s2_by_date.setdefault(date_str, []).append((s2_dt, it))

    # Map TSI files by date parsed from filename (YYYYMMDD)
    date_to_fp = {}
    date_pat = re.compile(r".*(\d{8}).*")
    for fp in sorted(cdf_dir.glob("*.cdf")):
        m = date_pat.match(fp.name)
        if not m:
            continue
        ymd = m.group(1)
        try:
            dt = pd.to_datetime(ymd, format="%Y%m%d")
            date_to_fp[dt.date().isoformat()] = fp
        except Exception:
            pass

    tsi_times, s2_items, tsi_files = [], [], []

    for date_str, s2_list in sorted(s2_by_date.items()):
        fp = date_to_fp.get(date_str)
        if fp is None:
            continue

        try:
            ds = xr.open_dataset(fp)
        except Exception as e:
            print(f"Skipping {fp.name}: cannot open ({e})")
            continue

        for v in ("time", "percent_thin", "percent_opaque"):
            if v not in ds:
                print(f"Skipping {fp.name}: missing '{v}'")
                ds.close()
                ds = None
                break
        if ds is None:
            continue

        thin = ds["percent_thin"]
        opaque = ds["percent_opaque"]

        # thresholds (≤) and exclude fill
        mask = (
            (thin <= thin_threshold) &
            (opaque <= opaque_threshold) &
            (thin != fill_value) &
            (opaque != fill_value)
        )

        valid_times = ds["time"].where(mask, drop=True)
        if valid_times.size == 0:
            ds.close()
            continue

        # Treat TSI times as UTC (already in UTC)
        valid_index = pd.DatetimeIndex(pd.to_datetime(valid_times.values))
        if valid_index.tz is None:
            valid_index = valid_index.tz_localize("UTC")
        else:
            valid_index = valid_index.tz_convert("UTC")

        # For each S2 item on this date: find nearest valid TSI time
        # (vectorized absolute differences)
        for s2_dt, s2_item in s2_list:
            diffs = (valid_index - s2_dt).asi8  # ns
            if diffs.size == 0:
                continue
            i_min = pd.Series(diffs).abs().idxmin()
            closest_time = valid_index[i_min]
            
            # # check max-gap here
            # if abs(closest_time - s2_dt) > pd.Timedelta("30min"): 
            #     continue  # skip this Sentinel-2 item if too far from any TSI time
                
            # Otherwise, keep the match
            tsi_times.append(pd.Timestamp(closest_time))
            s2_items.append(s2_item)
            tsi_files.append(str(fp.name))  # use fp.name if you prefer just the basename or just fp if you need the full path to the file.
        ds.close()
    return tsi_times, s2_items, tsi_files

######################################################################################
######################################################################################
"""
Steps:
-----
- builds a 'time' coordinate if it’s missing (using 'base_time' + 'time_offset', or just 'time_offset' if it’s already datetime64),
- masks the '-100' fill values,
- for each matched pair '(tsi_time, s2_item)' makes a figure with:
- Left: a short time-series (±k minutes window) of 'percent_thin' and 'percent_opaque' with a vertical line at the chosen time, plus big text labels of the exact values at that time,
- Right: the Sentinel-2 visualization (RGB if available; otherwise grayscale), with cloud % if present.
- We also auto-detects whether the 'percent_*' are fractions [0–1] or percentages [0–100] and formats labels accordingly.
"""

# ---------- helpers ----------
def _date_to_tsi_file(cdf_dir: Path, date_str: str) -> Path | None:
    """Find a TSI .cdf file in cdf_dir whose filename contains YYYYMMDD for date_str."""
    ymd = pd.to_datetime(date_str).strftime("%Y%m%d")
    for fp in sorted(Path(cdf_dir).glob("*.cdf")):
        if ymd in fp.name:
            return fp
    return None

def _s2_cloud_pct(item):
    props = getattr(item, "properties", {}) or {}
    for key in ("eo:cloud_cover", "s2:cloud_percentage", "cloud_cover"):
        if key in props:
            try:
                return float(props[key])
            except Exception:
                pass
    return None

# def _stretch_01(a, p_low=2, p_high=98):
#     """Percentile stretch to [0,1] for display."""
#     a = np.asarray(a, dtype="float32")
#     finite = np.isfinite(a)
#     if not finite.any():
#         return np.zeros_like(a, dtype="float32")
#     lo, hi = np.percentile(a[finite], [p_low, p_high])
#     if hi <= lo:
#         hi = lo + 1e-6
#     return np.clip((a - lo) / (hi - lo), 0, 1)

def _load_s2_visual(item, prefer_rgb=True, use_aws_env=True):
    """
    Return (image, label) for a Sentinel-2 item.
    - Tries to sign (Planetary Computer) if available.
    - Tries RGB (B04,B03,B02) then any single band; else falls back to 'thumbnail'.
    - Applies percentile stretch to uint8 for display.
    """
    # Try to sign for MPC
    try:
        # import planetary_computer as pc
        item = pc.sign(item)
    except Exception:
        pass

    def _stretch_01(a, p_low=2, p_high=98):
        a = np.asarray(a, dtype="float32")
        m = np.isfinite(a)
        if not m.any():
            return np.zeros_like(a, dtype="float32")
        lo, hi = np.percentile(a[m], [p_low, p_high])
        if hi <= lo:
            hi = lo + 1e-6
        return np.clip((a - lo) / (hi - lo), 0, 1)

    def _open_band(href):
        with rasterio.open(href) as src:
            return src.read(1)

    # Optionally provide AWS env for s3://
    env = rasterio.Env(AWS_NO_SIGN_REQUEST="YES") if use_aws_env else None
    ctx = env if env is not None else nullcontext()

    from contextlib import nullcontext
    with ctx:
        # Try RGB
        if prefer_rgb:
            for keys in (("B04","B03","B02"), ("red","green","blue")):
                hrefs = [item.assets[k].href for k in keys if k in item.assets]
                if len(hrefs) == 3:
                    try:
                        b = [_open_band(h) for h in hrefs]
                        rgb = np.dstack(b)               # H,W,3
                        rgb = (_stretch_01(rgb) * 255).astype("uint8")
                        return rgb, f"RGB: {keys}"
                    except Exception:
                        pass

        # Any single band
        for k, asset in item.assets.items():
            try:
                g = _open_band(asset.href)
                g = (_stretch_01(g) * 255).astype("uint8")
                return g, f"Band: {k}"
            except Exception:
                continue

        # Fallback: thumbnail (often HTTP)
        for k in ("thumbnail", "quicklook", "overview"):
            if k in item.assets:
                try:
                    import imageio.v3 as iio
                    img = iio.imread(item.assets[k].href)
                    # Ensure uint8
                    if img.dtype != np.uint8:
                        img = (_stretch_01(img) * 255).astype("uint8")
                    label = f"{k}"
                    return img, label
                except Exception:
                    pass

    raise ValueError(f"Could not read any visual asset for item {getattr(item, 'id', '<unknown>')}.")

# ---------- NEW: TSI time-series plotting per matched pair ----------
def _ensure_time_coord(ds: xr.Dataset) -> xr.Dataset:
    """Make sure ds has a proper 'time' coordinate (UTC)."""
    if "time" in ds.coords:
        return ds
    # Try to construct from base_time/time_offset
    if "time_offset" in ds:
        toff = ds["time_offset"].values
        if np.issubdtype(ds["time_offset"].dtype, np.datetime64):
            t = pd.to_datetime(toff)  # already absolute datetimes
        else:
            base = pd.Timestamp(ds["base_time"].values) if "base_time" in ds else pd.Timestamp(0, unit="s")
            # assume seconds for numeric offsets
            t = base + pd.to_timedelta(toff, unit="s")
        return ds.assign_coords(time=("time", pd.DatetimeIndex(t)))
    raise ValueError("No 'time' coordinate and cannot construct it (need 'time_offset' and optional 'base_time').")
    
# def _signed_asset_href(item, key="SCL"):
#     # Ensure we’re signing the asset object, not the string
#     signed_item = pc_sign(item)          # (re)sign the item now
#     return signed_item.assets[key].href  # fresh SAS token


def open_pc_asset_dataset(item, key="SCL"):
    """
    Open a Planetary Computer STAC asset with a fresh SAS token.
    Tries rasterio.open(href) first. If 403 persists, downloads bytes and
    opens via MemoryFile (bypasses some GDAL/vsicurl quirks).
    """
    # Fresh sign right now (avoid expired SAS)
    signed = pc.sign(item)
    href = signed.assets[key].href

    # Try direct GDAL open with a minimal env
    env = rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN='YES',
        GDAL_HTTP_MAX_RETRY='3',
        GDAL_HTTP_RETRY_DELAY='1',
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif,.tiff,.png,.jpg'
    )

    try:
        env.__enter__()
        return rasterio.open(href)  # <- caller must close
    except Exception as e:
        # Fallback: HTTP GET the signed URL, open from memory
        # (Helpful on networks where vsicurl/headers cause 403)
        try:
            resp = requests.get(href, stream=True, timeout=60)
            resp.raise_for_status()
            data = io.BytesIO(resp.content)
            mem = MemoryFile(data)
            return mem.open()
        except Exception as e2:
            # Propagate a clearer error with context
            raise RuntimeError(f"Failed to open asset '{key}'. "
                               f"Direct open error: {e}. Download fallback error: {e2}. "
                               f"Href (truncated): {href[:120]}...") from e2
    finally:
        # only close the env if direct open failed and we didn't return
        try:
            env.__exit__(None, None, None)
        except Exception:
            pass

# ---------- NEW: S2 3.5 km radius circle around ROI/AOI center and calculating cloud fraction within that circle ----------  
def plot_s2_circle_cloud_fraction(
    item,
    aoi_bounds_ll,
    s2_label,
    radius_m=3500.0,
    include_shadow=False,
    epsg_utm="EPSG:32613",
    ax=None,
):
    """
    Overlay a geodesic-approx circle (buffer in UTM) on the S2 visual and print cloud fraction
    inside the circle using SCL classes (8,9,10) [+ 3 if include_shadow=True].
    Returns the computed cloud_fraction (float in [0,1] or np.nan).
    """
    # AOI bbox center in WGS84
    minx, miny, maxx, maxy = aoi_bounds_ll
    cen_lon = (minx + maxx) / 2.0
    cen_lat = (miny + maxy) / 2.0
    pt_wgs = Point(cen_lon, cen_lat)

    # Circle (buffer) in UTM --> back to WGS84
    to_utm = Transformer.from_crs("EPSG:4326", epsg_utm, always_xy=True).transform
    to_wgs = Transformer.from_crs(epsg_utm, "EPSG:4326", always_xy=True).transform
    circle_utm = shp_transform(to_utm, pt_wgs).buffer(radius_m)
    circle_wgs = shp_transform(to_wgs, circle_utm)
    circ_bounds_ll = circle_wgs.bounds

    # Compute cloud fraction with SCL
    # scl_href = pc.sign(item.assets["SCL"].href)
    # scl_href = _signed_asset_href(item, "SCL")
    # signed = pc.sign(item) 
    # a = signed.assets["SCL"]
    # print(a.media_type, a.roles)  # expect GeoTIFF/COG-like
    # scl_href = signed.assets["SCL"].href   # string URL with SAS
    # print(scl_href)
    
    # with rasterio.open(scl_href) as scl_ds:
    # with env:
    #     with rasterio.open(scl_href) as scl_ds:
    with open_pc_asset_dataset(item, "SCL") as scl_ds:
        # Window covering the circle; read boundless so off-image areas fill with 0
        circ_bounds_scl = warp.transform_bounds("EPSG:4326", scl_ds.crs, *circ_bounds_ll, densify_pts=21)
        scl_window = windows.from_bounds(*circ_bounds_scl, transform=scl_ds.transform)
        scl_win_transform = windows.transform(scl_window, scl_ds.transform)
        scl_arr = scl_ds.read(1, window=scl_window, masked=True, boundless=True, fill_value=0)

        # Circle mask in the SCL window grid
        circle_scl_crs = shp_transform(
            Transformer.from_crs("EPSG:4326", scl_ds.crs, always_xy=True).transform,
            circle_wgs
        )
        circle_mask = features.rasterize(
            [(mapping(circle_scl_crs), 1)],
            out_shape=scl_arr.shape,
            transform=scl_win_transform,
            fill=0,
            dtype="uint8",
            all_touched=False,
        ).astype(bool)

        cloud_classes = [8, 9, 10] + ([3] if include_shadow else [])
        valid_inside = circle_mask & (~scl_arr.mask) & (scl_arr.data != 0)
        clouds_inside = valid_inside & np.isin(scl_arr.data.astype(np.uint8), np.array(cloud_classes, dtype=np.uint8))

        num_valid = int(valid_inside.sum())
        num_cloud = int(clouds_inside.sum())
        cloud_fraction = (num_cloud / num_valid) if num_valid > 0 else np.nan

    print(
        f"Circle center (lon, lat): ({cen_lon:.6f}, {cen_lat:.6f}); "
        f"cloud fraction in 3.5 km circle: "
        f"{'NaN' if np.isnan(cloud_fraction) else f'{cloud_fraction:.3%}'} "
        f"(clouds {num_cloud} / valid {num_valid})"
    )

    # Plot the visual window with circle overlay
    # vis_href = pc.sign(item.assets["visual"].href)
    with rasterio.open(item) as vds:
        circ_bounds_vis = warp.transform_bounds("EPSG:4326", vds.crs, *circ_bounds_ll, densify_pts=21)
        vis_window = windows.from_bounds(*circ_bounds_vis, transform=vds.transform)
        vis_img = vds.read(window=vis_window, boundless=True, fill_value=0)
        left, bottom, right, top = rasterio.windows.bounds(vis_window, vds.transform)

        # Prepare RGB
        if vis_img.shape[0] >= 3:
            rgb = np.moveaxis(vis_img[:3], 0, -1).astype("float32")
            if rgb.max() > 1.0:
                rgb /= 255.0
        else:
            rgb = np.moveaxis(vis_img, 0, -1).astype("float32")
            if rgb.max() > 1.0:
                rgb /= 255.0

        # Circle coords in visual CRS
        circle_vis_crs = shp_transform(
            Transformer.from_crs("EPSG:4326", vds.crs, always_xy=True).transform,
            circle_wgs
        )
        cx, cy = circle_vis_crs.exterior.xy

        # Make/use axes
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))
            made_fig = True
        else:
            made_fig = False
            
        s2_dt = pd.Timestamp(item.datetime).tz_convert("UTC")
        ax.imshow(rgb, extent=[left, right, bottom, top], origin="upper")
        ax.plot(cx, cy, linewidth=2, color="red")
        ax.set_xlabel(f"CRS: {vds.crs}")
        ax.set_ylabel("")
        # ax.set_title(
        #     f"{getattr(item,'id','')} — cloud in 3.5 km circle: "
        #     f"{'NaN' if np.isnan(cloud_fraction) else f'{cloud_fraction:.2%}'}"
        # )
        ax.set_title(f"S2 @ {s2_dt.isoformat()}\n{s2_label}", fontsize=11)
        
        ax.text(0.5, 0.5,
                f"{'NaN' if np.isnan(cloud_fraction) else f'{cloud_fraction:.2%}'}",
                transform=ax.transAxes, va="center", ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.9),
                fontsize=11
            )

        if made_fig:
            plt.show()

    return cloud_fraction

######################################################################################
######################################################################################
def plot_matched_pairs_timeseries(
    cdf_dir,
    tsi_times,
    s2_items,
    thin_var="percent_thin",
    opaque_var="percent_opaque",
    fill_value=-100,
    window="30min",   # half-window on each side
    max_pairs=None,
):
    """
    For each (tsi_time, s2_item):
      - open that date’s TSI file
      - ensure time coord exists
      - read thin/opaque at nearest time, mask fill
      - plot a ±window time-series with a vertical line at the matched time
      - show S2 visual next to it and annotate cloud %
    """
    cdf_dir = Path(cdf_dir)
    pairs = list(zip(tsi_times, s2_items))
    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    half = pd.Timedelta(window)

    for i, (tsi_ts, item) in enumerate(pairs, 1):
        date_str = pd.Timestamp(tsi_ts).tz_convert("UTC").date().isoformat()
        tsi_fp = _date_to_tsi_file(cdf_dir, date_str)
        if tsi_fp is None:
            print(f"[{i}] No TSI file for {date_str}, skipping.")
            continue

        try:
            ds = xr.open_dataset(tsi_fp)
            ds = _ensure_time_coord(ds)
        except Exception as e:
            print(f"[{i}] Failed TSI open/prepare for {tsi_fp.name}: {e}")
            continue

        # get time to sample (tz-naive UTC for xarray .sel)
        tsel = pd.Timestamp(tsi_ts).tz_convert("UTC").tz_localize(None)

        # mask fill values
        if thin_var not in ds or opaque_var not in ds:
            print(f"[{i}] Missing {thin_var}/{opaque_var} in {tsi_fp.name}, skipping.")
            ds.close()
            continue
        thin = ds[thin_var].where(ds[thin_var] != fill_value)
        opaque = ds[opaque_var].where(ds[opaque_var] != fill_value)

        # extract nearest value at tsel
        try:
            thin_val = float(thin.sel(time=tsel, method="nearest").item())
            opaque_val = float(opaque.sel(time=tsel, method="nearest").item())
        except Exception as e:
            print(f"[{i}] Could not sample thin/opaque at {tsel} in {tsi_fp.name}: {e}")
            ds.close()
            continue

        # determine display units (fraction vs percent)
        # heuristic: if max over the day > 1.5, treat as percent
        day_mask = (ds["time"] >= tsel.floor("D")) & (ds["time"] < (tsel.floor("D") + pd.Timedelta("1D")))
        day_thin = thin.where(day_mask, drop=True).values
        day_opaque = opaque.where(day_mask, drop=True).values
        scale_is_percent = False
        for arr in (day_thin, day_opaque):
            a = arr[np.isfinite(arr)]
            if a.size and np.nanmax(a) > 1.5:
                scale_is_percent = True
        fmt = (lambda x: f"{x:.1f}%") if scale_is_percent else (lambda x: f"{x:.3f}")

        # time window around the selected time
        t0, t1 = tsel - half, tsel + half
        thin_w = thin.sel(time=slice(t0, t1))
        opaque_w = opaque.sel(time=slice(t0, t1))

        # load Sentinel-2 visual
        try:
            s2_img, s2_label = _load_s2_visual(item)
        except Exception as e:
            print(f"[{i}] Could not load Sentinel-2 visual for {getattr(item, 'id','<unknown>')}: {e}")
            ds.close()
            continue
        s2_dt = pd.Timestamp(item.datetime).tz_convert("UTC")

        # --- plotting ---
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

        # Left: time series panel
        ax0 = axes[0]
        if thin_w.size > 0:
            ax0.plot(pd.to_datetime(thin_w["time"].values), thin_w.values, label="percent_thin")
        if opaque_w.size > 0:
            ax0.plot(pd.to_datetime(opaque_w["time"].values), opaque_w.values, label="percent_opaque")
        ax0.axvline(tsel, color="k", linestyle="--", linewidth=1, label="matched time")

        ax0.set_title(f"TSI time-series @ {tsel.isoformat()}", fontsize=11)
        ax0.set_xlabel("UTC time")
        ax0.set_ylabel("Value" + (" (%)" if scale_is_percent else " (fraction)"))
        ax0.tick_params(axis="x", rotation=45)
        # ax0.set_xticklabels(ax0.get_xticklabels(), rotation=45)
        ax0.legend(loc="upper right")
        ax0.grid(True, alpha=0.3)

        # Big text box with exact values at the matched time
        ax0.text(
            0.02, 0.98,
            f"thin = {fmt(thin_val)}\nopaque = {fmt(opaque_val)}",
            transform=ax0.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.9),
            fontsize=11
        )

        # # Right: Sentinel-2 panel
        # ax1 = axes[1]
        # if s2_img.ndim == 3:
        #     ax1.imshow(s2_img)
        # else:
        #     ax1.imshow(s2_img, cmap="gray")
        # ax1.set_title(f"S2 {getattr(item,'id','')} @ {s2_dt.isoformat()}\n{s2_label}", fontsize=11)
        # ax1.axis("off")

        # s2_cloud = _s2_cloud_pct(item)
        # if s2_cloud is not None:
        #     ax1.text(
        #         0.02, 0.98,
        #         f"cloud: {s2_cloud:.1f}%",
        #         transform=ax1.transAxes, va="top", ha="left",
        #         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.9),
        #         fontsize=11
        #     )
            
        # Right: Sentinel-2 panel panel WITH circle overlay + cloud fraction
        ax1 = axes[1]
        cf = plot_s2_circle_cloud_fraction(
            item,
            aoi_bounds_ll=aoi_bounds_ll,   # this is already defined earlier from the shapefile
            s2_label = s2_label,
            radius_m=3500.0,
            include_shadow=True,          # set True to count SCL=3 as cloudy
            epsg_utm="EPSG:32613",
            ax=ax1,
        )
        
        # Optional annotation for the item-level cloud percentage too:
        s2_cloud = _s2_cloud_pct(item)
        if s2_cloud is not None:
            ax1.text(
                0.02, 0.98,
                f"cloud: {s2_cloud:.1f}%",
                transform=ax1.transAxes, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.9),
                fontsize=11
            )

        fig.suptitle(f"Matched pair {i} — {date_str}", fontsize=12)
        plt.show()
        ds.close()
   
