"""WUSTL Randall Martin AOD-PM inversion dataset formatter."""

import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
from rasterio.enums import Resampling


def _index_martin(ds: xr.Dataset, bounds: dict | None = None) -> xr.Dataset:
    """Reproject, clip, and add a time dimension to a single Martin dataset tile.

    Args:
        ds: xarray Dataset for one Martin AOD-PM file.
        bounds: Bounding box dict with keys minx, miny, maxx, maxy.

    Returns:
        Indexed Martin dataset with a time dimension.
    """
    if bounds is None:
        bounds = {"minx": 64.28, "miny": 1.30, "maxx": 97.24, "maxy": 37.63}

    ds = ds.set_coords(["lat", "lon"])
    ds = ds.rename({"lon": "x", "lat": "y"})
    ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y")
    ds = ds.rio.write_crs("EPSG:4326")

    # Clip before reprojection to reduce memory
    ds = ds.rio.clip_box(
        minx=bounds["minx"] - 0.5,
        miny=bounds["miny"] - 0.5,
        maxx=bounds["maxx"] + 0.5,
        maxy=bounds["maxy"] + 0.5,
    )
    ds = ds.rio.reproject(ds.rio.crs, resolution=0.01, resampling=Resampling.bilinear)
    ds = ds.rio.clip_box(
        minx=bounds["minx"],
        miny=bounds["miny"],
        maxx=bounds["maxx"],
        maxy=bounds["maxy"],
    )
    ds = ds.assign_coords({"x": ds.x.round(2), "y": ds.y.round(2)})
    ds = ds.rename_vars({"PM25": "pm25_cnn"})

    time = pd.Timestamp(
        year=int(ds.attrs["TIMECOVERAGE"][:4]),
        month=int(ds.attrs["TIMECOVERAGE"][4:6]),
        day=1,
        hour=1,
        minute=0,
        second=0,
    )
    ds = ds.expand_dims({"time": pd.DatetimeIndex([time])})

    del ds["pm25_cnn"].attrs["_FillValue"]
    ds.pm25_cnn.encoding["_FillValue"] = np.nan

    return ds


def format_martin(in_path: str, batch_size: int = 5) -> xr.Dataset:
    """Format Martin AOD-PM files in batches to manage memory.

    Args:
        in_path: Path to the directory containing Martin NetCDF files.
        batch_size: Number of files to process per batch.

    Returns:
        Concatenated Martin dataset.
    """
    files = sorted(glob.glob(os.path.join(in_path, "*.nc")))
    datasets = []

    for i in range(0, len(files), batch_size):
        batch = files[i : i + batch_size]
        ds_batch = xr.open_mfdataset(
            batch,
            preprocess=_index_martin,
            combine="by_coords",
            parallel=False,
            chunks={"x": 300, "y": 300},
        )
        datasets.append(ds_batch.load())

    ds_martin = xr.concat(datasets, dim="time")
    ds_martin = ds_martin.rename({"x": "longitude", "y": "latitude"})
    ds_martin = ds_martin.sortby("time")

    return ds_martin
