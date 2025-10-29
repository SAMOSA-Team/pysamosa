""""
Martin
Last Updated: Oct 13, 2025
This script formats and QA's the WUSTL Randall Martin
AOD-PM inversion dataset.
@author: markjcampmier
"""
# Import Packages
import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
from rasterio.enums import Resampling


def index_martin(ds, bounds=None):
    """
    Indexes Randall Martin AOD-PM inversion data by date and time, and rounds the coordinates to 3 and 2 decimal
    places, respectively.

    :param ds: A xarray.Dataset containing Martin data.
    :type ds: xarray.Dataset
    :param bounds: Bounding box coordinates (minx, miny, maxx, maxy).
    :type bounds: dict
    :returns ds: An xarray.Dataset with indexed Martin data.
    :rtype: xarray.Dataset
    """

    if bounds is None:
        bounds = {"minx": 64.28, "miny": 1.30, "maxx": 97.24, "maxy": 37.63}

    ds = ds.set_coords(["lat", "lon"])
    ds = ds.rename({"lon": "x", "lat": "y"})

    ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y")
    ds = ds.rio.write_crs("EPSG:4326")

    # Clip BEFORE reprojection to reduce memory
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

    # Extract the date and time from the filename.
    time = pd.Timestamp(
        year=int(ds.attrs["TIMECOVERAGE"][:4]),
        month=int(ds.attrs["TIMECOVERAGE"][4:6]),
        day=1,
        hour=1,
        minute=0,
        second=0,
    )

    # Create a Pandas datetime index.
    idx_time = pd.DatetimeIndex([time])

    # Expand the dataset with the time dimension.
    ds = ds.expand_dims({"time": idx_time})

    del ds["pm25_cnn"].attrs["_FillValue"]
    ds.pm25_cnn.encoding["_FillValue"] = np.nan

    return ds


def format_martin(in_path, batch_size=5):
    """
    Formats Martin data by indexing and clipping in batches to manage memory.

    :param in_path: The path to the directory containing the Martin data files.
    :type in_path: str
    :param batch_size: Number of files to process at once (default 5 for 8GB RAM).
    :type batch_size: int
    :returns ds_martin: A xarray.Dataset containing Martin data.
    :rtype ds_martin: xarray.Dataset
    """
    files = sorted(glob.glob(os.path.join(in_path, "*.nc")))
    datasets = []

    for i in range(0, len(files), batch_size):
        batch = files[i : i + batch_size]
        ds_batch = xr.open_mfdataset(
            batch,
            preprocess=index_martin,
            combine="by_coords",
            parallel=False,
            chunks={"x": 300, "y": 300},
        )
        datasets.append(ds_batch.load())

    ds_martin = xr.concat(datasets, dim="time")
    ds_martin = ds_martin.rename({"x": "longitude", "y": "latitude"})
    ds_martin = ds_martin.sortby("time")

    return ds_martin
