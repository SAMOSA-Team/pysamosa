"""
GEE
Last Updated: Jan 28, 2025
This script formats and QA's GEE Satellite data
downloaded from Google Earth Engine.
@author: markjcampmier
"""
# Import Packages
import os
import glob
import re
import pandas as pd
import xarray as xr

from shapely.geometry import mapping
from rasterio.enums import Resampling


def index_gee(ds, bounds=None):
    """
    Indexes GEE data by date and time, and rounds the coordinates to 3 and 2 decimal places, respectively.

    :param ds: A xarray.Dataset containing GEE data.
    :type ds: xarray.Dataset
    :param bounds: A dictionary containing the bounding box coordinates (minx, miny, maxx, maxy).
    :type bounds: dict
    :returns ds: An xarray.Dataset with indexed GEE data.
    :rtype: xarray.Dataset
    """

    if bounds is None:
        bounds = {"minx": 73.29, "miny": 19.77, "maxx": 91.05, "maxy": 33.29}

    ds = ds.squeeze(dim="band").reset_coords(["band"], drop=True)

    ds = ds.assign_coords({"x": ds.x.round(3), "y": ds.y.round(3)})

    ds = ds.rio.reproject(ds.rio.crs, resolution=0.01, resampling=Resampling.bilinear)

    ds = ds.rio.clip_box(
        minx=bounds["minx"],
        miny=bounds["miny"],
        maxx=bounds["maxx"],
        maxy=bounds["maxy"],
    )

    ds = ds.assign_coords({"x": ds.x.round(2), "y": ds.y.round(2)})

    ds = ds.band_data.pad()

    # Extract the date and time from the filename.
    str_regex = re.compile(r"\d{4}-\d{2}-\d{2}")
    str_match = str_regex.search(ds.encoding["source"])
    str_extract = str_match.group()

    # Create a Pandas datetime index.
    idx_time = pd.DatetimeIndex([pd.to_datetime(str_extract, format="%Y-%m-%d")])

    # Expand the dataset with the time dimension.
    ds = ds.expand_dims({"time": idx_time})

    return ds


def clip_gee(ds_gee, gdf_shape, all_touched=False):
    """
    Clips TROPOMI data to a given shapefile.

    :param ds_gee: A xarray.Dataset containing GEE data.
    :type ds_gee: xarray.Dataset
    :param gdf_shape: A GeoPandas GeoDataFrame representing the shapefile.
    :type gdf_shape: GeoPandas GeoDataFrame
    :param all_touched: A boolean indicating whether to include all pixels that touch the shapefile.
    :type all_touched: bool
    :returns ds_gee: An xarray.Dataset containing GEE data.
    :rtype: xarray.Dataset
    """

    ds_gee = ds_gee.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
    ds_gee_clipped = ds_gee.rio.clip(
        gdf_shape.geometry.apply(mapping), ds_gee.rio.crs, all_touched=all_touched
    )
    return ds_gee_clipped


def format_gee(in_path):
    """
    Formats GEE data by indexing and clipping.
    :param in_path: The path to the directory containing the GEE data files.
    :type in_path: str
    :returns ds_gee: An xarray.Dataset containing GEE data.
    :rtype: xarray.Dataset
    """
    ds_list = []
    file_directory = glob.glob(os.path.join(in_path, "gee_*"))

    for directory in file_directory:
        ds_ = xr.open_mfdataset(
            glob.glob(os.path.join(directory, "*.tif")),
            preprocess=index_gee,
            combine="by_coords",
            parallel=True,
        )

        ds_ = ds_.rename(
            {"x": "longitude", "y": "latitude", "band_data": directory.split("_")[-1]}
        )

        ds_ = ds_.sortby("time")
        ds_list.append(ds_)

    ds_combined = xr.combine_by_coords(ds_list)
    return ds_combined
