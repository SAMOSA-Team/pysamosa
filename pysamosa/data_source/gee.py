"""Google Earth Engine satellite data formatter."""

import os
import glob
import re
import pandas as pd
import xarray as xr

from shapely.geometry import mapping
from rasterio.enums import Resampling


def _index_gee(ds: xr.Dataset, bounds: dict | None = None) -> xr.Dataset:
    """Reproject, clip, and add a time dimension to a single GEE tile.

    Args:
        ds: xarray Dataset for one GEE tile.
        bounds: Bounding box dict with keys minx, miny, maxx, maxy.

    Returns:
        Indexed GEE dataset with a time dimension.
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

    str_regex = re.compile(r"\d{4}-\d{2}-\d{2}")
    str_match = str_regex.search(ds.encoding["source"])
    idx_time = pd.DatetimeIndex([pd.to_datetime(str_match.group(), format="%Y-%m-%d")])
    ds = ds.expand_dims({"time": idx_time})

    return ds


def clip_gee(ds_gee: xr.Dataset, gdf_shape, all_touched: bool = False) -> xr.Dataset:
    """Clip a GEE dataset to a shapefile geometry.

    Args:
        ds_gee: GEE dataset.
        gdf_shape: GeoDataFrame whose geometry defines the clip boundary.
        all_touched: Include all pixels touching the shapefile boundary.

    Returns:
        Clipped GEE dataset.
    """
    ds_gee = ds_gee.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
    return ds_gee.rio.clip(
        gdf_shape.geometry.apply(mapping), ds_gee.rio.crs, all_touched=all_touched
    )


def format_gee(in_path: str) -> xr.Dataset:
    """Format all GEE products from a directory into a single xarray Dataset.

    Args:
        in_path: Path to the directory containing gee_* subdirectories.

    Returns:
        Combined GEE dataset.
    """
    ds_list = []
    for directory in glob.glob(os.path.join(in_path, "gee_*")):
        ds_ = xr.open_mfdataset(
            glob.glob(os.path.join(directory, "*.tif")),
            preprocess=_index_gee,
            combine="by_coords",
            parallel=True,
        )
        ds_ = ds_.rename(
            {"x": "longitude", "y": "latitude", "band_data": directory.split("_")[-1]}
        )
        ds_ = ds_.sortby("time")
        ds_list.append(ds_)

    return xr.combine_by_coords(ds_list)
