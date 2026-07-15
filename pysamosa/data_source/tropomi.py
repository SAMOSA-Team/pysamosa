"""TROPOMI NO2 column density formatter."""

import os
import glob
import re
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray as rxr  # noqa: F401  # pylint: disable=unused-import  # registers the rio accessor
from shapely.geometry import mapping
from rasterio.enums import Resampling


def _index_tropomi(ds: xr.Dataset, bounds: dict | None = None) -> xr.Dataset:
    """Reproject, clip, and add a time dimension to a single TROPOMI tile.

    Args:
        ds: xarray Dataset for one TROPOMI file.
        bounds: Bounding box dict with keys minx, miny, maxx, maxy.

    Returns:
        Indexed TROPOMI dataset with a time dimension.
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

    str_regex = re.compile(r"\d{2}[A-Za-z]{3}\d{4}_\d{2}")
    str_match = str_regex.search(ds.encoding["source"])
    idx_time = pd.DatetimeIndex([pd.to_datetime(str_match.group(), format="%d%b%Y_%H")])
    ds = ds.expand_dims({"time": idx_time})

    return ds


def clip_tropomi(
    ds_tropomi: xr.Dataset, gdf_shape: gpd.GeoDataFrame, all_touched: bool = False
) -> xr.Dataset:
    """Clip a TROPOMI dataset to a shapefile geometry.

    Args:
        ds_tropomi: TROPOMI dataset.
        gdf_shape: GeoDataFrame whose geometry defines the clip boundary.
        all_touched: Include all pixels touching the shapefile boundary.

    Returns:
        Clipped TROPOMI dataset.
    """
    ds_tropomi = ds_tropomi.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
    return ds_tropomi.rio.clip(
        gdf_shape.geometry.apply(mapping), ds_tropomi.rio.crs, all_touched=all_touched
    )


def format_tropomi(in_path: str) -> xr.Dataset:
    """Format TROPOMI TIF files from a directory into a single xarray Dataset.

    Args:
        in_path: Path to the directory containing TROPOMI TIF files.

    Returns:
        Combined TROPOMI NO2 column dataset.
    """
    ds_tropomi = xr.open_mfdataset(
        glob.glob(os.path.join(in_path, "*.tif")),
        preprocess=_index_tropomi,
        combine="by_coords",
        parallel=True,
    )
    ds_tropomi = ds_tropomi.rename(
        {"x": "longitude", "y": "latitude", "band_data": "no2_column"}
    )
    ds_tropomi = ds_tropomi.sortby("time")
    return ds_tropomi
