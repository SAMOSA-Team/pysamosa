"""
TROPOMI
Last Updated: Jan 12, 2024
This script formats and QA's TROPOMI NO2 column density
downloaded from Google Earth Engine.
@author: markjcampmier
"""
# Import Packages
import os
import glob
import re
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray as rxr
from shapely.geometry import mapping
from rasterio.enums import Resampling


def index_tropomi(ds, bounds=None):
    """
    Indexes TROPOMI data by date and time, and rounds the coordinates to 3 and 2 decimal places, respectively.

    :param ds: A xarray.Dataset containing TROPOMI data.
    :type ds: xarray.Dataset
    :param bounds: A dictionary containing the bounding box coordinates (minx, miny, maxx, maxy).
    :type bounds: dict
    :returns ds: An xarray.Dataset with indexed TROPOMI data.
    :rtype: xarray.Dataset
    """

    if bounds is None:
        bounds = {'minx': 73.29, 'miny': 19.77, 'maxx': 91.05, 'maxy': 33.29}

    ds = ds.squeeze(dim='band').reset_coords(['band'], drop=True)

    ds = ds.assign_coords({'x': ds.x.round(3),
                           'y': ds.y.round(3)})

    ds = ds.rio.reproject(ds.rio.crs, resolution=0.01,
                          resampling=Resampling.bilinear)

    ds = ds.rio.clip_box(minx=bounds['minx'], miny=bounds['miny'],
                         maxx=bounds['maxx'], maxy=bounds['maxy'])

    ds = ds.assign_coords({'x': ds.x.round(2),
                           'y': ds.y.round(2)})

    ds = ds.band_data.pad()

    # Extract the date and time from the filename.
    str_regex = re.compile(r'\d{2}[A-Za-z]{3}\d{4}_\d{2}')
    str_match = str_regex.search(ds.encoding["source"])
    str_extract = str_match.group()

    # Create a Pandas datetime index.
    idx_time = pd.DatetimeIndex([pd.to_datetime(str_extract,
                                                format='%d%b%Y_%H')])

    # Expand the dataset with the time dimension.
    ds = ds.expand_dims({'time': idx_time})

    return ds


def clip_tropmi(ds_tropomi, gdf_shape, all_touched=False):
    """
    Clips TROPOMI data to a given shapefile.

    :param ds_tropomi: A xarray.Dataset containing TROPOMI data.
    :type ds_tropomi: xarray.Dataset
    :param gdf_shape: A GeoPandas GeoDataFrame representing the shapefile.
    :type gdf_shape: GeoPandas GeoDataFrame
    :param all_touched: A boolean indicating whether to include all pixels that touch the shapefile.
    :type all_touched: bool
    :returns ds_tropomi: An xarray.Dataset containing TROPOMI data.
    :rtype: xarray.Dataset
    """

    ds_tropomi = ds_tropomi.rio.set_spatial_dims(x_dim='longitude',
                                                 y_dim='latitude')
    ds_tropomi_clipped = ds_tropomi.rio.clip(gdf_shape.geometry.apply(mapping),
                                             ds_tropomi.rio.crs, all_touched=all_touched)
    return ds_tropomi_clipped


def format_tropomi(in_path):
    """
    Formats TROPOMI data by indexing and clipping.
    :param in_path: The path to the directory containing the TROPOMI data files.
    :type in_path: str
    :returns ds_tropomi: An xarray.Dataset containing TROPOMI data.
    :rtype: xarray.Dataset
    """
    ds_tropomi = xr.open_mfdataset(glob.glob(os.path.join(in_path, '*.tif')),
                                   preprocess=index_tropomi,
                                   combine='by_coords',
                                   parallel=True)
    ds_tropomi = ds_tropomi.rename({'x': 'longitude',
                                    'y': 'latitude',
                                    'band_data': 'no2_column'})

    ds_tropomi = ds_tropomi.sortby('time')
    return ds_tropomi
