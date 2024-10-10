"""
Global Human Settlement Layer (GHSL) Population Data
Last Updated: June 15, 2024
This script formats data downloaded off the GHSL landing page.
@author: markjcampmier
"""
# Import Packages
import os
import glob
import geopandas as gpd
import xarray as xr


def format_ghsl(in_path):
    """
    Clips a GHSL raster dataset using a custom GeoDataFrame.

    :param in_path: The path to the input directory containing the files
    :type in_path: str

    :returns ds_clipped: Clipped xarray dataset
    :rtype: xarray.Dataset
    """

    # Read the raster data

    lst_tif = glob.glob(os.path.join(in_path, '*/*.tif'))
    ds_pop = xr.combine_by_coords([xr.open_dataset(tif) for tif in lst_tif])

    # Transform the raster to EPSG:4326
    ds_pop = ds_pop.rio.reproject('epsg:4326')

    # Read the custom GeoDataFrame for clipping
    gdf_clip = gpd.read_file(os.path.join(os.path.split(in_path)[0], 'shapefiles/India.gpkg'))

    # Clip the raster using the custom GeoDataFrame
    ds_clipped = ds_pop.rio.clip(gdf_clip.geometry.values, gdf_clip.crs, drop=True)
    ds_clipped = ds_clipped.squeeze().drop('band')
    ds_clipped = ds_clipped.drop('spatial_ref')

    ds_clipped = ds_clipped.rename({'x': 'longitude',
                                    'y': 'latitude'})
    ds_clipped = ds_clipped.rename({'band_data': 'population'})

    return ds_clipped
