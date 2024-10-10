"""
Trim
Last Updated: Jan 23, 2024
This script supplies the basic functions for trimming
xarray timeseries and clipping geospatial functions.
@author: markjcampmier
"""
# Import Packages
import xarray as xr
import geopandas as gpd

from functools import reduce
from shapely.geometry import Polygon, MultiPolygon


# Define Functions
def fill_holes_in_geometries(row):
    """
    Fills holes in a geometry below a certain area threshold.

    :param row: A GeoDataFrame row with a 'geometry' column, which can contain Polygon or MultiPolygon geospatial data.
    :type row: geopandas.GeoSeries
    :returns geom: The original geometry with small holes filled, if any found.
    :rtype: shapely.geometry.Polygon, shapely.geometry.MultiPolygon
    """
    geom = row['geometry']

    def handle_polygon(polygon):
        """
        Helper function to fill holes in individual polygon.

        :param polygon: Polygon possibly with holes ("rings")
        :type polygon: shapely.geometry.Polygon
        :returns newgeom: The original geometry with small holes filled, if any found.
        :rtype: shapely.geometry.Polygon, shapely.geometry.MultiPolygon
        """
        newgeom = None
        rings = [i for i in polygon.interiors]
        if len(rings) > 0:
            to_fill = [Polygon(ring) for ring in rings if Polygon(ring).area < 1000]
            if len(to_fill) > 0:
                newgeom = reduce(lambda geom1, geom2: geom1.union(geom2), [polygon] + to_fill)
        return newgeom if newgeom else polygon

    if geom.geom_type == 'MultiPolygon':
        return MultiPolygon([handle_polygon(polygon) for polygon in geom.geoms])
    elif geom.geom_type == 'Polygon':
        return handle_polygon(geom)
    else:
        return geom


def trim(ds, start_time, end_time):
    """
    Trims a dataset by selecting time steps between start_time and end_time.

    :param ds: Dataset to be trimmed.
    :type ds: xarray.Dataset
    :param start_time: Start time of trimming.
    :type start_time: pandas.Timestamp
    :param end_time: Start time of trimming.
    :type end_time: pandas.Timestamp
    :returns ds: The trimmed dataset.
    :rtype: xarray.Dataset
    """

    ds = ds.sel(time=slice(start_time, end_time))
    return ds


def clip_raster(ds, gdf):
    """
    Clips a raster dataset to the geometry of a geopandas DataFrame.

    :param ds: The xarray dataset to process.
    :type ds: xarray.Dataset
    :param gdf: The geopandas DataFrame to clip to.
    :type gdf: geopandas.GeoDataFrame
    :returns ds: The clipped xarray dataset.
    :rtype: xarray.Dataset
    """

    ds = ds.rio.write_crs('epsg:4326')
    ds = ds.rio.clip(gdf.geometry, gdf.crs)
    return ds


def clip_points(ds, gdf, key='sensor'):
    """
    Clips a point dataset to the geometry of a geopandas DataFrame.

    :param ds: The xarray dataset to process.
    :type ds: xarray.Dataset
    :param gdf: The geopandas DataFrame to clip to.
    :type gdf: geopandas.GeoDataFrame
    :param key: The key to clip to, either 'sensor' or 'position.'
    :returns ds: The clipped point dataset.
    :rtype: xarray.Dataset
    """
    df_points = ds[['latitude', 'longitude']].to_dataframe().reset_index()
    if 'time' in df_points.columns:
        df_points = df_points.drop(['time'], axis=1)
    df_points = df_points.dropna().drop_duplicates()
    gdf_points = gpd.GeoDataFrame(df_points,
                                  geometry=gpd.points_from_xy(df_points.longitude,
                                                              df_points.latitude,
                                                              crs='epsg:4326'))
    gdf_points = gpd.clip(gdf_points, gdf)
    gdf_points = gdf_points.reset_index().set_index(key)

    ds = ds.sel({key: gdf_points.index.unique().sort_values()})
    return ds


def trim_clip(in_path, start_time=None, end_time=None, gdf=None):
    """
    Trims and clips a dataset based on time and geometry.

    :param in_path: path to xarray file
    :type in_path: str
    :param start_time: Start time of trimming.
    :type start_time: pandas.Timestamp
    :param end_time: End time of trimming.
    :type end_time: pandas.Timestamp
    :param gdf: The geopandas DataFrame to clip to.
    :type gdf: geopandas.GeoDataFrame
    :returns ds: The trimmed xarray dataset.
    :rtype: xarray.Dataset
    """

    ds = xr.open_dataset(in_path, decode_coords='all')

    if 'time' in ds.dims:
        ds = trim(ds, start_time, end_time)

    if 'sensor' in ds.dims:
        ds = clip_points(ds, gdf)
    elif 'site' in ds.dims:
        ds = clip_points(ds, gdf, key='site')
    elif 'position' in ds.dims:
        ds = clip_points(ds, gdf, key='position')
    else:
        ds = clip_raster(ds, gdf)

    return ds
