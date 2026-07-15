"""Time trimming and geospatial clipping utilities for xarray Datasets."""

import xarray as xr
import geopandas as gpd

from functools import reduce
from shapely.geometry import Polygon, MultiPolygon


def fill_holes_in_geometries(row: gpd.GeoSeries) -> Polygon | MultiPolygon:
    """Fill small holes in a Polygon or MultiPolygon geometry.

    Args:
        row: GeoDataFrame row with a 'geometry' column.

    Returns:
        Geometry with small interior rings filled.
    """
    geom = row["geometry"]

    def _fill_polygon(polygon: Polygon) -> Polygon:
        rings = list(polygon.interiors)
        if not rings:
            return polygon
        to_fill = [Polygon(ring) for ring in rings if Polygon(ring).area < 1000]
        if not to_fill:
            return polygon
        return reduce(lambda g1, g2: g1.union(g2), [polygon] + to_fill)

    if geom.geom_type == "MultiPolygon":
        return MultiPolygon([_fill_polygon(polygon) for polygon in geom.geoms])
    if geom.geom_type == "Polygon":
        return _fill_polygon(geom)
    return geom


def trim(ds: xr.Dataset, start_time, end_time) -> xr.Dataset:
    """Trim a dataset to the time window [start_time, end_time].

    Args:
        ds: Dataset with a 'time' dimension.
        start_time: Start of the time window.
        end_time: End of the time window.

    Returns:
        Trimmed dataset.
    """
    return ds.sel(time=slice(start_time, end_time))


def clip_raster(ds: xr.Dataset, gdf: gpd.GeoDataFrame) -> xr.Dataset:
    """Clip a raster dataset to the geometry of a GeoDataFrame.

    Args:
        ds: xarray Dataset with spatial dimensions.
        gdf: GeoDataFrame defining the clip geometry.

    Returns:
        Clipped dataset.
    """
    ds = ds.rio.write_crs("epsg:4326")
    return ds.rio.clip(gdf.geometry, gdf.crs)


def clip_points(
    ds: xr.Dataset, gdf: gpd.GeoDataFrame, key: str = "sensor"
) -> xr.Dataset:
    """Clip a point dataset to the geometry of a GeoDataFrame.

    Args:
        ds: xarray Dataset with 'latitude' and 'longitude' coordinates.
        gdf: GeoDataFrame defining the clip geometry.
        key: Dimension name used to select retained points.

    Returns:
        Dataset subset to points within the geometry.
    """
    df_points = ds[["latitude", "longitude"]].to_dataframe().reset_index()
    if "time" in df_points.columns:
        df_points = df_points.drop(["time"], axis=1)
    df_points = df_points.dropna().drop_duplicates()

    gdf_points = gpd.GeoDataFrame(
        df_points,
        geometry=gpd.points_from_xy(
            df_points.longitude, df_points.latitude, crs="epsg:4326"
        ),
    )
    gdf_points = gpd.clip(gdf_points, gdf).reset_index().set_index(key)
    return ds.sel({key: gdf_points.index.unique().sort_values()})


def trim_clip(
    in_path: str,
    start_time=None,
    end_time=None,
    gdf: gpd.GeoDataFrame | None = None,
) -> xr.Dataset:
    """Open a dataset and apply time trimming and/or geospatial clipping.

    Args:
        in_path: Path to the xarray NetCDF file.
        start_time: Start of the time window.
        end_time: End of the time window.
        gdf: GeoDataFrame defining the spatial clip geometry.

    Returns:
        Trimmed and/or clipped dataset.
    """
    ds = xr.open_dataset(in_path, decode_coords="all")

    if "time" in ds.dims:
        ds = trim(ds, start_time, end_time)

    if "sensor" in ds.dims:
        ds = clip_points(ds, gdf)
    elif "site" in ds.dims:
        ds = clip_points(ds, gdf, key="site")
    elif "position" in ds.dims:
        ds = clip_points(ds, gdf, key="position")
    else:
        ds = clip_raster(ds, gdf)

    return ds
