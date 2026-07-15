"""Global Human Settlement Layer (GHSL) population density formatter."""

import os
import glob
import geopandas as gpd
import xarray as xr


def format_ghsl(in_path: str) -> xr.Dataset:
    """Reproject and clip a GHSL raster dataset to the India shapefile.

    Args:
        in_path: Path to the input directory containing GHSL TIF files.

    Returns:
        Clipped population density dataset.
    """
    lst_tif = glob.glob(os.path.join(in_path, "*/*.tif"))
    ds_pop = xr.combine_by_coords([xr.open_dataset(tif) for tif in lst_tif])
    ds_pop = ds_pop.rio.reproject("epsg:4326")

    gdf_clip = gpd.read_file(
        os.path.join(os.path.split(in_path)[0], "shapefiles/India.gpkg")
    )
    ds_clipped = ds_pop.rio.clip(gdf_clip.geometry.values, gdf_clip.crs, drop=True)
    ds_clipped = ds_clipped.squeeze().drop("band")
    ds_clipped = ds_clipped.drop("spatial_ref")
    ds_clipped = ds_clipped.rename(
        {"x": "longitude", "y": "latitude", "band_data": "population"}
    )

    return ds_clipped
