import os
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr


def points_to_xarray(gdf, resolution=0.1, value_column="rwi"):
    bounds = gdf.total_bounds

    lons = np.arange(bounds[0], bounds[2] + resolution, resolution)
    lats = np.arange(bounds[1], bounds[3] + resolution, resolution)

    data = np.full((len(lats), len(lons)), np.nan)

    for idx, row in gdf.iterrows():
        lon_idx = int((row.geometry.x - bounds[0]) / resolution)
        lat_idx = int((row.geometry.y - bounds[1]) / resolution)

        if (0 <= lon_idx < len(lons)) and (0 <= lat_idx < len(lats)):
            data[lat_idx, lon_idx] = row[value_column]

    ds = xr.Dataset(
        data_vars={value_column: (("latitude", "longitude"), data)},
        coords={"latitude": lats, "longitude": lons},
        attrs={
            "description": f"Gridded {value_column} data",
            "resolution_degrees": resolution,
            "crs": "EPSG:4326",
        },
    )

    return ds


def format_rwi(in_path):

    df_rwi = pd.read_csv(os.path.join(in_path, "rwi.csv"))
    gdf_rwi = gpd.GeoDataFrame(
        df_rwi.loc[:, "rwi"],
        geometry=gpd.points_from_xy(df_rwi.longitude, df_rwi.latitude, crs="epsg:4326"),
    )
    xr_rwi = points_to_xarray(gdf_rwi)

    return xr_rwi
