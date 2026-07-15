"""Central Pollution Control Board (CPCB) regulatory site data formatter."""

import os
import glob
import threading
import pandas as pd
import geopandas as gpd
import xarray as xr


def _fetch_reg(in_path: str, pollutant: str) -> list[str]:
    """Return paths to all feather files for a given pollutant.

    Args:
        in_path: Path to the CPCB data directory.
        pollutant: Pollutant code (e.g. 'PM25', 'NO2').

    Returns:
        List of matching feather file paths.
    """
    return glob.glob(os.path.join(in_path, f"*_{pollutant}_reg.feather"))


def _fetch_meta(in_path: str) -> gpd.GeoDataFrame:
    """Read CPCB site metadata from a GeoJSON file.

    Args:
        in_path: Path to the directory containing meta_reg.geojson.

    Returns:
        GeoDataFrame of CPCB site metadata indexed by location_name.
    """
    gdf_meta = gpd.read_file(os.path.join(in_path, "meta_reg.geojson"))
    gdf_meta = gdf_meta.set_index("location_name")
    gdf_meta.loc[:, "longitude"] = gdf_meta.geometry.x
    gdf_meta.loc[:, "latitude"] = gdf_meta.geometry.y
    return gdf_meta


def _dataset_reg(
    in_path: str, pollutant: str, gdf_meta: gpd.GeoDataFrame, ds_list: list
) -> None:
    """Build an xarray Dataset for one pollutant and append to ds_list (thread target).

    Args:
        in_path: Path to the CPCB data directory.
        pollutant: Pollutant code (e.g. 'PM25', 'NO2').
        gdf_meta: Metadata GeoDataFrame indexed by location_name.
        ds_list: Shared list accumulating per-pollutant datasets.
    """
    lst_path = _fetch_reg(in_path, pollutant)
    df = pd.concat(
        [
            pd.read_feather(
                file,
                columns=[
                    "observation_hour_start",
                    "location_name",
                    "average_value_in_hour",
                ],
            )
            for file in lst_path
        ]
    )
    df = df.pivot_table(
        index="observation_hour_start",
        columns="location_name",
        values="average_value_in_hour",
    )
    valid_locations = df.columns.intersection(gdf_meta.index)
    df = df.loc[:, valid_locations]
    df = df.resample("1h").mean()
    df.index = df.index - pd.Timedelta("1h")

    ds = xr.Dataset(
        {pollutant: (("time", "site"), df.values.astype(float))},
        coords={
            "time": ("time", df.index),
            "site": ("site", df.columns.astype(str)),
            "state": (
                "site",
                gdf_meta.loc[df.columns.astype(str), "state_name"].values,
            ),
            "district": (
                "site",
                gdf_meta.loc[df.columns.astype(str), "district_name"].values,
            ),
            "settlement_name": (
                "site",
                gdf_meta.loc[df.columns.astype(str), "city_name"].values,
            ),
        },
    )
    ds_list.append(ds)


def format_reg(in_path: str) -> xr.Dataset:
    """Format and combine CPCB pollutant data into a single merged xarray Dataset.

    Args:
        in_path: Path to the directory containing CPCB feather files and metadata.

    Returns:
        Merged CPCB dataset with all pollutants and spatial coordinates.
    """
    gdf_meta = _fetch_meta(in_path)
    ds_list: list = []
    threads = []

    for pollutant in ["CO", "NO2", "SO2", "O3", "PM25", "PM10"]:
        thread = threading.Thread(
            target=_dataset_reg, args=(in_path, pollutant, gdf_meta, ds_list)
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    ds_cpcb = xr.merge(ds_list)
    ds_cpcb = xr.merge(
        [
            gdf_meta.loc[ds_cpcb.site.values, "latitude"]
            .to_xarray()
            .rename({"location_name": "site"}),
            gdf_meta.loc[ds_cpcb.site.values, "longitude"]
            .to_xarray()
            .rename({"location_name": "site"}),
            ds_cpcb,
        ]
    )

    ds_cpcb = ds_cpcb.rename_vars(
        {
            "PM10": "pm10",
            "CO": "co",
            "SO2": "so2",
            "O3": "o3",
            "PM25": "pm25",
            "NO2": "no2",
        }
    )
    ds_cpcb = ds_cpcb.sortby("time").transpose("site", "time")
    ds_cpcb = ds_cpcb.set_coords(["latitude", "longitude"])

    ds_cpcb["pm25"] = ds_cpcb["pm25"].assign_attrs(
        long_name="PM2.5",
        units="ug/m^3",
        description="particulate matter less than 2.5 microns",
    )
    ds_cpcb["pm10"] = ds_cpcb["pm10"].assign_attrs(
        long_name="PM10",
        units="ug/m^3",
        description="particulate matter less than 10 microns",
    )
    ds_cpcb["no2"] = ds_cpcb["no2"].assign_attrs(
        long_name="nitrogen dioxide",
        units="ug/m^3",
        description="nitrogen dioxide concentration",
    )
    ds_cpcb["so2"] = ds_cpcb["so2"].assign_attrs(
        long_name="sulfur dioxide",
        units="ug/m^3",
        description="sulfur dioxide concentration",
    )
    ds_cpcb["co"] = ds_cpcb["co"].assign_attrs(
        long_name="carbon monoxide",
        units="mg/m^3",
        description="carbon monoxide concentration",
    )
    ds_cpcb["o3"] = ds_cpcb["o3"].assign_attrs(
        long_name="ozone", units="ug/m^3", description="ozone concentration"
    )
    ds_cpcb["settlement_name"] = ds_cpcb["settlement_name"].assign_attrs(
        long_name="settlement name", description="settlement name from project metadata"
    )
    ds_cpcb["district"] = ds_cpcb["district"].assign_attrs(
        long_name="district name", description="district name from shapefile"
    )
    ds_cpcb["state"] = ds_cpcb["state"].assign_attrs(
        long_name="state name", description="state name from shapefile"
    )
    ds_cpcb["latitude"] = ds_cpcb["latitude"].assign_attrs(
        long_name="latitude", units="degrees_north"
    )
    ds_cpcb["longitude"] = ds_cpcb["longitude"].assign_attrs(
        long_name="longitude", units="degrees_east"
    )
    ds_cpcb.attrs["history"] = "munged"

    return ds_cpcb
