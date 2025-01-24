"""
PurpleAir
Last Updated: December 5, 2024
This script formats the PurpleAir data downloaded
from the PurpleAir platform.
@author: markjcampmier
"""
# Import Packages
import os
import glob
import numpy as np
import pandas as pd
import xarray as xr


# Define Functions
def read_pr(file):
    df_pa = pd.read_feather(file)

    df_pa.columns = ["timestamp", "rh", "temperature", "bscat_a", "bscat_b", "a", "b"]

    df_pa.loc[:, "time"] = pd.to_datetime(df_pa.timestamp, unit="s", utc=True)

    df_pa = df_pa.drop("timestamp", axis=1)

    df_pa = df_pa.set_index("time").sort_index()

    df_pa.index = df_pa.index.tz_convert("Asia/Kolkata")

    df_pa = df_pa.resample("2min").mean().round(2)

    df_pa.loc[:, "purpleair_sensor_index"] = int(
        file.split("/")[-1].split("_")[-1].replace(".feather", "")
    )
    return df_pa


def index_pr(in_path):

    lst_folders = [
        f for f in os.listdir(in_path) if os.path.isdir(os.path.join(in_path, f))
    ]

    lst_files = [
        glob.glob(os.path.join(in_path, f, "sensor_*.feather")) for f in lst_folders
    ]
    lst_files = [item for sublist in lst_files for item in sublist]

    lst_files = lst_files

    lst_dict_pa = [read_pr(i) for i in lst_files]

    df = pd.concat(lst_dict_pa)

    df_a = df.pivot_table(
        index="time", columns="purpleair_sensor_index", values="a", aggfunc="mean"
    )
    df_b = df.pivot_table(
        index="time", columns="purpleair_sensor_index", values="b", aggfunc="mean"
    )
    df_rh = df.pivot_table(
        index="time", columns="purpleair_sensor_index", values="rh", aggfunc="mean"
    )

    df_a.index = df_a.index.tz_localize(None)
    df_b.index = df_b.index.tz_localize(None)
    df_rh.index = df_rh.index.tz_localize(None)

    dict_pa = {
        "a": df_a,
        "b": df_b,
        "rh": df_rh,
    }

    return dict_pa


def index_meta(in_path):
    """Index metadata preserving site changes"""
    # Define metadata columns we want to keep
    meta_cols = [
        "purpleair_sensor_index",
        "samosa_identifier",
        "site_id",
        "effective_date",
        "discontinue_date",
        "site_latitude",
        "site_longitude",
        "settlement_type",
        "cluster_name",
        "settlement_name",
        "site_district_name",
        "site_state_name",
        "all_land_uses",
        "is_collocation_site",
    ]

    # Read metadata
    df_meta = pd.read_feather(
        os.path.join(in_path, "history.feather"), columns=meta_cols
    )

    # Create unique position for each site-sensor combination and handle discontinue_date
    df_meta["site_sensor"] = (
        df_meta["site_id"].astype(str) + "&" + df_meta["samosa_identifier"].astype(str)
    )
    df_meta["position"] = pd.factorize(df_meta["site_sensor"])[0]
    df_meta["discontinue_date"] = df_meta["discontinue_date"].fillna(
        pd.Timestamp("2025-01-01")
    )

    df_meta.loc[df_meta.settlement_type == "Large City", "settlement_type"] = 2
    df_meta.loc[df_meta.settlement_type == "Small City", "settlement_type"] = 1
    df_meta.loc[
        (df_meta.settlement_type != 2) & (df_meta.settlement_type != 1),
        "settlement_type",
    ] = 0
    df_meta["settlement_type"] = df_meta["settlement_type"].astype(int)

    return df_meta


def format_pr(in_path):
    """Format PurpleAir data with vectorized operations"""
    df_meta = index_meta(in_path)
    dict_pa = index_pr(in_path)

    # Create sensor to position mapping
    sensor_to_pos = {}
    for _, row in df_meta.iterrows():
        if row.purpleair_sensor_index not in sensor_to_pos:
            sensor_to_pos[row.purpleair_sensor_index] = []
        sensor_to_pos[row.purpleair_sensor_index].append(
            {
                "position": row.position,
                "start": row.effective_date,
                "end": row.discontinue_date,
            }
        )

    # Process measurements
    datasets = []
    for var_name in ["a", "b", "rh"]:
        df = dict_pa[var_name]
        positions = sorted(df_meta["position"].unique())
        values = np.full((len(df), len(positions)), np.nan)

        for sensor_idx in df.columns:
            if sensor_idx in sensor_to_pos:
                for period in sensor_to_pos[sensor_idx]:
                    mask = (df.index >= period["start"]) & (df.index <= period["end"])
                    pos_idx = positions.index(period["position"])
                    values[mask, pos_idx] = df.loc[mask, sensor_idx]

        datasets.append(
            xr.Dataset(
                data_vars={var_name: (("time", "position"), values)},
                coords={"time": df.index, "position": positions},
            )
        )

    # Merge measurements and filter to valid positions
    ds = xr.merge(datasets)
    has_data = ~np.isnan(ds.a.values).all(axis=0)
    valid_positions = ds.position.values[has_data]
    ds = ds.sel(position=valid_positions)

    # Define column mappings for coordinates
    coord_mappings = {
        "site_id": "site",
        "site_latitude": "latitude",
        "site_longitude": "longitude",
        "settlement_type": "settlement_type",
        "cluster_name": "cluster",
        "site_district_name": "district",
        "site_state_name": "state",
        "settlement_name": "settlement",
        "all_land_uses": "land_uses",
        "is_collocation_site": "is_collocation_site",
        "samosa_identifier": "sensor",
    }

    # Process metadata and add as coordinates
    meta_coords = (
        df_meta[["position"] + list(coord_mappings.keys())]
        .drop_duplicates("position")
        .set_index("position")
        .loc[valid_positions]
        .rename(columns=coord_mappings)
    )

    # Add coordinates to dataset
    for new_name in coord_mappings.values():
        ds = ds.assign_coords({new_name: ("position", meta_coords[new_name].values)})

    # Add attributes
    ds["a"] = ds["a"].assign_attrs(long_name="PurpleAir Sensor A", units="ug/m^3")
    ds["b"] = ds["b"].assign_attrs(long_name="PurpleAir Sensor B", units="ug/m^3")
    ds["rh"] = ds["rh"].assign_attrs(long_name="Relative Humidity", units="%")
    ds["latitude"].attrs["units"] = "degrees_north"
    ds["longitude"].attrs["units"] = "degrees_east"

    ds["disagreement"] = np.abs((2 * (ds["a"] - ds["b"])) / (ds["a"] + ds["b"])) * 100

    ds["disagreement"] = ds["disagreement"].assign_attrs(
        long_name="Channel Disagreement",
        units="%",
        description="PurpleAir Node Internal Disagreement",
    )

    return ds
