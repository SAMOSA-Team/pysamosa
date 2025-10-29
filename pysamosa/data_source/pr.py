"""
PurpleAir
Last Updated: October 29, 2025
This script formats the PurpleAir data downloaded
from the PurpleAir platform.
@author: markjcampmier
"""
# Import Packages
import os
import glob
import gc
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

    # Use dictionaries to accumulate data by sensor
    sensor_data_a = {}
    sensor_data_b = {}
    sensor_data_rh = {}

    # Process files one at a time
    for i, file in enumerate(lst_files):
        if i % 10 == 0:
            print(f"Processing file {i + 1}/{len(lst_files)}")

        # Read single file
        df_single = read_pr(file)
        sensor_idx = df_single["purpleair_sensor_index"].iloc[0]

        # Accumulate data for this sensor
        if sensor_idx not in sensor_data_a:
            sensor_data_a[sensor_idx] = []
            sensor_data_b[sensor_idx] = []
            sensor_data_rh[sensor_idx] = []

        sensor_data_a[sensor_idx].append(df_single["a"])
        sensor_data_b[sensor_idx].append(df_single["b"])
        sensor_data_rh[sensor_idx].append(df_single["rh"])

        # Clear temporary variable
        del df_single

    # Now create the final DataFrames by concatenating time chunks for each sensor
    df_a = pd.DataFrame()
    df_b = pd.DataFrame()
    df_rh = pd.DataFrame()

    for sensor_idx in sensor_data_a.keys():
        # Concatenate all time chunks for this sensor
        sensor_a_combined = pd.concat(sensor_data_a[sensor_idx], axis=0).sort_index()
        sensor_b_combined = pd.concat(sensor_data_b[sensor_idx], axis=0).sort_index()
        sensor_rh_combined = pd.concat(sensor_data_rh[sensor_idx], axis=0).sort_index()

        # Remove any duplicate timestamps (in case of overlaps)
        sensor_a_combined = sensor_a_combined[
            ~sensor_a_combined.index.duplicated(keep="first")
        ]
        sensor_b_combined = sensor_b_combined[
            ~sensor_b_combined.index.duplicated(keep="first")
        ]
        sensor_rh_combined = sensor_rh_combined[
            ~sensor_rh_combined.index.duplicated(keep="first")
        ]

        # Convert to DataFrame with sensor_idx as column name
        temp_a = pd.DataFrame({sensor_idx: sensor_a_combined})
        temp_b = pd.DataFrame({sensor_idx: sensor_b_combined})
        temp_rh = pd.DataFrame({sensor_idx: sensor_rh_combined})

        # Now join (this works because each sensor is a different column)
        if df_a.empty:
            df_a = temp_a
            df_b = temp_b
            df_rh = temp_rh
        else:
            df_a = df_a.join(temp_a, how="outer")
            df_b = df_b.join(temp_b, how="outer")
            df_rh = df_rh.join(temp_rh, how="outer")

    # Clear the dictionaries
    del sensor_data_a, sensor_data_b, sensor_data_rh

    print("Finalizing time index...")

    # Ensure index name is 'time' and remove timezone
    df_a.index.name = "time"
    df_b.index.name = "time"
    df_rh.index.name = "time"

    df_a.index = df_a.index.tz_localize(None)
    df_b.index = df_b.index.tz_localize(None)
    df_rh.index = df_rh.index.tz_localize(None)

    # Final sort by time index
    df_a = df_a.sort_index()
    df_b = df_b.sort_index()
    df_rh = df_rh.sort_index()

    print("exporting dict")
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
    """Format PurpleAir data without pre-allocating massive arrays"""
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

    # Get dimensions
    positions = sorted(df_meta["position"].unique())
    time_index = dict_pa["a"].index  # All three DataFrames share the same time index

    # Process measurements - build sparse then densify only what's needed
    datasets = []

    for var_name in ["a", "b", "rh"]:
        print(f"Processing {var_name}...")
        df = dict_pa[var_name]

        # Use a dictionary to store only non-NaN values
        sparse_data = {}

        for sensor_idx in df.columns:
            if sensor_idx in sensor_to_pos:
                sensor_col = df[sensor_idx]
                # Skip if entire column is NaN
                if sensor_col.isna().all():
                    continue

                for period in sensor_to_pos[sensor_idx]:

                    pos_idx = positions.index(period["position"])

                    # Store only non-NaN values
                    valid_data = sensor_col.dropna()  # [mask].dropna()
                    for time_val, data_val in valid_data.items():
                        time_idx = time_index.get_loc(time_val)
                        sparse_data[(time_idx, pos_idx)] = data_val

        # Now create the array more efficiently
        print(f"Creating array for {var_name}...")
        values = np.full(
            (len(time_index), len(positions)), np.nan, dtype=np.float32
        )  # Use float32 to save memory

        # Fill only the non-NaN values
        for (t_idx, p_idx), val in sparse_data.items():
            values[t_idx, p_idx] = val

        # Create dataset and immediately clear the array from memory
        datasets.append(
            xr.Dataset(
                data_vars={var_name: (("time", "position"), values)},
                coords={"time": time_index, "position": positions},
            )
        )

        # Clear memory immediately
        del values, sparse_data, df
        gc.collect()

    # Merge measurements and filter to valid positions
    print("Merging datasets...")
    ds = xr.merge(datasets)

    # Clear the datasets list
    del datasets
    gc.collect()

    # Filter to positions that actually have data
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

    # Calculate disagreement
    ds["disagreement"] = np.abs((2 * (ds["a"] - ds["b"])) / (ds["a"] + ds["b"])) * 100

    ds["disagreement"] = ds["disagreement"].assign_attrs(
        long_name="Channel Disagreement",
        units="%",
        description="PurpleAir Node Internal Disagreement",
    )

    return ds
