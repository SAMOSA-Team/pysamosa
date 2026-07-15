"""PurpleAir data formatter — PurpleAir platform."""

import os
import glob
import gc
import numpy as np
import pandas as pd
import xarray as xr


def _read_pr(file: str) -> pd.DataFrame:
    """Read a single PurpleAir feather file and resample to 2-minute intervals.

    Args:
        file: Path to the sensor feather file.

    Returns:
        DataFrame indexed by time with sensor data and purpleair_sensor_index column.
    """
    df_pa = pd.read_feather(file)
    df_pa.columns = ["timestamp", "rh", "temperature", "bscat_a", "bscat_b", "a", "b"]
    df_pa.loc[:, "time"] = pd.to_datetime(df_pa.timestamp, unit="s", utc=True)
    df_pa = df_pa.drop("timestamp", axis=1).set_index("time").sort_index()
    df_pa.index = df_pa.index.tz_convert("Asia/Kolkata")
    df_pa = df_pa.resample("2min").mean().round(2)
    df_pa.loc[:, "purpleair_sensor_index"] = int(
        file.split("/")[-1].split("_")[-1].replace(".feather", "")
    )
    return df_pa


def _index_pr(in_path: str) -> dict[str, pd.DataFrame]:
    """Read all PurpleAir feather files and build per-channel DataFrames.

    Args:
        in_path: Path to the directory containing per-sensor feather files.

    Returns:
        Dict with keys 'a', 'b', 'rh' mapping to wide DataFrames (columns = sensor_idx).
    """
    lst_folders = [
        f for f in os.listdir(in_path) if os.path.isdir(os.path.join(in_path, f))
    ]
    lst_files = [
        item
        for folder in lst_folders
        for item in glob.glob(os.path.join(in_path, folder, "sensor_*.feather"))
    ]

    sensor_data_a: dict = {}
    sensor_data_b: dict = {}
    sensor_data_rh: dict = {}

    for i, file in enumerate(lst_files):
        if i % 10 == 0:
            print(f"Processing file {i + 1}/{len(lst_files)}")

        df_single = _read_pr(file)
        sensor_idx = df_single["purpleair_sensor_index"].iloc[0]

        if sensor_idx not in sensor_data_a:
            sensor_data_a[sensor_idx] = []
            sensor_data_b[sensor_idx] = []
            sensor_data_rh[sensor_idx] = []

        sensor_data_a[sensor_idx].append(df_single["a"])
        sensor_data_b[sensor_idx].append(df_single["b"])
        sensor_data_rh[sensor_idx].append(df_single["rh"])
        del df_single

    df_a = pd.DataFrame()
    df_b = pd.DataFrame()
    df_rh = pd.DataFrame()

    for sensor_idx, data_a in sensor_data_a.items():
        sensor_a = pd.concat(data_a).sort_index()
        sensor_b = pd.concat(sensor_data_b[sensor_idx]).sort_index()
        sensor_rh = pd.concat(sensor_data_rh[sensor_idx]).sort_index()

        sensor_a = sensor_a[~sensor_a.index.duplicated(keep="first")]
        sensor_b = sensor_b[~sensor_b.index.duplicated(keep="first")]
        sensor_rh = sensor_rh[~sensor_rh.index.duplicated(keep="first")]

        if df_a.empty:
            df_a = pd.DataFrame({sensor_idx: sensor_a})
            df_b = pd.DataFrame({sensor_idx: sensor_b})
            df_rh = pd.DataFrame({sensor_idx: sensor_rh})
        else:
            df_a = df_a.join(pd.DataFrame({sensor_idx: sensor_a}), how="outer")
            df_b = df_b.join(pd.DataFrame({sensor_idx: sensor_b}), how="outer")
            df_rh = df_rh.join(pd.DataFrame({sensor_idx: sensor_rh}), how="outer")

    del sensor_data_a, sensor_data_b, sensor_data_rh

    for df in (df_a, df_b, df_rh):
        df.index.name = "time"
        df.index = df.index.tz_localize(None)

    return {"a": df_a.sort_index(), "b": df_b.sort_index(), "rh": df_rh.sort_index()}


def _index_meta(in_path: str) -> pd.DataFrame:
    """Read PurpleAir history metadata, assign positions, and encode settlement types.

    Args:
        in_path: Path to the directory containing history.feather.

    Returns:
        Metadata DataFrame with a 'position' column.
    """
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

    df_meta = pd.read_feather(
        os.path.join(in_path, "history.feather"), columns=meta_cols
    )
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


def format_pr(in_path: str) -> xr.Dataset:
    """Format PurpleAir (PurpleAir platform) data into a munged xarray Dataset.

    Args:
        in_path: Path to the directory containing per-sensor feather files and history.feather.

    Returns:
        PurpleAir dataset with sensor data and metadata coordinates.
    """
    df_meta = _index_meta(in_path)
    dict_pa = _index_pr(in_path)

    sensor_to_pos: dict = {}
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

    positions = sorted(df_meta["position"].unique())
    time_index = dict_pa["a"].index

    datasets = []
    for var_name in ["a", "b", "rh"]:
        print(f"Processing {var_name}...")
        df = dict_pa[var_name]
        sparse_data: dict = {}

        for sensor_idx in df.columns:
            if sensor_idx in sensor_to_pos:
                sensor_col = df[sensor_idx]
                if sensor_col.isna().all():
                    continue
                for period in sensor_to_pos[sensor_idx]:
                    pos_idx = positions.index(period["position"])
                    for time_val, data_val in sensor_col.dropna().items():
                        sparse_data[(time_index.get_loc(time_val), pos_idx)] = data_val

        print(f"Creating array for {var_name}...")
        values = np.full((len(time_index), len(positions)), np.nan, dtype=np.float32)
        for (t_idx, p_idx), val in sparse_data.items():
            values[t_idx, p_idx] = val

        datasets.append(
            xr.Dataset(
                data_vars={var_name: (("time", "position"), values)},
                coords={"time": time_index, "position": positions},
            )
        )
        del values, sparse_data, df
        gc.collect()

    print("Merging datasets...")
    ds = xr.merge(datasets)
    del datasets
    gc.collect()

    has_data = ~np.isnan(ds.a.values).all(axis=0)
    valid_positions = ds.position.values[has_data]
    ds = ds.sel(position=valid_positions)

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

    meta_coords = (
        df_meta[["position"] + list(coord_mappings.keys())]
        .drop_duplicates("position")
        .set_index("position")
        .loc[valid_positions]
        .rename(columns=coord_mappings)
    )

    for new_name in coord_mappings.values():
        ds = ds.assign_coords({new_name: ("position", meta_coords[new_name].values)})

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
