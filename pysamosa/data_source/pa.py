"""PurpleAir data formatter — Earthmetry platform."""

import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

from scipy.interpolate import BSpline


def _read_pa(file: str, dict_position: dict) -> dict[str, pd.DataFrame]:
    """Format a single PurpleAir feather file into per-channel DataFrames.

    Args:
        file: Path to a PurpleAir feather file.
        dict_position: Mapping of site_sensor string to integer position code.

    Returns:
        Dict with keys 'a', 'b', 'rh' and pivoted DataFrames as values.
    """
    df_pa = pd.read_feather(
        file,
        columns=[
            "observation_hour_start",
            "samosa_identifier",
            "pm25_atm_ch_a_average",
            "pm25_atm_ch_b_average",
            "humidity_average",
            "site_id",
        ],
    )

    df_pa["site_sensor"] = (
        df_pa["site_id"].astype(str) + "&" + df_pa["samosa_identifier"].astype(str)
    )
    df_pa["site_sensor"] = df_pa["site_sensor"].replace(dict_position)
    df_pa.rename(columns={"site_sensor": "position"}, inplace=True)

    kwargs = dict(index="observation_hour_start", aggfunc="mean")
    df_pa_a = (
        df_pa.pivot_table(columns="position", values="pm25_atm_ch_a_average", **kwargs)
        .round(2)
        .resample("1h")
        .mean()
    )
    df_pa_b = (
        df_pa.pivot_table(columns="position", values="pm25_atm_ch_b_average", **kwargs)
        .round(2)
        .resample("1h")
        .mean()
    )
    df_pa_rh = (
        df_pa.pivot_table(columns="position", values="humidity_average", **kwargs)
        .round(2)
        .resample("1h")
        .mean()
    )

    return {"a": df_pa_a, "b": df_pa_b, "rh": df_pa_rh}


def _index_pa(in_path: str, dict_position: dict) -> dict[str, pd.DataFrame]:
    """Read all PurpleAir feather files from a directory and concatenate them.

    Args:
        in_path: Path to the directory containing PurpleAir feather files.
        dict_position: Mapping of site_sensor string to integer position code.

    Returns:
        Dict with keys 'a', 'b', 'rh' and concatenated DataFrames.
    """
    lst_purpleair = glob.glob(os.path.join(in_path, "*_samosa.feather"))
    lst_dict_pa = [_read_pa(i, dict_position) for i in lst_purpleair]

    return {
        "a": pd.concat([i["a"] for i in lst_dict_pa]),
        "b": pd.concat([i["b"] for i in lst_dict_pa]),
        "rh": pd.concat([i["rh"] for i in lst_dict_pa]),
    }


def _index_meta(in_path: str) -> xr.Dataset:
    """Read and format the PurpleAir history metadata into an xarray Dataset.

    Args:
        in_path: Path to the directory containing history.feather.

    Returns:
        Metadata dataset indexed by position.
    """
    df_meta = pd.read_feather(
        os.path.join(in_path, "history.feather"),
        columns=[
            "samosa_identifier",
            "site_id",
            "effective_date",
            "discontinue_date",
            "settlement_name",
            "settlement_type",
            "is_collocation_site",
            "site_latitude",
            "site_longitude",
            "site_district_name",
            "site_state_name",
            "all_land_uses",
            "cluster_name",
        ],
    )

    df_meta = df_meta.drop(["effective_date", "discontinue_date"], axis=1)
    df_meta["site_sensor"] = (
        df_meta["site_id"].astype(str) + "&" + df_meta["samosa_identifier"].astype(str)
    )
    df_meta = df_meta.set_index("site_sensor").drop_duplicates()
    df_meta["position"] = range(len(df_meta.index))
    df_meta = df_meta.reset_index().set_index("position")

    df_meta.loc[df_meta.settlement_type == "Large City", "settlement_type"] = 2
    df_meta.loc[df_meta.settlement_type == "Small City", "settlement_type"] = 1
    df_meta.loc[
        (df_meta.settlement_type != 2) & (df_meta.settlement_type != 1),
        "settlement_type",
    ] = 0
    df_meta["settlement_type"] = df_meta["settlement_type"].astype(int)

    ds_meta = df_meta.to_xarray()
    ds_meta = ds_meta.rename(
        {
            "site_id": "site",
            "samosa_identifier": "sensor",
            "site_latitude": "latitude",
            "site_longitude": "longitude",
            "settlement_name": "settlement",
            "all_land_uses": "land_uses",
            "site_district_name": "district",
            "site_state_name": "state",
            "cluster_name": "cluster",
        }
    )
    return ds_meta


def _atm_to_cf1(atm: np.ndarray) -> np.ndarray:
    """Convert PurpleAir ATM concentration to CF1 using a fitted B-spline.

    Args:
        atm: Array of ATM PM2.5 values from PurpleAir.

    Returns:
        CF1-corrected PM2.5 array.
    """
    spline_path = Path(__file__).parent / "spline_parameters.npz"
    spl_data = np.load(spline_path)
    spl = BSpline(spl_data["knots"], spl_data["coefficients"], 3)

    cf1 = atm.copy()
    cf1[atm < 20] = atm[atm < 20]
    cf1[(atm >= 20) & (atm <= 80)] = spl(atm[(atm >= 20) & (atm <= 80)])
    cf1[atm >= 80] = atm[atm >= 80] * 1.5

    return cf1


def format_pa(in_path: str) -> xr.Dataset:
    """Format PurpleAir (Earthmetry) data into a munged xarray Dataset.

    Args:
        in_path: Path to the directory containing PurpleAir feather files.

    Returns:
        Merged PurpleAir dataset with CF1-corrected PM2.5 and metadata.
    """
    ds_meta = _index_meta(in_path)
    dict_position = {
        v: k
        for k, v in (
            ds_meta[["site_sensor"]].to_dataframe().to_dict()["site_sensor"]
        ).items()
    }
    ds_meta = ds_meta.drop(["site_sensor"])

    dict_pa = _index_pa(in_path, dict_position)

    ds_a = (
        dict_pa["a"]
        .reset_index()
        .melt(id_vars="observation_hour_start", value_name="a")
        .set_index(["position", "observation_hour_start"])
        .to_xarray()
    )
    ds_b = (
        dict_pa["b"]
        .reset_index()
        .melt(id_vars="observation_hour_start", value_name="b")
        .set_index(["position", "observation_hour_start"])
        .to_xarray()
    )
    ds_rh = (
        dict_pa["rh"]
        .reset_index()
        .melt(id_vars="observation_hour_start", value_name="rh")
        .set_index(["position", "observation_hour_start"])
        .to_xarray()
    )
    ds_pa = xr.merge([ds_a, ds_b, ds_rh])
    ds_pa = ds_pa.rename({"observation_hour_start": "time"})

    ds_pa["a"] = xr.apply_ufunc(_atm_to_cf1, ds_pa.a)
    ds_pa["b"] = xr.apply_ufunc(_atm_to_cf1, ds_pa.b)

    ds = xr.merge([ds_pa, ds_meta])
    ds = ds.set_coords(
        [
            "latitude",
            "longitude",
            "land_uses",
            "settlement",
            "settlement_type",
            "is_collocation_site",
            "sensor",
            "site",
            "state",
            "district",
            "cluster",
        ]
    )

    for var in ds.data_vars:
        ds[var].encoding = {"_FillValue": np.nan}

    ds["a"] = ds["a"].assign_attrs(
        long_name="PurpleAir Sensor A",
        units="ug/m^3",
        description="PurpleAir Sensor A PM2.5 CF1 data",
    )
    ds["b"] = ds["b"].assign_attrs(
        long_name="PurpleAir Sensor B",
        units="ug/m^3",
        description="PurpleAir Sensor B PM2.5 CF1 data",
    )
    ds["rh"] = ds["rh"].assign_attrs(
        long_name="Relative Humidity",
        units="%",
        description="PurpleAir Relative Humidity",
    )

    ds["disagreement"] = np.abs((2 * (ds["a"] - ds["b"])) / (ds["a"] + ds["b"])) * 100
    ds["disagreement"] = ds["disagreement"].assign_attrs(
        long_name="Channel Disagreement",
        units="%",
        description="PurpleAir Node Internal Disagreement",
    )

    ds["latitude"].attrs.update({"units": "degrees_north", "long_name": "Latitude"})
    ds["longitude"].attrs.update({"units": "degrees_east", "long_name": "Longitude"})
    ds.attrs["history"] = "munged"

    return ds
