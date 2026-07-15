"""QuantAQ sensor data formatter."""

import os
import glob
import pandas as pd
import xarray as xr


def _index_meta(in_path: str) -> xr.Dataset:
    """Read QuantAQ metadata and return as an xarray Dataset.

    Args:
        in_path: Path to the directory containing meta.feather.

    Returns:
        Metadata dataset indexed by sensor with latitude, longitude, and site coordinates.
    """
    df_meta = pd.read_feather(os.path.join(in_path, "meta.feather")).set_index("sensor")
    return df_meta.to_xarray().set_coords(["latitude", "longitude", "site"])


def _read_quantaq(file_processed: str, file_flag: str) -> xr.Dataset:
    """Read a single QuantAQ processed file and its flag file into an xarray Dataset.

    Args:
        file_processed: Path to the processed feather file.
        file_flag: Path to the flag feather file.

    Returns:
        Hourly QuantAQ dataset indexed by sensor and time.
    """
    df_qaq = (
        pd.read_feather(
            file_processed,
            columns=["time", "pm1", "pm25", "pm10", "sn", "met.rh", "met.temp"],
        )
        .set_index("time")
        .sort_index()
    )

    df_flag = (
        pd.read_feather(file_flag, columns=["timestamp_local", "flag", "sn"])
        .rename(columns={"timestamp_local": "time"})
        .set_index("time")
        .sort_index()
    )

    df_qaq = pd.concat([df_qaq, df_flag])
    df_qaq = df_qaq.resample("1h").agg(
        {
            "pm1": "mean",
            "pm25": "mean",
            "pm10": "mean",
            "met.rh": "mean",
            "met.temp": "mean",
            "flag": "sum",
            "sn": "first",
        }
    )
    df_qaq = (
        df_qaq.dropna(subset=["pm1", "pm25", "pm10"])
        .reset_index()
        .set_index(["sn", "time"])
    )

    return df_qaq.to_xarray()


def _index_quantaq(in_path: str) -> xr.Dataset:
    """Combine all QuantAQ processed files into a single xarray Dataset.

    Args:
        in_path: Path to the directory containing QuantAQ feather files.

    Returns:
        Merged QuantAQ dataset with renamed variables.
    """
    file_list = glob.glob(os.path.join(in_path, "*_processed.feather"))
    ds_list = [_read_quantaq(f, f.replace("_processed", "")) for f in file_list]
    ds_qaq = xr.merge(ds_list)
    ds_qaq = ds_qaq.rename(
        {
            "sn": "sensor",
            "pm1": "pm1_qaq",
            "pm25": "pm25_qaq",
            "pm10": "pm10_qaq",
            "met.rh": "rh_qaq",
            "met.temp": "temp_qaq",
            "flag": "flag_qaq",
        }
    )
    return ds_qaq


def format_quantaq(in_path: str) -> xr.Dataset:
    """Format QuantAQ sensor data from a directory of feather files.

    Args:
        in_path: Path to the directory containing QuantAQ feather files.

    Returns:
        Merged QuantAQ dataset with metadata, indexed by site and time.
    """
    ds_meta = _index_meta(in_path)
    ds_qaq = _index_quantaq(in_path)
    ds_qaq = xr.merge([ds_qaq, ds_meta])
    ds_qaq = ds_qaq.swap_dims({"sensor": "site"}).drop("sensor")
    return ds_qaq
