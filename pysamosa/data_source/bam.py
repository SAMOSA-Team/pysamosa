"""MetOne BAM-1022 format and export."""

import os
import threading
import numpy as np
import pandas as pd
import xarray as xr


def _check_bam(bam_path: str) -> list[str]:
    """Check BAM files for 'Data Report' header and return a list of valid file paths.

    Args:
        bam_path: Path to the BAM data directory.

    Returns:
        Valid BAM file paths.
    """
    csv_files, bam_files = [], []
    for root, _, files in os.walk(bam_path):
        for filename in files:
            if filename.endswith(".csv") or filename.endswith(".CSV"):
                csv_files.append(os.path.join(root, filename))
    for csv_file in csv_files:
        with open(csv_file, "rb") as csv:
            header = csv.read(4096)
            header_lines = header.decode("utf-8").splitlines()[:5]
            if "Data Report" in header_lines:
                bam_files.append(csv_file)
    return bam_files


def _read_bam(bam_path: str) -> tuple[pd.DataFrame, str]:
    """Read BAM files from a directory into a DataFrame.

    Args:
        bam_path: Path to a single BAM station directory.

    Returns:
        Tuple of the hourly DataFrame and the station name.
    """
    bam_name = os.path.basename(bam_path)
    files = _check_bam(bam_path)
    df_bam = pd.concat(
        [
            pd.read_csv(
                file,
                skiprows=4,
                parse_dates=True,
                index_col=0,
                usecols=["Time", "ConcHR(ug/m3)", "Status"],
            )
            for file in files
        ]
    )
    df_bam = df_bam.resample("1h").aggregate({"ConcHR(ug/m3)": "mean", "Status": "sum"})
    df_bam.index = df_bam.index - pd.Timedelta("1h")
    df_bam.loc[:, "bam_name"] = bam_name
    df_bam = df_bam.reset_index().set_index(["Time", "bam_name"])
    return df_bam, bam_name


def _read_meta(in_path: str) -> xr.Dataset:
    """Read BAM metadata CSV and return as an xarray Dataset.

    Args:
        in_path: Path to the BAM data directory containing bam_meta.csv.

    Returns:
        Metadata dataset.
    """
    df_meta = pd.read_csv(os.path.join(in_path, "bam_meta.csv")).set_index("sensor")
    return df_meta.to_xarray()


def _index_bam(bam_path: str, ds_list: list) -> None:
    """Read and append a single BAM station dataset to ds_list (thread target).

    Args:
        bam_path: Path to a single BAM station directory.
        ds_list: Shared list accumulating per-station datasets.
    """
    df_bam, _ = _read_bam(bam_path)
    ds_list.append(df_bam.to_xarray())


def format_bam(in_path: str) -> xr.Dataset:
    """Format BAM files into a munged PM2.5 xarray Dataset.

    Args:
        in_path: Path to the BAM data directory.

    Returns:
        Merged, QA-ready BAM dataset.
    """
    threads, ds_list = [], []
    bam_paths = [
        os.path.join(in_path, f)
        for f in os.listdir(in_path)
        if os.path.isdir(os.path.join(in_path, f))
    ]

    ds_meta = _read_meta(in_path)

    for bam_path in bam_paths:
        thread = threading.Thread(target=_index_bam, args=(bam_path, ds_list))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    ds_bams = xr.merge(ds_list)
    ds_bams = ds_bams.rename({"bam_name": "sensor", "Time": "time"})
    ds_bams = xr.merge([ds_bams, ds_meta])
    ds_bams = (
        ds_bams.to_dataframe()
        .reset_index()
        .groupby(["site", "time"])
        .agg({"ConcHR(ug/m3)": "mean", "Status": "sum"})
        .to_xarray()
    )
    ds_bams = xr.merge(
        [
            ds_bams,
            ds_meta.to_dataframe()
            .reset_index(drop=True)
            .set_index("site")
            .drop_duplicates()
            .to_xarray()
            .set_coords(["latitude", "longitude"]),
        ]
    )

    ds_bams = ds_bams.rename_vars({"ConcHR(ug/m3)": "pm25", "Status": "status"})
    ds_bams = ds_bams.sortby("time")

    ds_bams["pm25"].encoding = {"_FillValue": np.nan}
    ds_bams["status"].encoding = {"_FillValue": 256}

    ds_bams["pm25"] = ds_bams["pm25"].assign_attrs(
        long_name="PM2.5", units="ug/m^3", description="BAM-1022 PM2.5 data"
    )
    ds_bams["status"] = ds_bams["status"].assign_attrs(
        long_name="Status", description="BAM reported bit-masked status flags."
    )
    ds_bams["latitude"].attrs.update(
        {"units": "degrees_north", "long_name": "Latitude"}
    )
    ds_bams["longitude"].attrs.update(
        {"units": "degrees_east", "long_name": "Longitude"}
    )
    ds_bams.attrs["history"] = "munged"

    return ds_bams
