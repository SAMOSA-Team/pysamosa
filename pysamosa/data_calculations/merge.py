"""
Merge
Last Updated: June 18, 2024
This script supplies the basic functions for merging
xarray timeseries from both raster and point
geometries.
@author: markjcampmier
"""

# Import Packages
import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import xarray as xr

from pysamosa.data_calculations import india


# Define Functions
def set_phase(start, end, phase_num):
    """
    Function to make a time-series of labeled campaign collocation phases.

    :param start: Beginning of collocation phase
    :type start: pd.Timestamp
    :param end: End of collocation phase
    :type end: pd.Timestamp
    :param phase_num: Collocation phase label, typically an integer
    :type phase_num: int
    :return: Time-indexed collocation phase dataframe
    :rtype: pandas.DataFrame

    """
    index = pd.Series(pd.date_range(start, end, freq="1H"), name="time")
    arr_phase_num = np.repeat(phase_num, len(index))
    return pd.DataFrame(arr_phase_num, index=index, columns=["collocation_phase"])


def fixed_raster_merge(ds_points, ds_raster, keys="site", xy=None):
    """
    Function to extract and merge raster cells based on point data.

    :param ds_points: The xarray dataset containing the point data.
    :type ds_points: xarray.Dataset
    :param ds_raster: The xarray dataset containing the raster data.
    :type ds_raster: xarray.Dataset
    :param keys: The key to use for the point data merge.
    :type keys: str
    :param xy: The x and y coordinates to merge for single cell merges.
    :type xy: list
    :returns The merged dataset containing the raster cells for each point.
    :rtype: xarray.Dataset
    """

    cells = []

    for key in ds_points[keys]:
        if xy is not None:
            latitude, longitude = xy[0], xy[1]
        else:
            latitude = ds_points.sel({keys: key}).latitude.values
            longitude = ds_points.sel({keys: key}).longitude.values

        # Extract the corresponding raster cell (for now it's a try/except, will fix the x,y vs lat,lon later)
        try:
            cell = ds_raster.sel(
                indexers={"latitude": latitude, "longitude": longitude},
                method="nearest",
            )
            cell = cell.drop_vars(names=["longitude", "latitude"])
            cell.coords[keys] = key
        except KeyError:
            cell = ds_raster.sel(
                indexers={"x": latitude, "y": longitude}, method="nearest"
            )
            cell = cell.drop_vars(names=["x", "y"])
            cell.coords[keys] = key
        cells.append(cell)

    return xr.combine_nested(cells, concat_dim=keys)


def find_nearest_point(df_coords_1, df_coords_2, tolerance=0.005):
    """
    Finds nearest collocation station based on tolerance between two points.

    :param df_coords_1: Dataframe with latitude and longitude coordinates of regulatory sites
    :type df_coords_1: pandas.DataFrame
    :param df_coords_2:  with latitude and longitude coordinates of LCS sites
    :type df_coords_2: pandas.DataFrame
    :param tolerance:
    :type tolerance: float
    :return: df_coords_2: Matched DataFrame
    :rtype df_coords_2: pandas.DataFrame
    """

    tree = cKDTree(list(zip(df_coords_1["latitude"], df_coords_1["longitude"])))

    for i, row in df_coords_2.iterrows():
        dist, idx = tree.query(
            [row["latitude"], row["longitude"]], k=1, distance_upper_bound=tolerance
        )

        if not np.isinf(dist):
            df_coords_2.loc[i, "collocation_site"] = df_coords_1.iloc[idx, :].name

    df_coords_2 = df_coords_2.replace("nan", np.nan).dropna(subset=["collocation_site"])
    return df_coords_2


def fixed_point_merge(ds_points, ds_merge, tolerance=0.005, keys="site", xy=None):
    """
    Merge two datasets with fixed point geometries.

    :param ds_points: fixed point dataset to merge on
    :type: ds_points: xarray.DataArray
    :param ds_merge: fixed point dataset to merge to
    :type: ds_merge: xarray.DataArray
    :param tolerance: float
    :type: tolerance: float
    :param keys: Dimension to use for merging, 'sensor' by default.
    :type keys: str
    :param xy: The x and y coordinates to merge for single cell merges.
    :type xy: list
    :return: ds_matched: The merged xarray.DataArray
    :rtype: xarray.DataArray
    """
    if xy is not None:
        df_coords_points = pd.DataFrame(columns=["latitude", "longitude"])
        df_coords_points.loc[0, "latitude"] = xy[0]
        df_coords_points.loc[0, "longitude"] = xy[1]
    else:
        df_coords_points = (
            ds_points[["latitude", "longitude"]]
            .to_dataframe()
            .dropna()
            .drop_duplicates()
        )

    df_coords_merge = (
        ds_merge[["latitude", "longitude"]]
        .to_dataframe()
        .dropna(subset=["latitude", "longitude"])
        .drop_duplicates()
    )
    df_matched = find_nearest_point(
        df_coords_merge, df_coords_points, tolerance=tolerance
    )

    ds_matched_merge = ds_merge.sel({keys: df_matched.collocation_site.values})
    ds_matched_points = ds_points.sel({keys: df_matched.index.values})

    return ds_matched_points, ds_matched_merge


def open_and_merge(in_path, file_list):
    """
    Open and merge all datasets in the list.

    :param in_path: path to flagged data
    :param file_list:
    :return: merged dataset
    :rtype: xarray.Dataset
    """
    return xr.merge(
        [xr.open_dataset(os.path.join(in_path, f"{file}.nc")) for file in file_list]
    )


def open_and_merge_raster(in_path, ds_ref, raster_list, xy=None, keys="site"):
    """
    Open a list of files and merge all raster cells.

    :param in_path: path to flagged data
    :type in_path: str
    :param ds_ref:
    :type ds_ref: xarray.Dataset
    :param raster_list
    :type: raster_list: list
    :param xy: The x and y coordinates to merge for single cell merges.
    :type xy: list
    :param keys: Dimension to use for merging, 'sites' by default.
    :type keys: str
    :return: merged dataset
    :rtype: xarray.Dataset
    """
    return xr.merge(
        [
            fixed_raster_merge(
                ds_ref,
                xr.open_dataset(os.path.join(in_path, f"{raster}.nc")),
                xy=xy,
                keys=keys,
            )
            for raster in raster_list
        ]
    )


def make_history(ds):
    """
    Sensor-wise sensor dataset, location agnostic.

    :param ds: PurpleAir dataset
    :type ds: xarray.Dataset
    :return: History dataset
    :rtype: xarray.Dataset
    """
    ds_history = ds.drop(
        [
            "land_uses",
            "settlement",
            "district",
            "site",
            "state",
            "cluster",
            "settlement_type",
            "is_collocation_site",
            "latitude",
            "longitude",
        ]
    )
    ds_history = (
        ds_history.to_dataframe()
        .reset_index()
        .drop(["position"], axis=1)
        .set_index(["sensor", "time"])
        .drop_duplicates()
        .dropna(how="all")
        .to_xarray()
    )
    ds_history_season = india.get_season(
        ds_history["time"].to_dataframe().reset_index(drop=True).set_index("time"),
        False,
    ).to_xarray()
    ds_history = xr.merge([ds_history, ds_history_season]).set_coords("season")
    ds_history["pa_raw_mean"] = (ds_history.a + ds_history.b) * 0.5
    return ds_history


def make_phases(ds, dict_phases):
    """
    Pre-deployment collocation phase, here pre-deployment is set as IITD.

    :param ds: PurpleAir dataset
    :type ds: xarray.Dataset
    :param dict_phases: Dictionary with start_date and end_date as keys
    :type dict_phases: dict
    :return: Phase dataset
    :rtype: xarray.Dataset
    """
    df_collocation_phase = pd.concat(
        [
            set_phase(start, end, i + 1)
            for start, end, i in zip(
                dict_phases["start"],
                dict_phases["end"],
                range(len(dict_phases["start"])),
            )
        ]
    )

    ds_phases = (
        ds.where(ds.site == "IITD")
        .dropna(dim="position", how="all")
        .dropna(dim="time", how="all")
    )
    ds_phases = ds_phases.drop(
        [
            "land_uses",
            "settlement",
            "district",
            "state",
            "cluster",
            "is_collocation_site",
            "latitude",
            "longitude",
            "settlement_type",
            "site",
        ]
    )
    ds_phases = (
        ds_phases.to_dataframe()
        .reset_index()
        .drop(["position"], axis=1)
        .set_index(["sensor", "time"])
        .to_xarray()
    )
    ds_phases_season = india.get_season(
        ds_phases["time"].to_dataframe().reset_index(drop=True).set_index("time"), False
    ).to_xarray()
    ds_phases = xr.merge(
        [ds_phases, df_collocation_phase.to_xarray(), ds_phases_season]
    ).set_coords(["collocation_phase", "season"])
    return ds_phases


def make_collocation(ds):
    """
    Makes collocation dataset based on known collocations with regulatory instruments.

    :param ds: PurpleAir dataset
    :type ds: xarray.Dataset
    :return ds_collocation: Collocation dataset
    :rtype: xarray.Dataset
    """
    ds_collocation = ds.where(ds.settlement == "Delhi", drop=True)
    ds_collocation = ds_collocation.drop(
        [
            "settlement",
            "district",
            "state",
            "cluster",
            "is_collocation_site",
            "land_uses",
            "settlement_type",
        ]
    )

    df_pa = pd.concat(
        [
            ds_collocation[["a"]]
            .to_dataframe()
            .reset_index()
            .drop(["position", "sensor", "latitude", "longitude"], axis=1)
            .rename(columns={"a": "pa"}),
            ds_collocation[["b"]]
            .to_dataframe()
            .reset_index()
            .drop(["position", "sensor", "latitude", "longitude"], axis=1)
            .rename(columns={"b": "pa"}),
        ]
    )

    df_pa = df_pa.groupby(["site", "time"]).agg(
        pa_raw=("pa", "mean"), cv=("pa", lambda x: x.std() / x.mean())
    )

    df_rh = (
        ds_collocation[["rh"]]
        .to_dataframe()
        .reset_index()
        .drop(["position", "sensor", "latitude", "longitude"], axis=1)
        .groupby(["site", "time"])
        .mean()
    )

    df_disagreement = (
        ds_collocation[["disagreement"]]
        .to_dataframe()
        .reset_index()
        .drop(["position", "sensor", "latitude", "longitude"], axis=1)
        .groupby(["site", "time"])
        .mean()
    )

    df_collocation = pd.concat([df_pa, df_rh, df_disagreement], axis=1).dropna(
        how="all"
    )

    ds_collocation_meta = (
        ds_collocation[["latitude", "longitude"]]
        .drop("sensor")
        .to_dataframe()
        .reset_index(drop=True)
        .set_index("site")
        .drop_duplicates()
        .to_xarray()
    )
    ds_collocation_season = india.get_season(
        ds_collocation["time"].to_dataframe().reset_index(drop=True).set_index("time"),
        False,
    ).to_xarray()

    ds_collocation = xr.merge(
        [df_collocation.to_xarray(), ds_collocation_meta, ds_collocation_season]
    ).set_coords(["latitude", "longitude", "season"])

    return ds_collocation


def make_deployment(ds):
    """
    Makes campaign deployment (i.e., non-collocation) dataset.

    :param ds: PurpleAir dataset
    :type ds: xarray.Dataset
    :return ds_deployment: Deployment dataset
    :rtype: xarray.Dataset
    """

    ds_deployment = ds.where(ds.site != "IITD")

    ds_deployment["pa_raw_mean"] = (ds_deployment.a + ds_deployment.b) * 0.5
    ds_deployment["pa_raw_std"] = (ds_deployment.a + ds_deployment.a) * 0.5

    ds_deployment = xr.merge(
        [
            ds_deployment.pa_raw_mean.resample(time="1h").mean(),
            ds_deployment.pa_raw_std.resample(time="1h").std(),
            ds_deployment.rh.resample(time="1h").mean(),
            ds_deployment.disagreement.resample(time="1h").mean(),
        ]
    )

    df_deploy = ds_deployment.to_dataframe().reset_index().drop(["position"], axis=1)
    ds_deploy = (
        df_deploy.groupby(["site", "time"])
        .agg(
            {
                "pa_raw_mean": "first",
                "pa_raw_std": "first",
                "rh": "first",
                "disagreement": "first",
                "sensor": "first",
            }
        )
        .to_xarray()
    )

    print(ds_deploy.pa_raw_mean.shape)

    ds_deploy_meta = (
        ds_deployment[["site"]]
        .to_dataframe()
        .reset_index(drop=True)
        .set_index("site")
        .sort_index()
        .drop(["sensor"], axis=1)
        .drop_duplicates()
        .to_xarray()
    )
    ds_deployment = xr.merge([ds_deploy, ds_deploy_meta])

    print(ds_deployment.pa_raw_mean.shape)

    ds_deployment = ds_deployment.set_coords(
        [
            "land_uses",
            "settlement",
            "district",
            "state",
            "settlement_type",
            "cluster",
            "latitude",
            "longitude",
        ]
    ).drop(["is_collocation_site"])

    ds_deployment_season = india.get_season(
        ds_deployment["time"].to_dataframe().reset_index(drop=True).set_index("time"),
        False,
    ).to_xarray()
    ds_deployment = xr.merge([ds_deployment, ds_deployment_season]).set_coords("season")

    print(ds_deployment.pa_raw_mean.shape)

    return ds_deployment


def merge_history(in_path):
    # ds_pa = xr.open_dataset(os.path.join(in_path, "pa.nc"))
    ds_pa = xr.open_dataset(os.path.join(in_path, "pr.nc"))
    ds_history = make_history(ds_pa)
    return ds_history


def merge_reference(in_path):
    def dataset_exists(dataset_name):
        return os.path.exists(os.path.join(in_path, f"{dataset_name}.nc"))

    ds_reference = open_and_merge(in_path, ["bam", "reg"])

    datasets_to_merge = ["era", "tropomi", "martin", "ghsl"]

    existing_datasets = [
        dataset for dataset in datasets_to_merge if dataset_exists(dataset)
    ]

    if existing_datasets:
        ds_reference_rasters = open_and_merge_raster(
            in_path, ds_reference, existing_datasets, keys="site"
        )
        return xr.merge([ds_reference, ds_reference_rasters])
    else:
        return ds_reference


def merge_phases(in_path, dict_phases):
    # ds_pa = xr.open_dataset(os.path.join(in_path, "pa.nc"))
    ds_pa = xr.open_dataset(os.path.join(in_path, "pr.nc"))
    ds_phases = make_phases(ds_pa, dict_phases)
    ds_bam = fixed_point_merge(
        ds_phases,
        xr.open_dataset(os.path.join(in_path, "bam.nc")),
        xy=[28.5468, 77.1906],
    )
    return xr.merge([ds_phases, ds_bam])


def merge_collocation(in_path):
    # ds_pa = xr.open_dataset(os.path.join(in_path, "pa.nc"))
    ds_pa = xr.open_dataset(os.path.join(in_path, "pr.nc"))

    ds_pa = xr.merge(
        [
            ds_pa.a.resample(time="1h").mean(),
            ds_pa.b.resample(time="1h").mean(),
            ds_pa.rh.resample(time="1h").mean(),
            ds_pa.disagreement.resample(time="1h").mean(),
            ds_pa.a_flag.resample(time="1h").sum(),
            ds_pa.b_flag.resample(time="1h").sum(),
        ]
    )

    ds_collocation = make_collocation(ds_pa)
    ds_reference = open_and_merge(in_path, ["bam", "reg"])

    ds_collocation, ds_collocation_ref = fixed_point_merge(ds_collocation, ds_reference)
    ds_collocation["site"] = ds_collocation_ref["site"]
    ds_collocation["latitude"] = ds_collocation_ref["latitude"]
    ds_collocation["longitude"] = ds_collocation_ref["longitude"]
    ds_collocation = xr.merge(
        [ds_collocation, xr.open_dataset(os.path.join(in_path, "legacy.nc"))]
    )
    ds_rasters = open_and_merge_raster(
        in_path, ds_collocation, ["era", "tropomi", "martin", "ghsl"], keys="site"
    )
    return xr.merge([ds_collocation, ds_collocation_ref, ds_rasters])


def merge_deployment(in_path):
    def dataset_exists(dataset_name):
        return os.path.exists(os.path.join(in_path, f"{dataset_name}.nc"))

    # ds_pa = xr.open_dataset(os.path.join(in_path, "pa.nc"))
    ds_pa = xr.open_dataset(os.path.join(in_path, "pr.nc"))
    ds_deployment = make_deployment(ds_pa)
    datasets_to_merge = ["era", "tropomi", "martin", "ghsl", "rwi"]

    existing_datasets = [
        dataset for dataset in datasets_to_merge if dataset_exists(dataset)
    ]

    if existing_datasets:
        ds_raster = open_and_merge_raster(
            in_path, ds_deployment, existing_datasets, keys="site"
        )
        return xr.merge([ds_deployment, ds_raster])
    else:
        return ds_deployment
