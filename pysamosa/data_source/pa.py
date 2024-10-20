"""
PurpleAir
Last Updated: June 15, 2024
This script formats the PurpleAir data downloaded
from the Earthmetry platform.
@author: markjcampmier
"""
# Import Packages
import os
import glob
import numpy as np
import pandas as pd
import xarray as xr

from scipy.interpolate import BSpline


# Define Functions
def read_pa(file, dict_position):
    """
    Formats PurpleAir data into a dictionary of DataFrames.

    :param file: The path to the PurpleAir data file.
    :type file: str
    :param dict_position: The position codes for site-sensor pairs.
    :type dict_position: dict
    :returns A dictionary of DataFrames, where the keys are `'a'`, `'b'`, and `'rh'`, and
        the values are the corresponding DataFrames for PM2.5 concentration,
        PM2.5 concentration, and relative humidity, respectively.
    :rtype: dict
    """

    # Read the PurpleAir data file.
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

    # Pivot table the data by observation hour and sensor identifier.
    df_pa_a = (
        df_pa.pivot_table(
            index="observation_hour_start",
            columns="position",
            values="pm25_atm_ch_a_average",
            aggfunc="mean",
        )
        .round(2)
        .resample("1h")
        .mean()
    )
    df_pa_b = (
        df_pa.pivot_table(
            index="observation_hour_start",
            columns="position",
            values="pm25_atm_ch_b_average",
            aggfunc="mean",
        )
        .round(2)
        .resample("1h")
        .mean()
    )
    df_pa_rh = (
        df_pa.pivot_table(
            index="observation_hour_start",
            columns="position",
            values="humidity_average",
            aggfunc="mean",
        )
        .round(2)
        .resample("1h")
        .mean()
    )

    # Return a dictionary of the DataFrames.
    return {"a": df_pa_a, "b": df_pa_b, "rh": df_pa_rh}


def index_pa(in_path, dict_position):
    """
    Reads PurpleAir data from a directory of files.

    :param in_path: The path to the directory containing the PurpleAir data files.
    :type in_path: str
    :param dict_position: The position codes for site-sensor pairs.
    :type dict_position: dict
    :returns dict_pa: A dictionary of DataFrames, where the keys are `'a'`, `'b'`, and `'rh'`, and
            the values are the corresponding DataFrames for PM2.5 concentration,
            PM2.5 concentration, and relative humidity, respectively.
    :rtype: dict_pa: dict
    """

    # Get a list of all the PurpleAir data files in the directory.
    lst_purpleair = glob.glob(os.path.join(in_path, "*_samosa.feather"))

    # Format each PurpleAir data file into a dictionary of DataFrames.
    lst_dict_pa = [read_pa(i, dict_position) for i in lst_purpleair]

    # Concatenate the DataFrames from each file into a single DataFrame for each
    # variable.
    dict_pa = {
        "a": pd.concat([i["a"] for i in lst_dict_pa]),
        "b": pd.concat([i["b"] for i in lst_dict_pa]),
        "rh": pd.concat([i["rh"] for i in lst_dict_pa]),
    }

    return dict_pa


def index_meta(in_path):
    """
    Formats the metadata file into a xarray compatible dataset to merge
    with the PurpleAir dataset.

    :param in_path: The path to the directory containing the PurpleAir data files.
    :type in_path: str
    :returns ds_meta: Dataset of PurpleAir metadata.
    :rtype: xarray.Dataset
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
    df_meta = df_meta.set_index("site_sensor")
    df_meta = df_meta.drop_duplicates()

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


def atm_to_cf1(atm):
    """
    Fits the PurpleAir ATM concentration into CF1.

    :param atm: Array of ATM data from PurpleAir data.
    :type atm: numpy.ndarray
    :return cf1: Fitted array of CF1 data from PurpleAir data.
    :rtype: cf1: numpy.ndarray
    """
    spl_data = np.load(
        "/Users/markcampmier/pysamosa/pysamosa/data_source/spline_parameters.npz"
    )
    spl = BSpline(spl_data["knots"], spl_data["coefficients"], 3)

    cf1 = atm.copy()
    cf1[atm < 20] = atm[atm < 20]
    cf1[(atm >= 20) & (atm <= 80)] = spl(atm[(atm >= 20) & (atm <= 80)])
    cf1[atm >= 80] = atm[atm >= 80] * 1.5

    return cf1


def format_pa(in_path):
    """
    Reads PurpleAir data from a directory of files.

    :param in_path:
    :type in_path: str
    :return ds: PurpleAir dataset
    :rtype ds: xarray.Dataset
    """
    ds_meta = index_meta(in_path)
    dict_position = {
        v: k
        for k, v in (
            ds_meta[["site_sensor"]].to_dataframe().to_dict()["site_sensor"]
        ).items()
    }
    ds_meta = ds_meta.drop(["site_sensor"])

    dict_pa = index_pa(in_path, dict_position)

    ds_a = (
        dict_pa["a"]
        .reset_index()
        .melt(id_vars="observation_hour_start", value_name="A")
        .set_index(["position", "observation_hour_start"])
        .to_xarray()
    )
    ds_b = (
        dict_pa["b"]
        .reset_index()
        .melt(id_vars="observation_hour_start", value_name="B")
        .set_index(["position", "observation_hour_start"])
        .to_xarray()
    )
    ds_rh = (
        dict_pa["rh"]
        .reset_index()
        .melt(id_vars="observation_hour_start", value_name="RH")
        .set_index(["position", "observation_hour_start"])
        .to_xarray()
    )
    ds_pa = xr.merge([ds_a, ds_b, ds_rh])

    ds_pa = ds_pa.rename({"observation_hour_start": "time"})

    ds_pa["A"] = xr.apply_ufunc(atm_to_cf1, ds_pa.A)
    ds_pa["B"] = xr.apply_ufunc(atm_to_cf1, ds_pa.B)

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

    ds = ds.rename_vars({"A": "a", "B": "b", "RH": "rh"})

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

    ds["latitude"].attrs["units"] = "degrees_north"
    ds["latitude"].attrs["long_name"] = "Latitude"

    ds["longitude"].attrs["units"] = "degrees_east"
    ds["longitude"].attrs["long_name"] = "Longitude"

    ds.attrs["history"] = "munged"

    return ds


# Fitting new spline for ATM -> CF1
"""
new_df = df[(df.loc[:,'PA_ATM (ug/m3)']>=20) & (df.loc[:,'PA_ATM (ug/m3)']<=80)].sort_values(by='PA_ATM (ug/m3)').round(0)
new_df = new_df.groupby('PA_ATM (ug/m3)').mean().reset_index()

x = new_df['PA_ATM (ug/m3)']
y = new_df['PA_CF1 (ug/m3)'].rolling(3, min_periods=1).mean()

# monotonic cubic spline interpolation
spl = make_interp_spline(x, y, k=3)

np.savez('spline_parameters.npz',
         knots=spl.t,
         coefficients=spl.c)
"""
