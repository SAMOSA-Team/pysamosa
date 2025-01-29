"""
ECMWF Reanalysis Data
Last Updated: June 15, 2024
This script formats data downloaded off the Copernicus Data platform
from ERA5 API.
@author: markjcampmier
"""

# Import Packages
import os
import glob
import numpy as np
import xarray as xr


def calculate_rh(ds):
    """
    Calculate the relative humidity based on temperature and dew point in a xarray dataset.

    :param ds: Input dataset that contains 'temperature' and 'dew_point' data arrays.
    :type ds: xarray.Dataset
    :return Output dataset that includes the new 'relative_humidity' data array,
                        in addition to the existing arrays.
    :rtype: xarray.Dataset
    """

    # Extract temperature and dew point from dataset
    t2m = ds["t2m"].values
    d2m = ds["d2m"].values

    return 100 * (
        np.exp((17.625 * d2m) / (243.04 + d2m))
        / np.exp((17.625 * t2m) / (243.04 + t2m))
    )


def calculate_spd(u10, v10):
    """
    Calculates 10-meter wind speed. See NCAR wind reference for details.

    :param u10: 10-meter wind component in the x-direction (east-west) [m/s].
    :type u10: numpy.array
    :param v10: 10-meter wind component in the y-direction (north-south) [m/s].
    :type v10: numpy.array
    :return spd10: 10-meter wind speed [m/s].
    :rtype: numpy.array
    """

    spd10 = (u10**2 + v10**2) ** 0.5
    return spd10


def calculate_dir(u10, v10):
    """
    Calculates 10-meter wind direction. See NCAR wind reference for details.

    :param u10: 10-meter wind component in the x-direction (east-west) [m/s].
    :type u10: numpy.array
    :param v10: 10-meter wind component in the y-direction (north-south) [m/s].
    :type v10: numpy.array
    :returns dir10: 10-meter wind direction [degrees].
    :rtype: numpy.array
    """

    dir10 = np.round(np.rad2deg(np.arctan2(-u10, -v10)) % 360, 0)
    return dir10


def index_era(ds_era):
    """
    Removes the expver dimension from an ERA5 dataset.

    :param ds_era: xarray Dataset containing ERA5 data.
    :type ds_era: xarray.Dataset
    :returns ds_era: xarray Dataset with the expver dimension removed.
    :rtype: xarray.Dataset
    """

    if len(ds_era.dims) > 3:
        ds_era1 = ds_era.isel(expver=0).reset_coords("expver", drop=True)
        ds_era5 = ds_era.isel(expver=1).reset_coords("expver", drop=True)
        ds_era = xr.merge([ds_era5, ds_era1])

    ds_era = ds_era.rename({"valid_time": "time"})

    ds_era = ds_era.drop(["number", "expver"])

    return ds_era


def kelvin_to_celsius(data_array, var_name):
    """
    Converts a variable in a xarray Dataset from Kelvin to Celsius.

    :param data_array: The Dataset containing the variable to convert.
    :type data_array: xarray.Dataset
    :param var_name: The name of the variable to convert.
    :type var_name: str
    :returns A new Dataset with the variable converted to Celsius.
    :rtype: xarray.Dataset
    """
    # Make a copy of the original Dataset
    converted_array = data_array.copy()

    # Convert the specified variable from Kelvin to Celsius
    converted_array[var_name] = data_array[var_name] - 273.15

    # Return the updated Dataset
    return converted_array


def format_era(in_path, crs="EPSG:4326"):
    """
    Formats ERA5 data and calculates additional variables.

    :param in_path: Path to the ERA5 data file.
    :type in_path: str
    :param crs: Coordinate reference system (default: EPSG:4326).
    :type crs: str
    :returns ds_era: xarray Dataset containing formatted ERA5 data and additional variables.
    :rtype: xarray.Dataset
    """

    # Open ERA5 data with parallel processing

    if len(glob.glob(os.path.join(in_path, "*_ERA5.nc"))) > 1:

        ds_era = xr.open_mfdataset(
            glob.glob(os.path.join(in_path, "*_ERA5.nc")),
            preprocess=index_era,
            parallel=True,
            combine="nested",
            concat_dim="time",
            engine="scipy",
        )

    else:
        ds_era = xr.open_dataset(glob.glob(os.path.join(in_path, "*_ERA5.nc"))[0])
        ds_era = index_era(ds_era)

    ds_era["time"] = (
        ds_era["time"]
        .to_index()
        .tz_localize("UTC")
        .tz_convert("Asia/Kolkata")
        .tz_localize(None)
    )
    ds_era = ds_era.sortby("time")

    # ds_era = ds_era.resample(time='1h').mean()

    ds_era = kelvin_to_celsius(ds_era, "t2m")
    ds_era = kelvin_to_celsius(ds_era, "d2m")

    # Calculate relative humidity from temperature and dew point temperature
    rh2m = xr.Dataset(
        {"rh2m": (("time", "latitude", "longitude"), calculate_rh(ds_era))},
        coords={
            "time": ds_era.time,
            "latitude": ds_era.latitude,
            "longitude": ds_era.longitude,
        },
        attrs={"units": "%", "long_name": "2 metre relative humidity"},
    )

    # Calculate horizontal direction and speed from U & V components
    spd10 = xr.Dataset(
        {
            "spd10": (
                ("time", "latitude", "longitude"),
                calculate_spd(ds_era.u10.values, ds_era.v10.values),
            )
        },
        coords={
            "time": ds_era.time,
            "latitude": ds_era.latitude,
            "longitude": ds_era.longitude,
        },
        attrs={"units": "m s**-1", "long_name": "10 metre wind-speed"},
    )

    wd10 = xr.Dataset(
        {
            "wd10": (
                ("time", "latitude", "longitude"),
                calculate_dir(ds_era.u10.values, ds_era.v10.values),
            )
        },
        coords={
            "time": ds_era.time,
            "latitude": ds_era.latitude,
            "longitude": ds_era.longitude,
        },
        attrs={"units": "degrees north", "long_name": "10 metre wind-direction"},
    )

    # Merge all datasets
    ds_era = xr.merge([ds_era, rh2m, spd10, wd10])

    # Set spatial dimensions and coordinate reference system
    ds_era = ds_era.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
    ds_era = ds_era.rio.write_crs(crs)

    # Sort data by time
    ds_era = ds_era.sortby("time")
    ds_era = ds_era.resample(time="1h").mean()

    return ds_era
