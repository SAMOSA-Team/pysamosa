"""ECMWF ERA5 reanalysis data formatter."""

import os
import glob
import numpy as np
import xarray as xr


def _calculate_rh(ds: xr.Dataset) -> np.ndarray:
    """Calculate relative humidity from temperature and dew point.

    Args:
        ds: Dataset containing 't2m' and 'd2m' variables.

    Returns:
        Relative humidity array [%].
    """
    t2m = ds["t2m"].values
    d2m = ds["d2m"].values
    return 100 * (
        np.exp((17.625 * d2m) / (243.04 + d2m))
        / np.exp((17.625 * t2m) / (243.04 + t2m))
    )


def _calculate_spd(u10: np.ndarray, v10: np.ndarray) -> np.ndarray:
    """Calculate 10-meter wind speed from U and V components.

    Args:
        u10: East-west wind component [m/s].
        v10: North-south wind component [m/s].

    Returns:
        10-meter wind speed [m/s].
    """
    return (u10**2 + v10**2) ** 0.5


def _calculate_dir(u10: np.ndarray, v10: np.ndarray) -> np.ndarray:
    """Calculate 10-meter wind direction from U and V components.

    Args:
        u10: East-west wind component [m/s].
        v10: North-south wind component [m/s].

    Returns:
        10-meter wind direction [degrees].
    """
    return np.round(np.rad2deg(np.arctan2(-u10, -v10)) % 360, 0)


def _kelvin_to_celsius(ds: xr.Dataset, var_name: str) -> xr.Dataset:
    """Convert a dataset variable from Kelvin to Celsius.

    Args:
        ds: Dataset containing the variable to convert.
        var_name: Name of the variable to convert.

    Returns:
        Dataset with variable converted to Celsius.
    """
    converted = ds.copy()
    converted[var_name] = ds[var_name] - 273.15
    return converted


def _index_era(ds_era: xr.Dataset) -> xr.Dataset:
    """Remove the expver dimension and rename valid_time from an ERA5 dataset.

    Args:
        ds_era: ERA5 dataset possibly containing an expver dimension.

    Returns:
        Cleaned ERA5 dataset.
    """
    if len(ds_era.dims) > 3:
        ds_era1 = ds_era.isel(expver=0).reset_coords("expver", drop=True)
        ds_era5 = ds_era.isel(expver=1).reset_coords("expver", drop=True)
        ds_era = xr.merge([ds_era5, ds_era1])

    ds_era = ds_era.rename({"valid_time": "time"})
    ds_era = ds_era.drop(["number", "expver"])
    return ds_era


def format_era(in_path: str, crs: str = "EPSG:4326") -> xr.Dataset:
    """Format ERA5 data and compute derived variables (RH, wind speed, wind direction).

    Args:
        in_path: Path to the ERA5 data directory.
        crs: Coordinate reference system (default: EPSG:4326).

    Returns:
        Formatted ERA5 dataset with additional derived variables.
    """
    nc_files = glob.glob(os.path.join(in_path, "*_ERA5.nc"))

    if len(nc_files) > 1:
        ds_era = xr.open_mfdataset(
            nc_files,
            preprocess=_index_era,
            parallel=True,
            combine="nested",
            concat_dim="time",
            engine="scipy",
        )
    else:
        ds_era = xr.open_dataset(nc_files[0])
        ds_era = _index_era(ds_era)

    ds_era["time"] = (
        ds_era["time"]
        .to_index()
        .tz_localize("UTC")
        .tz_convert("Asia/Kolkata")
        .tz_localize(None)
    )
    ds_era = ds_era.sortby("time")

    ds_era = _kelvin_to_celsius(ds_era, "t2m")
    ds_era = _kelvin_to_celsius(ds_era, "d2m")

    rh2m = xr.Dataset(
        {"rh2m": (("time", "latitude", "longitude"), _calculate_rh(ds_era))},
        coords={
            "time": ds_era.time,
            "latitude": ds_era.latitude,
            "longitude": ds_era.longitude,
        },
        attrs={"units": "%", "long_name": "2 metre relative humidity"},
    )

    spd10 = xr.Dataset(
        {
            "spd10": (
                ("time", "latitude", "longitude"),
                _calculate_spd(ds_era.u10.values, ds_era.v10.values),
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
                _calculate_dir(ds_era.u10.values, ds_era.v10.values),
            )
        },
        coords={
            "time": ds_era.time,
            "latitude": ds_era.latitude,
            "longitude": ds_era.longitude,
        },
        attrs={"units": "degrees north", "long_name": "10 metre wind-direction"},
    )

    ds_era = xr.merge([ds_era, rh2m, spd10, wd10])
    ds_era = ds_era.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
    ds_era = ds_era.rio.write_crs(crs)
    ds_era = ds_era.sortby("time")
    ds_era = ds_era.resample(time="1h").mean()

    return ds_era
