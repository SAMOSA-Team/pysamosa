"""Quality assurance flag functions for each data source."""

import numpy as np
import xarray as xr


def qa_bam(ds: xr.Dataset) -> xr.Dataset:
    """Apply quality assurance flags to MetOne BAM data.

    Flag values:
        0 — valid data
        1 — missing (NaN)
        2 — outside limit of detection (< 3 or > 1000 µg/m³)
        3 — instrument-reported error (Status > 0)

    Args:
        ds: BAM dataset containing 'pm25' and 'status' variables.

    Returns:
        Dataset with 'pm25_flag' added and 'status' dropped.
    """
    ds["pm25_flag"] = xr.where(
        np.isnan(ds.pm25),
        1,
        xr.where((ds.pm25 < 3) | (ds.pm25 > 1000), 2, xr.where(ds.status > 0, 3, 0)),
    )
    ds.pm25_flag.encoding["_FillValue"] = 1
    ds = ds.drop_vars("status")
    ds.attrs["history"] = "flagged"
    return ds


def qa_reg(ds: xr.Dataset) -> xr.Dataset:
    """Apply quality assurance flags to CPCB regulatory data.

    Flag values per pollutant:
        0 — valid data
        1 — missing (NaN)
        2 — outside limit of detection
        3 — PM2.5 > PM10 while both flags are zero (physical inconsistency)

    Args:
        ds: CPCB dataset containing pm25, pm10, co, no2, so2, o3 variables.

    Returns:
        Dataset with flag variables added.
    """
    ds["pm25_flag"] = xr.where(
        np.isnan(ds.pm25),
        1,
        xr.where(
            (ds.pm25 < 3) | (ds.pm25 > 1000), 2, xr.where(ds.pm25 > ds.pm10, 3, 0)
        ),
    )
    ds["pm10_flag"] = xr.where(
        np.isnan(ds.pm10),
        1,
        xr.where(
            (ds.pm10 < 3) | (ds.pm10 > 2000), 2, xr.where(ds.pm25 > ds.pm10, 3, 0)
        ),
    )
    ds["co_flag"] = xr.where(
        np.isnan(ds.co), 1, xr.where((ds.co < 10) | (ds.co > 4000), 2, 0)
    )
    ds["no2_flag"] = xr.where(
        np.isnan(ds.no2), 1, xr.where((ds.no2 < 2) | (ds.no2 > 100), 2, 0)
    )
    ds["so2_flag"] = xr.where(
        np.isnan(ds.so2), 1, xr.where((ds.so2 < 2) | (ds.so2 > 200), 2, 0)
    )
    ds["o3_flag"] = xr.where(
        np.isnan(ds.o3), 1, xr.where((ds.o3 < 2) | (ds.o3 > 200), 2, 0)
    )

    for flag_var in [
        "pm25_flag",
        "pm10_flag",
        "co_flag",
        "no2_flag",
        "so2_flag",
        "o3_flag",
    ]:
        ds[flag_var].encoding["_FillValue"] = 1

    ds.attrs["history"] = "flagged"
    return ds


def qa_pa(ds: xr.Dataset, dt: float = 40) -> xr.Dataset:
    """Apply quality assurance flags to PurpleAir monitor data.

    Flag values for channel A and B:
        0 — valid data
        1 — missing (NaN)
        2 — outside limit of detection (< 5 or > 500 µg/m³)
        3 — channel disagreement exceeds dt threshold

    Flag values for RH:
        0 — valid data
        1 — missing (NaN)
        2 — outside physical range (< 0 or > 100 %)

    Args:
        ds: PurpleAir dataset containing 'a', 'b', 'rh', and 'disagreement' variables.
        dt: Maximum allowable channel disagreement [%].

    Returns:
        Dataset with channel and RH flag variables added.
    """
    ds["a_flag"] = xr.where(
        np.isnan(ds.a),
        1,
        xr.where((ds.a < 5) | (ds.a > 500), 2, xr.where(ds.disagreement > dt, 3, 0)),
    )
    ds["b_flag"] = xr.where(
        np.isnan(ds.b),
        1,
        xr.where((ds.b < 5) | (ds.b > 500), 2, xr.where(ds.disagreement > dt, 3, 0)),
    )
    ds["rh_flag"] = xr.where(
        np.isnan(ds.rh), 1, xr.where((ds.rh < 0) | (ds.rh > 100), 2, 0)
    )

    for flag_var in ["a_flag", "b_flag", "rh_flag"]:
        ds[flag_var].encoding["_FillValue"] = 1

    ds.attrs["history"] = "flagged"
    return ds


def qa_ghsl(ds: xr.Dataset) -> xr.Dataset:
    """Apply quality assurance flags to GHSL population density data.

    Flag values:
        0 — valid data
        1 — missing (NaN)
        2 — population density is zero

    Args:
        ds: GHSL dataset containing a 'population' variable.

    Returns:
        Dataset with 'population_flag' added.
    """
    ds["population_flag"] = xr.where(
        np.isnan(ds.population), 1, xr.where(ds.population == 0, 2, 0)
    )
    ds.population.encoding["_FillValue"] = 1
    ds.attrs["history"] = "flagged"
    return ds


def qa_tropomi(ds: xr.Dataset) -> xr.Dataset:
    """Apply quality assurance flags to TROPOMI NO2 column data.

    Flag values:
        0 — valid data
        1 — missing (NaN)
        2 — NO2 column value ≤ 0

    Args:
        ds: TROPOMI dataset containing a 'no2_column' variable.

    Returns:
        Dataset with 'no2_column_flag' added.
    """
    ds["no2_column_flag"] = xr.where(
        np.isnan(ds.no2_column), 1, xr.where(ds.no2_column <= 0, 2, 0)
    )
    ds["no2_column_flag"].encoding["_FillValue"] = 1
    ds.attrs["history"] = "flagged"
    return ds


def qa_quantaq(ds: xr.Dataset) -> xr.Dataset:
    """Apply quality assurance flags to QuantAQ PM data.

    Flag values:
        0 — valid data
        2 — QuantAQ internal flag active (flag_qaq > 0)

    Args:
        ds: QuantAQ dataset containing a 'flag_qaq' variable.

    Returns:
        Dataset with 'quantaq_flag' added and 'flag_qaq' dropped.
    """
    ds["quantaq_flag"] = xr.where(ds.flag_qaq > 0, 2, 0)
    ds = ds.drop("flag_qaq")
    ds["quantaq_flag"].encoding["_FillValue"] = 1
    ds.attrs["history"] = "flagged"
    return ds
