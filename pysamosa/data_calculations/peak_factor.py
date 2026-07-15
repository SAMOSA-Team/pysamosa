"""Peak factor and baseline estimation pipeline for PurpleAir sensor data."""

import warnings
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

warnings.filterwarnings("ignore")


def calculate_daily_baseline(df, col, quantile=0.1, dof=6):
    """Fit a cubic regression spline baseline at a given quantile.

    Args:
        df: DataFrame with a datetime index and a measurement column.
        col: Column name for the target measurement.
        quantile: Target quantile for the baseline regression (e.g. 0.1 = 10th percentile).
        dof: Degrees of freedom for the cubic regression spline.

    Returns:
        DataFrame with 'baseline' and 'peak' columns, or None if fitting fails.
    """
    # Check if we have enough data points
    if df is None or len(df) < dof + 2 or len(df.index.hour.unique()) < 3:
        return None

    try:
        # Convert to Julian dates as a numpy array
        julian_dates = df.index.to_julian_date().values

        # Filter out NaN values
        valid_idx = ~np.isnan(julian_dates) & ~df[col].isna().values
        if np.sum(valid_idx) < dof + 2:
            return None

        # Create a clean dataset for modeling
        X_clean = julian_dates[valid_idx].reshape(-1, 1)
        y_clean = df.loc[valid_idx, col].values

        # Manual implementation of cubic regression spline
        from scipy.interpolate import LSQUnivariateSpline

        # Create knots at regular intervals
        num_knots = max(0, dof - 1)  # dof internal knots for cubic spline
        if num_knots <= 0:
            return None  # Not enough dof for proper splines

        # Create internal knots at regular intervals
        knots = np.linspace(
            np.min(X_clean) + 0.1 * (np.max(X_clean) - np.min(X_clean)),
            np.max(X_clean) - 0.1 * (np.max(X_clean) - np.min(X_clean)),
            num_knots,
        )

        # Direct spline approach for baseline
        spline = LSQUnivariateSpline(
            X_clean.ravel(), y_clean, knots, k=3, check_finite=True  # cubic
        )

        # Get baseline prediction for all valid points
        baselines = spline(julian_dates[valid_idx])

        # Find the quantile offset to convert to quantile regression
        residuals = y_clean - baselines
        quantile_offset = np.percentile(residuals, quantile * 100)

        # Apply offset to get the quantile regression equivalent
        adjusted_baselines = baselines + quantile_offset

        # Apply non-negativity constraint
        adjusted_baselines = np.maximum(0, adjusted_baselines)

        # Create result DataFrame
        result = pd.DataFrame(index=df.index, columns=["baseline", "peak"])

        # Set values for valid indices
        result.loc[df.index[valid_idx], "baseline"] = adjusted_baselines
        result.loc[df.index[valid_idx], "peak"] = np.maximum(
            df.loc[df.index[valid_idx], col].values - adjusted_baselines, 0
        )

        return result

    except Exception as e:
        print(f"Error in calculate_daily_baseline: {e}")
        return None


def _process_sensor_chunk(args):
    """Process a chunk of data for a single sensor and date."""
    sensor, date, data_dict, quantile, dof = args

    try:
        # Reconstruct DataFrame from dictionary
        df = pd.DataFrame(data_dict)
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)

        # Check if we have enough data
        if len(df) <= 30:
            return None

        # Calculate baseline
        result = calculate_daily_baseline(
            df, "pa_campmier_delhi_mean", quantile=quantile, dof=dof
        )

        if result is None or result["baseline"].isna().all():
            return None

        result["sensor"] = sensor
        return result

    except Exception as e:
        print(f"Error processing sensor {sensor}, date {date}: {e}")
        return None


def baseline_pipeline(ds, quantile=0.1, dof=6, n_processes=None):
    """Run the parallel baseline estimation pipeline across all sensors.

    Args:
        ds: Dataset containing 'pa_campmier_delhi_mean' with a 'sensor' dimension.
        quantile: Baseline quantile for cubic spline regression.
        dof: Degrees of freedom for the spline.
        n_processes: Number of worker processes; defaults to cpu_count - 1.

    Returns:
        Dataset with 'baseline' and 'peak' variables indexed by sensor and time.
    """

    # Set number of processes
    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)  # Leave one CPU free

    print(f"Starting baseline pipeline with {n_processes} processes...")

    # Load data
    df_pa = ds["pa_campmier_delhi_mean"].to_dataframe().reset_index()
    df_pa["date"] = df_pa["time"].dt.date

    # Create chunks of work (sensor-date combinations)
    chunks = []
    for sensor in df_pa["sensor"].unique():
        sensor_data = df_pa[df_pa["sensor"] == sensor]
        for date in sensor_data["date"].unique():
            date_data = sensor_data[sensor_data["date"] == date]
            if len(date_data) > 30:  # Only process if enough data
                # Convert to dictionary for serialization
                data_dict = {
                    "time": date_data["time"].astype(str).tolist(),
                    "pa_campmier_delhi_mean": date_data[
                        "pa_campmier_delhi_mean"
                    ].tolist(),
                }
                chunks.append((sensor, date, data_dict, quantile, dof))

    print(f"Created {len(chunks)} chunks to process")

    # Process chunks in parallel with progress bar
    lst_daily = []

    # Use imap instead of map for better memory efficiency
    with Pool(n_processes) as pool:
        # Process in batches to avoid overwhelming the system
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            results = list(
                tqdm(
                    pool.imap(_process_sensor_chunk, batch),
                    total=len(batch),
                    desc=f"Processing batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1}",
                )
            )

            # Collect non-None results
            lst_daily.extend([r for r in results if r is not None])

            # Print progress
            print(f"Completed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")

    if not lst_daily:
        print("Warning: No valid results from processing")
        return xr.Dataset()

    # Combine results
    print(f"Combining {len(lst_daily)} results...")
    df_daily = pd.concat(lst_daily, ignore_index=False)

    # Convert to xarray
    ds_peak = df_daily.reset_index().set_index(["sensor", "time"]).to_xarray()

    print("Baseline pipeline completed")
    return ds_peak


# Alternative simpler version without multiprocessing for debugging
def baseline_pipeline_serial(ds, quantile=0.1, dof=6):
    """Serial baseline pipeline (single-process, for debugging)."""
    print("Running serial baseline pipeline...")

    # Load data
    df_pa = ds["pa_campmier_delhi_mean"].to_dataframe().reset_index()
    df_pa["date"] = df_pa["time"].dt.date
    df_pa.set_index("time", inplace=True)

    lst_daily = []

    # Process each sensor
    for sensor in tqdm(df_pa["sensor"].unique(), desc="Processing sensors"):
        sensor_data = df_pa[df_pa["sensor"] == sensor]

        # Process each date
        for date in sensor_data["date"].unique():
            date_data = sensor_data[sensor_data["date"] == date]
            date_data = date_data.dropna(subset=["pa_campmier_delhi_mean"])

            if len(date_data) > 30:
                result = calculate_daily_baseline(
                    date_data, "pa_campmier_delhi_mean", quantile=quantile, dof=dof
                )

                if result is not None and not result["baseline"].isna().all():
                    result["sensor"] = sensor
                    lst_daily.append(result)

    if not lst_daily:
        print("Warning: No valid results from processing")
        return xr.Dataset()

    # Combine results
    df_daily = pd.concat(lst_daily)
    ds_peak = df_daily.reset_index().set_index(["sensor", "time"]).to_xarray()

    return ds_peak


def sensitivity_pipeline(in_path, n_processes=None, sensor_list=None):

    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)

    # Load data once
    ds = xr.open_dataset(in_path)

    if sensor_list is not None:
        ds = ds.sel(sensor=sensor_list)

    df_pa = ds["pa_raw_mean"].to_dataframe().reset_index()
    df_pa["date"] = df_pa["time"].dt.date
    df_pa.set_index("time", inplace=True)

    # Parameter combinations
    lst_quantiles = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.30]
    lst_dof = [
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
    ]

    # Create all work combinations with serializable data
    work_items = []
    for sensor, sensor_data in df_pa.groupby("sensor"):
        # Convert DataFrame to dictionary for serialization
        sensor_dict = {
            "time": sensor_data.index.astype(str).tolist(),
            "pa_raw_mean": sensor_data["pa_raw_mean"].tolist(),
            "sensor": sensor_data["sensor"].tolist(),
            "date": [d.strftime("%Y-%m-%d") for d in sensor_data["date"]],
        }

        unique_dates = pd.to_datetime(sensor_data["date"].unique())
        for date in unique_dates:
            for quantile in lst_quantiles:
                for dof in lst_dof:
                    work_items.append((sensor_dict, date, quantile, dof))

    # Process all combinations in a single pool
    results_by_params = {}
    with Pool(n_processes) as pool:
        for result in tqdm(
            pool.imap(process_single_combination, work_items),  # noqa: F821
            total=len(work_items),
            desc=f"Processing all combinations using {n_processes} processes",
        ):
            if result is not None:
                key = (result["quantile"], result["dof"])
                if key not in results_by_params:
                    results_by_params[key] = []

                # Reconstruct DataFrame from dictionary
                if "index" in result["result"] and result["result"]["index"]:
                    # Convert string time back to datetime
                    time_index = pd.to_datetime(result["result"]["index"])

                    df = pd.DataFrame(
                        {
                            "baseline": result["result"]["baseline"],
                            "peak": result["result"]["peak"],
                            "time": time_index,  # Add time as a column
                            "sensor": result["result"]["sensor"],
                        }
                    )

                    results_by_params[key].append(df)

        datasets = []
        for (quantile, dof), result_list in results_by_params.items():
            if result_list:
                df_combined = pd.concat(result_list, copy=False)

                # Create a dataset with both baseline and peak variables
                # But rename baseline to be unique for each parameter combination
                df_with_time = (
                    df_combined.reset_index()
                    if "time" not in df_combined.columns
                    else df_combined
                )

                # Set multi-index
                df_indexed = df_with_time.set_index(["sensor", "time"])

                # Convert to xarray dataset
                ds_peak = df_indexed.to_xarray()

                # Rename both variables to include the parameters
                ds_peak = ds_peak.rename(
                    {
                        "baseline": f"baseline_{quantile}_{dof}",
                        "peak": f"peak_{quantile}_{dof}",
                    }
                )

                datasets.append(ds_peak)

        # Now merge will work without conflicts
        return xr.merge(datasets)


def run_sensitivity_analysis(in_path, n_processes=None, sensor_list=None):
    return sensitivity_pipeline(
        in_path, n_processes=n_processes, sensor_list=sensor_list
    )
