import pandas as pd
import xarray as xr
import numpy as np

# from patsy import dmatrix
from tqdm import tqdm
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial

warnings.filterwarnings("ignore")


def calculate_daily_baseline(df, col, quantile=0.1, dof=6):
    """Calculate daily baseline using cubic regression splines.

    Args:
        df: DataFrame with time index and measurement column
        col: Column name for measurements
        quantile: Quantile for regression (e.g., 0.1 for 10th percentile)
        dof: Degrees of freedom for the cubic regression spline

    Returns:
        DataFrame with baseline and peak columns or None if calculation fails
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


def process_single_combination(args):
    data_dict, date, quantile, dof = args

    try:
        # Reconstruct DataFrame from dict
        if not data_dict or "time" not in data_dict or "pa_raw_mean" not in data_dict:
            print(f"Invalid data dictionary for date {date}")
            return None

        sensor_data = pd.DataFrame.from_dict(data_dict)

        # Check if we have data
        if sensor_data.empty:
            print(f"Empty DataFrame for date {date}")
            return None

        sensor_data["time"] = pd.to_datetime(sensor_data["time"])
        sensor_data.set_index("time", inplace=True)

        # Filter for the specific date
        date_str = date.strftime("%Y-%m-%d")
        df_ = sensor_data[sensor_data["date"] == date_str]

        # Drop NaN values
        df_ = df_.dropna(subset=["pa_raw_mean"])

        # Check if we have enough data points
        if len(df_) <= 30:
            print(f"Not enough data points for date {date_str}: {len(df_)}")
            return None

        # Print debug info
        print(f"Processing {date_str} with {len(df_)} points, q={quantile}, dof={dof}")

        # Calculate baseline and peaks
        df = calculate_daily_baseline(df_, "pa_raw_mean", quantile=quantile, dof=dof)

        # Check if we got valid results
        if df is None or df.empty or df["baseline"].isna().all():
            print(f"No valid results for {date_str}")
            return None

        # Get sensor ID (should be the same for all rows)
        if "sensor" in sensor_data.columns and not sensor_data["sensor"].empty:
            sensor_val = str(sensor_data["sensor"].iloc[0])
        else:
            print(f"No sensor information for {date_str}")
            sensor_val = "unknown"

        # Convert to serializable format
        result_dict = {
            "index": [str(idx) for idx in df.index],
            "baseline": [
                float(val) if not pd.isna(val) else None for val in df["baseline"]
            ],
            "peak": [float(val) if not pd.isna(val) else None for val in df["peak"]],
            "sensor": sensor_val,
        }

        return {"quantile": float(quantile), "dof": int(dof), "result": result_dict}
    except Exception as e:
        date_str = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)
        print(
            f"Error processing combination (q={quantile}, dof={dof}, date={date_str}): {e}"
        )
        return None


def process_sensor(sensor_data, unique_dates, quantile=0.1, dof=6):
    """Process a single sensor's data"""
    sensor = sensor_data["sensor"].iloc[0]
    lst_daily = []

    for date in unique_dates:
        df_ = sensor_data[sensor_data["date"] == date].dropna()

        if len(df_) > 30:
            df = calculate_daily_baseline(
                df_, "pa_campmier_delhi_mean", quantile=quantile, dof=dof
            )
            df["sensor"] = sensor
            lst_daily.append(df)

    return lst_daily if lst_daily else None


def baseline_pipeline(ds, quantile=0.1, dof=6, n_processes=None):
    """Parallel processing pipeline with reduced memory usage and faster operations"""
    # Set number of processes
    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)  # Leave one CPU free

    # Load data
    df_pa = ds["pa_campmier_delhi_mean"].to_dataframe().reset_index()
    df_pa["date"] = df_pa["time"].dt.date
    df_pa.set_index("time", inplace=True)

    # Pre-calculate unique values
    unique_dates = df_pa["date"].unique()

    # Group data by sensor
    grouped = df_pa.groupby("sensor")
    sensor_groups = [group for _, group in grouped]

    # Create partial function with fixed parameters
    process_sensor_partial = partial(
        process_sensor, unique_dates=unique_dates, quantile=quantile, dof=dof
    )

    # Process in parallel with progress bar
    with Pool(n_processes) as pool:
        results = list(
            tqdm(
                pool.imap(process_sensor_partial, sensor_groups),
                total=len(sensor_groups),
                desc=f"Processing sensors using {n_processes} processes",
            )
        )

    # Flatten results and remove None values
    lst_daily = [item for sublist in results if sublist is not None for item in sublist]

    # Combine results
    df_daily = pd.concat(lst_daily, copy=False)
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
            pool.imap(process_single_combination, work_items),
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
