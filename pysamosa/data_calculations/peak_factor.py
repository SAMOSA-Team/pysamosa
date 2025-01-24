import pandas as pd
import xarray as xr
import numpy as np
from patsy import dmatrix
from sklearn.linear_model import QuantileRegressor
from tqdm import tqdm
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial

warnings.filterwarnings("ignore")


def calculate_daily_baseline(df, col, quantile=0.1, dof=6):
    if len(df.index.hour.unique()) < 3:
        return pd.DataFrame(index=df.index, columns=["baseline", "peak"])

    transformed_x3 = dmatrix(  # noqa: E225
        f"cr(train,df = {dof})",  # noqa: E225, E231
        {"train": df.index.to_julian_date().values},  # noqa: E225
        return_type="dataframe",  # noqa: E225
    )  # noqa: E225

    qr = QuantileRegressor(quantile=quantile, solver="highs-ds", alpha=0)
    try:
        baseline = qr.fit(transformed_x3, df[col].values.reshape(-1, 1)).predict(
            transformed_x3
        )
        baseline = np.maximum(baseline, 0)
        peak = np.maximum(df[col] - baseline, 0)

        return pd.DataFrame({"baseline": baseline, "peak": peak}, index=df.index)
    except TypeError:
        return pd.DataFrame(index=df.index, columns=["baseline", "peak"])


def process_single_combination(args):
    data_dict, date, quantile, dof = args

    # Reconstruct DataFrame from dict
    sensor_data = pd.DataFrame.from_dict(data_dict)
    sensor_data["time"] = pd.to_datetime(sensor_data["time"])
    sensor_data.set_index("time", inplace=True)

    # Filter for the specific date
    df_ = sensor_data[sensor_data["date"] == date].dropna()

    if len(df_) > 30:
        df = calculate_daily_baseline(
            df_, "pa_campmier_delhi_mean", quantile=quantile, dof=dof
        )
        df["sensor"] = sensor_data["sensor"].iloc[0]

        # Convert the result to a dictionary for serialization
        result_dict = {
            "index": df.index.astype(str).tolist(),
            "baseline": df["baseline"].tolist(),
            "peak": df["peak"].tolist(),
            "sensor": df["sensor"],
        }

        return {"quantile": quantile, "dof": dof, "result": result_dict}
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
    lst_dof = [2, 3, 4, 5, 6, 7, 8, 9, 10]

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
                df = pd.DataFrame(
                    {
                        "baseline": result["result"]["baseline"],
                        "peak": result["result"]["peak"],
                    },
                    index=pd.to_datetime(result["result"]["index"]),
                )
                df["sensor"] = result["result"]["sensor"]

                results_by_params[key].append(df)

    # Combine results for each parameter combination
    datasets = []
    for (quantile, dof), result_list in results_by_params.items():
        if result_list:
            df_combined = pd.concat(result_list, copy=False)
            ds_peak = (
                df_combined.reset_index().set_index(["sensor", "time"]).to_xarray()
            )
            ds_peak = ds_peak.rename({"peak": f"peak_{quantile}_{dof}"})
            datasets.append(ds_peak)

    return xr.merge(datasets)


def run_sensitivity_analysis(in_path, n_processes=None, sensor_list=None):
    return sensitivity_pipeline(
        in_path, n_processes=n_processes, sensor_list=sensor_list
    )
