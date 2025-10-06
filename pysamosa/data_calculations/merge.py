"""
Merge
Last Updated: Aug 28, 2025
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
import gc

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
    index = pd.Series(pd.date_range(start, end, freq="1h"), name="time")
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

    return xr.concat(
        cells, dim=keys, coords="minimal", compat="override"
    )  # xr.combine_nested(cells, concat_dim=keys)


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

    if type(keys) is dict:
        ds_matched_merge = ds_merge.sel(
            {keys["merge"]: df_matched.collocation_site.values}
        )
        ds_matched_points = ds_points.sel({keys["points"]: df_matched.index.values})
    else:
        ds_matched_merge = ds_merge.sel({keys: df_matched.collocation_site.values})
        ds_matched_points = ds_points.sel({keys: df_matched.index.values})

    return ds_matched_points, ds_matched_merge


def get_chunk_spec(path, reference_chunks=None):
    """
    Get appropriate chunk specification for a dataset, with option to match reference chunks.

    :param path: path to dataset
    :param reference_chunks: optional reference chunks to match
    :return: chunk specification or None
    """
    try:
        with xr.open_dataset(path, engine="h5netcdf") as ds:
            # Check if any data variables have object dtype
            has_object_dtype = any(
                ds[var].dtype == object or ds[var].dtype.kind in ["U", "S", "O"]
                for var in ds.data_vars
            )

            # If there are object dtypes, don't chunk at all
            if has_object_dtype:
                return None

            # If reference chunks provided, try to match them
            if reference_chunks is not None:
                matching_dims = {
                    k: v for k, v in reference_chunks.items() if k in ds.dims
                }
                if matching_dims:
                    return matching_dims

            # Otherwise, chunk based on available dimensions
            if "time" in ds.dims:
                return {"time": 1000}
            elif "site" in ds.dims:
                return {"site": 100}
            elif "position" in ds.dims:
                return {"position": 100}
            else:
                return None

    except (OSError, ValueError, ImportError):
        return None


def open_and_merge(in_path, file_list):
    """
    Open and merge all datasets with consistent chunking.

    :param in_path: path to flagged data
    :param file_list: list of file names (without .nc extension)
    :return: merged dataset
    :rtype: xarray.Dataset
    """
    import gc

    if not file_list:
        return xr.Dataset()

    print(f"Merging {len(file_list)} datasets...")

    def open_dataset_safely(path, chunks=None):
        """Helper to open datasets with appropriate engine."""
        try:
            if chunks is not None:
                return xr.open_dataset(path, engine="h5netcdf", chunks=chunks)
            else:
                return xr.open_dataset(path, engine="h5netcdf")
        except (ImportError, OSError, ValueError):
            if chunks is not None:
                return xr.open_dataset(path, chunks=chunks)
            else:
                return xr.open_dataset(path)

    # Get chunk spec for first dataset - this will be our reference
    first_path = os.path.join(in_path, f"{file_list[0]}.nc")
    reference_chunks = get_chunk_spec(first_path)

    # Start with the first dataset
    result = open_dataset_safely(first_path, reference_chunks)

    # If first dataset is chunked, rechunk it to consistent sizes to avoid the warning
    if reference_chunks and hasattr(result, "chunks"):
        result = result.chunk(reference_chunks)

    # Incrementally merge remaining datasets
    for i, file in enumerate(file_list[1:], 1):
        print(f"Merging dataset {i + 1}/{len(file_list)}: {file}")

        # Get chunk spec matching the reference
        next_path = os.path.join(in_path, f"{file}.nc")
        chunks = get_chunk_spec(next_path, reference_chunks=reference_chunks)

        # Open next dataset
        next_ds = open_dataset_safely(next_path, chunks)

        # If dataset is chunked, ensure consistent chunking before merge
        if chunks and hasattr(next_ds, "chunks"):
            # Rechunk to match reference if needed
            next_ds = next_ds.chunk(reference_chunks if reference_chunks else chunks)

        # Merge with existing result
        result = xr.merge([result, next_ds])

        # Clean up
        del next_ds
        gc.collect()

    return result


def open_and_merge_raster(in_path, ds_ref, raster_list, xy=None, keys="site"):
    """
    Open a list of files and merge all raster cells using incremental processing.

    :param in_path: path to flagged data
    :type in_path: str
    :param ds_ref: reference dataset
    :type ds_ref: xarray.Dataset
    :param raster_list: list of raster file names
    :type raster_list: list
    :param xy: The x and y coordinates to merge for single cell merges.
    :type xy: list
    :param keys: The key to use for the point data merge, 'sites' by default.
    :type keys: str
    :return: merged dataset
    :rtype: xarray.Dataset
    """
    import gc

    if not raster_list:
        return xr.Dataset()

    print(f"Merging {len(raster_list)} raster datasets...")

    # Process first raster
    first_path = os.path.join(in_path, f"{raster_list[0]}.nc")
    chunks = get_chunk_spec(first_path)

    if chunks is not None:
        first_raster = xr.open_dataset(first_path, chunks=chunks)
    else:
        first_raster = xr.open_dataset(first_path)

    result = fixed_raster_merge(ds_ref, first_raster, xy=xy, keys=keys)
    del first_raster
    gc.collect()

    # Incrementally merge remaining rasters
    for i, raster in enumerate(raster_list[1:], 1):
        print(f"Processing raster {i + 1}/{len(raster_list)}: {raster}")

        # Get chunk spec for this raster
        raster_path = os.path.join(in_path, f"{raster}.nc")
        chunks = get_chunk_spec(raster_path)

        # Open next raster with appropriate chunks
        if chunks is not None:
            next_raster = xr.open_dataset(raster_path, chunks=chunks)
        else:
            next_raster = xr.open_dataset(raster_path)

        # Apply fixed_raster_merge
        merged_raster = fixed_raster_merge(ds_ref, next_raster, xy=xy, keys=keys)

        # Merge with existing result
        result = xr.merge([result, merged_raster])

        # Clean up
        del next_raster, merged_raster
        gc.collect()

    return result


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
    ds = (
        ds.where(ds.site == "IITD")
        .dropna(dim="position", how="all")
        .dropna(dim="time", how="all")
    )
    ds = ds.drop(
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
        ]
    )

    ds = ds.swap_dims({"position": "site"})
    ds = ds.set_index(position=["site", "sensor"])
    ds = ds.unstack("position")
    ds_season = india.get_season(
        ds["time"].to_dataframe().reset_index(drop=True).set_index("time"), False
    ).to_xarray()
    ds = xr.merge([ds, df_collocation_phase.to_xarray(), ds_season]).set_coords(
        ["collocation_phase", "season"]
    )
    return ds


def merge_reference(in_path):
    """
    Memory-efficient version with consistent chunking to avoid performance warnings.

    :param in_path: path to the data
    :return: merged reference dataset
    :rtype: xarray.Dataset
    """

    def dataset_exists(dataset_name):
        return os.path.exists(os.path.join(in_path, f"{dataset_name}.nc"))

    def open_dataset_safely(path, chunks=None):
        """Open dataset with appropriate engine and error handling."""
        try:
            if chunks is not None:
                ds = xr.open_dataset(path, engine="h5netcdf", chunks=chunks)
                # Ensure consistent chunking
                if hasattr(ds, "chunks"):
                    ds = ds.chunk(chunks)
                return ds
            else:
                return xr.open_dataset(path, engine="h5netcdf")
        except (OSError, ValueError, ImportError):
            if chunks is not None:
                ds = xr.open_dataset(path, chunks=chunks)
                if hasattr(ds, "chunks"):
                    ds = ds.chunk(chunks)
                return ds
            else:
                return xr.open_dataset(path)

    print("Starting reference merge...")

    # First merge BAM and REG datasets
    ds_reference = open_and_merge(in_path, ["bam", "reg"])

    # Get reference chunks from the merged dataset
    reference_chunks = None
    if hasattr(ds_reference, "chunks"):
        # Get chunks from first data variable
        for var in ds_reference.data_vars:
            # Check if this variable actually has chunks
            if (
                hasattr(ds_reference[var], "chunks")
                and ds_reference[var].chunks is not None
            ):
                var_chunks = dict(
                    zip(
                        ds_reference[var].dims, [c[0] for c in ds_reference[var].chunks]
                    )
                )
                reference_chunks = {
                    k: v
                    for k, v in var_chunks.items()
                    if k in ["time", "site", "position"]
                }
                break

    # If no chunks found in the reference, use default chunks
    if reference_chunks is None:
        reference_chunks = {"time": 1000} if "time" in ds_reference.dims else None

    # Define raster datasets to check
    datasets_to_merge = ["era", "martin", "ghsl"]  # "tropomi",

    # Check which datasets exist
    existing_datasets = [
        dataset for dataset in datasets_to_merge if dataset_exists(dataset)
    ]

    if existing_datasets:
        print(
            f"Found {len(existing_datasets)} raster datasets to merge: {existing_datasets}"
        )

        # Process them incrementally
        for i, raster_name in enumerate(existing_datasets):
            print(f"Merging raster {i + 1}/{len(existing_datasets)}: {raster_name}")

            # Get appropriate chunks for this raster, matching reference if possible
            raster_path = os.path.join(in_path, f"{raster_name}.nc")
            chunks = get_chunk_spec(raster_path, reference_chunks=reference_chunks)

            # Open raster with appropriate chunks and engine
            raster_ds = open_dataset_safely(raster_path, chunks)

            # Apply fixed_raster_merge
            merged_raster = fixed_raster_merge(ds_reference, raster_ds, keys="site")

            # Merge with reference
            ds_reference = xr.merge([ds_reference, merged_raster], compat="override")

            # Clean up immediately after each merge
            del raster_ds, merged_raster
            gc.collect()

        print("All rasters merged successfully")
        return ds_reference
    else:
        print("No raster datasets found")
        return ds_reference


def merge_phases(in_path, dict_phases):
    # ds_pa = xr.open_dataset(os.path.join(in_path, "pa.nc"))
    ds_pa = xr.open_dataset(os.path.join(in_path, "pr.nc"), chunks={"time": 1000})
    ds_phases = make_phases(ds_pa, dict_phases)
    """
    ds_bam = fixed_point_merge(
        ds_phases,
        xr.open_dataset(os.path.join(in_path, "bam.nc")),
        xy=[28.5468, 77.1906]
    )
    """
    return ds_phases  # xr.merge([ds_phases, ds_bam])


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
        in_path, ds_collocation, ["era", "martin", "ghsl"], keys="site"
    )  # "tropomi",
    return xr.merge([ds_collocation, ds_collocation_ref, ds_rasters], compat="override")


def make_deployment(ds):
    """
    Makes campaign deployment dataset with proper handling of chunked data.
    """
    # For chunked data, compute just the site coordinate to filter
    site_values = ds.site.compute() if hasattr(ds.site, "compute") else ds.site

    # Filter out IITD site using integer indexing
    if "position" in ds.dims:
        keep_positions = np.where(site_values.values != "IITD")[0]
        ds = ds.isel(position=keep_positions)
        # Update site_values after filtering
        site_values = ds.site.compute() if hasattr(ds.site, "compute") else ds.site
    else:
        sites_to_keep = [s for s in site_values.values if s != "IITD"]
        ds = ds.sel(site=sites_to_keep)

    # Calculate raw mean and pre-allocate for std
    ds["pa_raw_mean"] = (ds.a + ds.b) * 0.5
    ds["pa_raw_std"] = (ds.a + ds.b) * 0.5  # Pre-allocation for memory efficiency

    # Resample to hourly
    ds_hourly = xr.Dataset(
        {
            "pa_raw_mean": ds.pa_raw_mean.resample(time="1h").mean(),
            "pa_raw_std": ds.pa_raw_std.resample(time="1h").std(),
            "rh": ds.rh.resample(time="1h").mean(),
            "disagreement": ds.disagreement.resample(time="1h").mean(),
        }
    )

    # Process groupby - handle chunked data efficiently
    if "position" in ds_hourly.dims:
        # Process in time chunks to manage memory
        chunk_size = 5000
        n_times = len(ds_hourly.time)

        results = []
        for start_idx in range(0, n_times, chunk_size):
            end_idx = min(start_idx + chunk_size, n_times)
            print(f"  Processing time chunk {start_idx}-{end_idx} of {n_times}")

            # Process this time chunk
            ds_chunk = ds_hourly.isel(time=slice(start_idx, end_idx))
            ds_chunk = ds_chunk.compute()

            # Convert to pandas for groupby
            df = ds_chunk.to_dataframe().reset_index()
            if "position" in df.columns:
                df = df.drop("position", axis=1)

            # Group by site and time, taking first value
            df_grouped = df.groupby(["site", "time"]).agg(
                {
                    "pa_raw_mean": "first",
                    "pa_raw_std": "first",
                    "rh": "first",
                    "disagreement": "first",
                }
            )

            results.append(df_grouped)

            del ds_chunk, df, df_grouped
            gc.collect()

        # Combine all chunks
        print("  Combining time chunks...")
        df_combined = pd.concat(results)
        ds_grouped = df_combined.to_xarray()

        del results, df_combined
        gc.collect()

        # Add metadata coordinates
        print("  Adding metadata coordinates...")
        meta_coords = [
            "land_uses",
            "settlement",
            "district",
            "state",
            "settlement_type",
            "cluster",
            "latitude",
            "longitude",
            "sensor",
        ]

        # Create a mapping from site to position index in the FILTERED dataset
        site_to_position = {}
        for pos_idx, site in enumerate(site_values.values):
            if site not in site_to_position:
                site_to_position[site] = pos_idx

        for coord in meta_coords:
            if coord in ds.coords:
                coord_data = (
                    ds[coord].compute() if hasattr(ds[coord], "compute") else ds[coord]
                )

                if "position" in coord_data.dims:
                    # Get first value per site using the correct position indices
                    unique_sites = np.unique(ds_grouped.site.values)
                    coord_values = []

                    for site in unique_sites:
                        if site in site_to_position:
                            # Use the position index from the FILTERED dataset
                            pos_idx = site_to_position[site]
                            coord_values.append(
                                coord_data.isel(position=pos_idx).values
                            )
                        else:
                            coord_values.append(np.nan)

                    if coord_values and not all(
                        np.isnan(v) if np.isscalar(v) else False for v in coord_values
                    ):
                        ds_grouped = ds_grouped.assign_coords(
                            {coord: ("site", coord_values)}
                        )
    else:
        # No position dimension, simpler case
        ds_grouped = ds_hourly

    # Set coordinates
    coord_list = [
        "land_uses",
        "settlement",
        "district",
        "state",
        "settlement_type",
        "cluster",
        "latitude",
        "longitude",
    ]
    existing_coords = [c for c in coord_list if c in ds_grouped.coords]
    if existing_coords:
        ds_grouped = ds_grouped.set_coords(existing_coords)

    # Drop is_collocation_site if present
    if "is_collocation_site" in ds_grouped:
        ds_grouped = ds_grouped.drop_vars("is_collocation_site")

    # Add season
    print("  Adding season coordinate...")
    time_values = (
        ds_grouped.time.compute()
        if hasattr(ds_grouped.time, "compute")
        else ds_grouped.time
    )
    season_df = india.get_season(
        pd.DataFrame(index=time_values.values).rename_axis("time"), False
    )
    ds_grouped = xr.merge([ds_grouped, season_df.to_xarray()]).set_coords("season")

    return ds_grouped


def merge_deployment(in_path):
    """
    Memory-efficient deployment merge with proper string handling for NetCDF export.
    """
    import os
    import gc

    def dataset_exists(dataset_name):
        return os.path.exists(os.path.join(in_path, f"{dataset_name}.nc"))

    # Open with chunks
    print("Loading PurpleAir data...")
    ds = xr.open_dataset(os.path.join(in_path, "pr.nc"), chunks={"time": 1000})

    print("Processing deployment dataset...")
    ds = make_deployment(ds)

    # Define datasets to merge
    datasets_to_merge = ["era", "martin", "ghsl", "rwi"]  # "tropomi", "gee"

    # Check which datasets exist
    existing_datasets = [
        dataset for dataset in datasets_to_merge if dataset_exists(dataset)
    ]

    if existing_datasets:
        print(f"Merging raster datasets: {existing_datasets}")

        # Process rasters incrementally
        for i, raster_name in enumerate(existing_datasets):
            print(f"Merging {raster_name} ({i + 1}/{len(existing_datasets)})")

            raster_path = os.path.join(in_path, f"{raster_name}.nc")
            chunks = get_chunk_spec(raster_path)

            if chunks is not None:
                raster_ds = xr.open_dataset(raster_path, chunks=chunks)
            else:
                raster_ds = xr.open_dataset(raster_path)

            # Apply fixed_raster_merge
            merged_raster = fixed_raster_merge(ds, raster_ds, keys="site")

            # Merge with main dataset
            ds = xr.merge([ds, merged_raster], compat="override")

            del raster_ds, merged_raster
            gc.collect()

    # Fix string encoding for NetCDF export
    print("Preparing for NetCDF export...")
    for var in ds.data_vars:
        if ds[var].dtype == object:
            # Convert object arrays to string arrays
            ds[var] = ds[var].astype(str)

    for coord in ds.coords:
        if ds[coord].dtype == object:
            # Convert coordinate object arrays to string arrays
            ds[coord] = ds[coord].astype(str)

    print("Deployment dataset complete")
    return ds


def make_history(ds):
    """
    Memory-efficient version using xarray operations without pandas conversion.
    """

    # Drop coordinates more efficiently
    drop_vars = [
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

    # Drop variables that exist
    existing_drops = [v for v in drop_vars if v in ds.coords or v in ds.data_vars]
    ds_history = ds.drop_vars(existing_drops)

    # Instead of converting to dataframe, work directly with xarray
    # Stack position dimension to sensor dimension if needed
    if "position" in ds_history.dims:
        # Get sensor values for each position - compute this since it's small
        sensor_da = ds_history[
            "sensor"
        ].compute()  # Compute sensor array since it's small

        # Create a new dataset indexed by sensor and time
        unique_sensors = np.unique(sensor_da.values)

        print(f"Processing {len(unique_sensors)} unique sensors...")

        # Build new dataset sensor by sensor (memory efficient)
        new_data_vars = {}
        for var in ["a", "b", "rh", "disagreement"]:
            if var in ds_history.data_vars:
                print(f"Processing {var}...")

                # Create list to hold data for each sensor
                sensor_arrays = []

                for sensor in unique_sensors:
                    # Find positions for this sensor (sensor_da is computed, so this is a numpy array)
                    position_indices = np.where(sensor_da.values == sensor)[0]

                    if len(position_indices) > 0:
                        # Select data for these positions using integer indexing (works with dask)
                        sensor_data = ds_history[var].isel(position=position_indices)

                        # If sensor appears in multiple positions, combine them
                        if len(position_indices) > 1:
                            # Take max across positions (assumes non-overlapping valid data)
                            sensor_data = sensor_data.max(dim="position", skipna=True)
                        else:
                            sensor_data = sensor_data.squeeze("position")

                        # IMPORTANT: Drop the position coordinate to avoid conflicts
                        if "position" in sensor_data.coords:
                            sensor_data = sensor_data.drop_vars("position")

                        # Also drop the sensor coordinate if it exists (we'll add it back with concat)
                        if "sensor" in sensor_data.coords:
                            sensor_data = sensor_data.drop_vars("sensor")

                        sensor_arrays.append(sensor_data)

                # Stack all sensors
                if sensor_arrays:
                    # Create sensor coordinate
                    sensor_coord = pd.Index(unique_sensors, name="sensor")
                    # Use compat='override' to handle any remaining coordinate conflicts
                    combined = xr.concat(
                        sensor_arrays,
                        dim=sensor_coord,
                        coords="minimal",
                        compat="override",
                    )
                    new_data_vars[var] = combined

                del sensor_arrays
                gc.collect()

        # Create new dataset
        ds_history = xr.Dataset(new_data_vars)
    else:
        # If no position dimension, just use the dataset as is
        print("No position dimension found, using dataset as is")
        ds_history = ds_history

    # Remove all-NaN time steps (compute just the mask to avoid loading all data)
    print("Removing all-NaN time steps...")
    time_has_data = ds_history.to_array().notnull().any(dim=["variable", "sensor"])
    time_has_data = time_has_data.compute()  # Compute the boolean mask
    ds_history = ds_history.sel(time=time_has_data)

    # Add season
    print("Adding season...")
    # Get time values (compute if needed)
    time_values = ds_history.time.values
    if hasattr(time_values, "compute"):
        time_values = time_values.compute()

    ds_history_season = india.get_season(
        pd.DataFrame(index=time_values).rename_axis("time"),
        False,
    ).to_xarray()

    ds_history = xr.merge([ds_history, ds_history_season]).set_coords("season")

    # Calculate mean
    print("Calculating pa_raw_mean...")
    ds_history["pa_raw_mean"] = (ds_history.a + ds_history.b) * 0.5

    return ds_history


def merge_history(in_path):
    """
    Open dataset with chunking for memory efficiency
    """

    # Use chunks to load data lazily with dask
    print("Opening dataset with chunking...")
    ds_pa = xr.open_dataset(
        os.path.join(in_path, "pr.nc"),
        chunks={"time": 1000},  # Process 1000 time steps at a time
    )

    print("Processing history...")
    ds_history = make_history(ds_pa)

    # Compute the final result efficiently
    print("Computing final result...")
    # If memory is still an issue, save to disk first
    temp_path = os.path.join(in_path, "history_temp.nc")

    # Use to_netcdf which will compute chunks progressively
    print("Saving to temporary file...")
    ds_history.to_netcdf(temp_path)

    # Load it back
    print("Loading result...")
    ds_history = xr.open_dataset(temp_path)

    # Optionally remove temp file
    # os.remove(temp_path)

    return ds_history
