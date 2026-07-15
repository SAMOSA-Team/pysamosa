"""Gaussian Process Regression (GPR) functions for spatial and temporal interpolation."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels


def build_gpr_model_for_imf_pod(pod_results, imf_idx, mode_idx=0):
    """Fit a Gaussian Process Regression model for a specific IMF-POD temporal mode.

    Args:
        pod_results: POD result dict from :func:`perform_gappy_pod_on_imf`.
        imf_idx: IMF level index (controls kernel selection).
        mode_idx: Which spatial mode to model (0 = dominant mode).

    Returns:
        Dict with keys 'gpr', 'imf_idx', 'mode_idx', 'eigenvector', 'site_ids', etc.
    """
    # Extract data for this mode
    eigenvector = pod_results["eigenvectors"][:, mode_idx]  # Spatial pattern
    temporal_coeff = pod_results["temporal_coefficients"][
        :, mode_idx
    ]  # Temporal evolution
    timestamps = pod_results["timestamps"]
    site_ids = pod_results["site_ids"]

    # Convert timestamps to numeric (days since start)
    if isinstance(timestamps[0], (pd.Timestamp, np.datetime64)):
        t0 = pd.Timestamp(timestamps[0])
        t_numeric = np.array(
            [(pd.Timestamp(t) - t0).total_seconds() / (24 * 3600) for t in timestamps]
        )
    else:
        t_numeric = np.array(timestamps)

    # Define kernels
    # For IMF 0 (highest frequency): Use RBF + WhiteKernel
    # For IMF 1-2 (medium frequency): Use RBF + ExpSineSquared (for periodicity)
    # For IMF 3+ (low frequency): Use RBF with larger length scale
    if imf_idx == 0:
        kernel = kernels.ConstantKernel() * kernels.RBF(
            length_scale=1.0
        ) + kernels.WhiteKernel(noise_level=0.1)
    elif imf_idx in [1, 2]:
        # Add periodic component for daily/weekly patterns
        period = 1.0  # Start with 1-day period, will be optimized
        kernel = (
            kernels.ConstantKernel() * kernels.RBF(length_scale=5.0)
            + kernels.ConstantKernel()
            * kernels.ExpSineSquared(length_scale=1.0, periodicity=period)
            + kernels.WhiteKernel(noise_level=0.1)
        )
    else:
        # Longer length scale for slower variations
        kernel = kernels.ConstantKernel() * kernels.RBF(
            length_scale=10.0
        ) + kernels.WhiteKernel(noise_level=0.1)

    # Create and fit GP model
    gp = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=10, alpha=1e-10, normalize_y=True
    )
    gp.fit(t_numeric.reshape(-1, 1), temporal_coeff)

    print(f"Fitted GPR model for IMF {imf_idx}, Mode {mode_idx + 1}")
    print(f"Optimized kernel: {gp.kernel_}")

    # Return model and related data
    return {
        "gpr": gp,
        "imf_idx": imf_idx,
        "mode_idx": mode_idx,
        "eigenvector": eigenvector,
        "site_ids": site_ids,
        "t_train": t_numeric,
        "y_train": temporal_coeff,
        "explained_variance": pod_results["explained_variance"][mode_idx],
    }


def build_spatiotemporal_gpr_model(pod_results, station_coords, imf_idx, mode_idx=0):
    """Fit a joint spatiotemporal GPR model using station coordinates and temporal coefficients.

    Args:
        pod_results: POD result dict from :func:`perform_gappy_pod_on_imf`.
        station_coords: Array of shape (n_stations, 2) with [lat, lon] per station.
        imf_idx: IMF level index (controls kernel selection).
        mode_idx: Which spatial mode to model (0 = dominant mode).

    Returns:
        Dict with fitted GPR and supporting metadata.
    """
    # Extract data for this mode
    eigenvector = pod_results["eigenvectors"][:, mode_idx]  # Spatial pattern
    temporal_coeff = pod_results["temporal_coefficients"][
        :, mode_idx
    ]  # Temporal evolution
    timestamps = pod_results["timestamps"]
    site_ids = pod_results["site_ids"]

    # Convert timestamps to numeric (days since start)
    if isinstance(timestamps[0], (pd.Timestamp, np.datetime64)):
        t0 = pd.Timestamp(timestamps[0])
        t_numeric = np.array(
            [(pd.Timestamp(t) - t0).total_seconds() / (24 * 3600) for t in timestamps]
        )
    else:
        t_numeric = np.array(timestamps)

    # Get coordinates for each station
    # Build training data for spatiotemporal model
    X_train = []
    y_train = []

    # For each station and time point, create a training example
    for t_idx, t in enumerate(t_numeric):
        for s_idx, site in enumerate(site_ids):
            if site in station_coords.keys():
                # Feature vector: [time, lat, lon]
                X_train.append(
                    [
                        t,
                        station_coords[site]["latitude"],
                        station_coords[site]["longitude"],
                    ]
                )

                # Target value: mode contribution at this space-time point
                y_train.append(eigenvector[s_idx] * temporal_coeff[t_idx])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Normalize features to similar scales
    # Time and spatial coordinates can be on very different scales
    X_means = np.mean(X_train, axis=0)
    X_stds = np.std(X_train, axis=0)
    X_train_norm = (X_train - X_means) / X_stds

    # Define appropriate kernel for spatiotemporal modeling
    # Separate kernels for temporal and spatial components
    if imf_idx == 0:  # High frequency
        # Time kernel - shorter length scale for high frequency patterns
        time_kernel = kernels.ConstantKernel() * kernels.RBF(
            length_scale=1.0, length_scale_bounds=(0.1, 10)
        )
        # Spatial kernel - Matern is typically better for geographic data
        spatial_kernel = kernels.ConstantKernel() * kernels.Matern(
            length_scale=[1.0, 1.0], nu=1.5
        )

    elif imf_idx in [1, 2]:  # Medium frequency
        # Add periodic component for daily/weekly patterns
        time_kernel = kernels.ConstantKernel() * kernels.RBF(
            length_scale=5.0
        ) + kernels.ConstantKernel() * kernels.ExpSineSquared(
            length_scale=1.0, periodicity=1.0
        )
        spatial_kernel = kernels.ConstantKernel() * kernels.Matern(
            length_scale=[1.0, 1.0], nu=1.5
        )

    else:  # Low frequency
        # Longer length scales for both time and space
        time_kernel = kernels.ConstantKernel() * kernels.RBF(length_scale=10.0)
        spatial_kernel = kernels.ConstantKernel() * kernels.Matern(
            length_scale=[2.0, 2.0], nu=1.5
        )

    # Full kernel - using tensor product to model interactions between space and time
    # Note: we treat the first dimension as time, second and third as space
    kernel = kernels.WhiteKernel(noise_level=0.1)

    # Option 1: Sum kernel (time + space)
    # kernel += time_kernel + spatial_kernel

    # Option 2: Product kernel (time * space) - can better capture space-time interactions
    kernel += time_kernel * spatial_kernel

    # Create and fit GP model
    gp = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=5, normalize_y=True, alpha=1e-10
    )
    gp.fit(X_train_norm, y_train)

    print(f"Fitted spatiotemporal GPR model for IMF {imf_idx}, Mode {mode_idx + 1}")
    print(f"Optimized kernel: {gp.kernel_}")

    # Return model and related data
    return {
        "gpr": gp,
        "imf_idx": imf_idx,
        "mode_idx": mode_idx,
        "eigenvector": eigenvector,
        "temporal_coeff": temporal_coeff,
        "site_ids": site_ids,
        "station_locs": station_coords,
        "X_train": X_train,
        "X_train_norm": X_train_norm,
        "X_means": X_means,
        "X_stds": X_stds,
        "y_train": y_train,
        "t_numeric": t_numeric,
        "timestamps": timestamps,
        "explained_variance": pod_results["explained_variance"][mode_idx],
    }


def predict_with_gpr_model(model, timestamps=None, n_points=100, return_std=True):
    """Generate predictions from a fitted GPR model over a given time axis.

    Args:
        model: Model dict from :func:`build_gpr_model_for_imf_pod`.
        timestamps: Prediction timestamps; evenly spaced over training range if None.
        n_points: Number of prediction points when *timestamps* is None.
        return_std: Include predictive standard deviation in the result.

    Returns:
        Dict with 'times', 'values', and 'std' arrays.
    """
    # Get model components
    gp = model["gpr"]
    t_train = model["t_train"]

    # Generate prediction times if not provided
    if timestamps is None:
        t_predict = np.linspace(t_train.min(), t_train.max(), n_points)
    else:
        # Convert timestamps to same scale as training data
        if isinstance(timestamps[0], (pd.Timestamp, np.datetime64)):
            t0 = pd.Timestamp(model["t_train"][0])
            t_predict = np.array(
                [
                    (pd.Timestamp(t) - t0).total_seconds() / (24 * 3600)
                    for t in timestamps
                ]
            )
        else:
            t_predict = np.array(timestamps)

    # Make prediction
    if return_std:
        y_predict, y_std = gp.predict(t_predict.reshape(-1, 1), return_std=True)
    else:
        y_predict = gp.predict(t_predict.reshape(-1, 1))
        y_std = None

    # Return predictions
    return {"times": t_predict, "values": y_predict, "std": y_std, "model": model}


def visualize_gpr_predictions(
    predictions, original_timestamps=None, original_values=None
):
    """Plot GPR predictions with uncertainty band and optional training overlay."""
    # Get prediction components
    t_predict = predictions["times"]
    y_predict = predictions["values"]
    y_std = predictions["std"]
    model = predictions["model"]

    # Get model info for title
    imf_idx = model["imf_idx"]
    mode_idx = model["mode_idx"]
    var_explained = model["explained_variance"]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot original data if provided
    if original_timestamps is not None and original_values is not None:
        # Convert to same scale as predictions
        if isinstance(original_timestamps[0], (pd.Timestamp, np.datetime64)):
            t0 = pd.Timestamp(model["t_train"][0])
            t_orig = np.array(
                [
                    (pd.Timestamp(t) - t0).total_seconds() / (24 * 3600)
                    for t in original_timestamps
                ]
            )
        else:
            t_orig = np.array(original_timestamps)

        ax.scatter(
            t_orig, original_values, color="blue", alpha=0.5, label="Original data"
        )

    # Plot predictions
    ax.plot(t_predict, y_predict, "r-", label="Prediction")

    # Plot uncertainty
    if y_std is not None:
        ax.fill_between(
            t_predict,
            y_predict - 2 * y_std,
            y_predict + 2 * y_std,
            color="red",
            alpha=0.2,
            label="2σ confidence",
        )

    # Set title and labels
    ax.set_title(
        f"GPR Prediction for IMF {imf_idx}, Mode {mode_idx + 1} ({var_explained: .1f}% variance)"
    )
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Mode Amplitude")
    ax.legend()

    plt.tight_layout()
    return fig


def reconstruct_data_from_gpr_models(models, predictions_list, site_ids=None):
    """Reconstruct the full data field from GPR predictions across multiple IMF-POD modes."""
    # Check that models and predictions match
    if len(models) != len(predictions_list):
        raise ValueError("Number of models and predictions must match")

    # Get common prediction times (assume all predictions use same times)
    t_predict = predictions_list[0]["times"]

    # If site_ids not provided, use from first model
    if site_ids is None:
        site_ids = models[0]["site_ids"]

    # Initialize reconstruction array
    n_times = len(t_predict)
    n_sites = len(site_ids)
    reconstruction = np.zeros((n_times, n_sites))

    # For each model/prediction pair
    for model, pred in zip(models, predictions_list):
        # Get mode components
        eigenvector = model["eigenvector"]
        pred_values = pred["values"]

        # Get site IDs for this model
        model_sites = model["site_ids"]

        # Create mapping from model sites to output sites
        site_mapping = [
            model_sites.index(site) if site in model_sites else None
            for site in site_ids
        ]

        # Add contribution of this mode
        for i, site_idx in enumerate(site_mapping):
            if site_idx is not None:
                # For each time point, add mode contribution
                for t in range(n_times):
                    reconstruction[t, i] += pred_values[t] * eigenvector[site_idx]

    # Return reconstruction
    return {"times": t_predict, "site_ids": site_ids, "values": reconstruction}


def predict_with_spatiotemporal_gpr(
    model, times=None, locations=None, n_time_points=100, return_std=True
):
    """Generate predictions from a fitted spatiotemporal GPR model over time and space."""
    # Get model components
    gp = model["gpr"]
    t_train = model["t_numeric"]
    station_locs = model["station_locs"]
    X_means = model["X_means"]
    X_stds = model["X_stds"]

    # Generate prediction times if not provided
    if times is None:
        t_predict = np.linspace(t_train.min(), t_train.max(), n_time_points)
    else:
        # Convert timestamps to same scale as training data
        if isinstance(times[0], (pd.Timestamp, np.datetime64)):
            t0 = pd.Timestamp(model["t_numeric"][0])
            t_predict = np.array(
                [(pd.Timestamp(t) - t0).total_seconds() / (24 * 3600) for t in times]
            )
        else:
            t_predict = np.array(times)

    # Use original station locations if none provided
    if locations is None:
        locations = [
            (lat, lon)
            for lat, lon in station_locs.values()
            if not np.isnan(lat) and not np.isnan(lon)
        ]

    # Build prediction points grid (all combinations of times and locations)
    X_pred = []
    for t in t_predict:
        for lat, lon in locations:
            X_pred.append([t, lat, lon])

    X_pred = np.array(X_pred)

    # Normalize prediction points using the same normalization as training data
    X_pred_norm = (X_pred - X_means) / X_stds

    # Make prediction
    if return_std:
        y_predict, y_std = gp.predict(X_pred_norm, return_std=True)
    else:
        y_predict = gp.predict(X_pred_norm)
        y_std = None

    # Reshape predictions to a grid: times × locations
    predictions_grid = y_predict.reshape(len(t_predict), len(locations))

    if y_std is not None:
        std_grid = y_std.reshape(len(t_predict), len(locations))
    else:
        std_grid = None

    # Return predictions
    return {
        "times": t_predict,
        "locations": locations,
        "values": predictions_grid,
        "std": std_grid,
        "flat_values": y_predict,
        "flat_std": y_std,
        "X_pred": X_pred,
        "model": model,
    }


def reconstruct_data_from_spatiotemporal_gpr(models, predictions_list, locations=None):
    """Reconstruct the full data field by summing contributions from all IMF-POD spatiotemporal GPR modes.

    Args:
        models: List of model dicts from :func:`build_spatiotemporal_gpr_model`.
        predictions_list: List of prediction dicts from :func:`predict_with_spatiotemporal_gpr`.
        locations: List of (lat, lon) tuples to reconstruct; taken from first prediction if None.

    Returns:
        Dict with 'times', 'locations', 'values', and 'uncertainty' arrays.
    """
    # Check that models and predictions match
    if len(models) != len(predictions_list):
        raise ValueError("Number of models and predictions must match")

    # Get common prediction times and locations
    t_predict = predictions_list[0]["times"]

    if locations is None:
        locations = predictions_list[0]["locations"]

    # Initialize reconstruction array: times × locations
    n_times = len(t_predict)
    n_locs = len(locations)
    reconstruction = np.zeros((n_times, n_locs))
    uncertainty = np.zeros((n_times, n_locs))

    # For each model/prediction pair
    for _model, pred in zip(models, predictions_list):
        # Get prediction components
        pred_times = pred["times"]
        pred_locs = pred["locations"]
        pred_values = pred["values"]
        pred_std = pred["std"]

        # Check time compatibility
        if len(pred_times) != len(t_predict) or not np.allclose(pred_times, t_predict):
            raise ValueError("Prediction times don't match across models")

        # Check location compatibility
        if len(pred_locs) != len(locations) or not all(
            loc in pred_locs for loc in locations
        ):
            raise ValueError("Prediction locations don't match across models")

        # Add contribution of this mode
        reconstruction += pred_values

        # Propagate uncertainty (summing variances)
        if pred_std is not None:
            uncertainty += pred_std**2

    # Convert variance to standard deviation
    uncertainty = np.sqrt(uncertainty)

    # Return reconstruction
    return {
        "times": t_predict,
        "locations": locations,
        "values": reconstruction,
        "uncertainty": uncertainty,
    }


def leave_one_out_validation(pod_results, station_coords, imf_idx=0, n_modes=3):
    """Leave-one-out cross-validation of the spatiotemporal GPR model.

    Args:
        pod_results: POD result dict from :func:`perform_gappy_pod_on_imf`.
        station_coords: DataFrame with 'latitude' and 'longitude' per station.
        imf_idx: IMF level to validate.
        n_modes: Number of leading spatial modes to include.

    Returns:
        Dict with per-site RMSE, R², predictions, actuals, and uncertainties.
    """
    site_ids = pod_results["site_ids"]
    n_sites = len(site_ids)

    # Store results
    site_errors = {}
    site_r2_scores = {}
    site_predictions = {}
    site_actuals = {}
    site_uncertainties = {}

    # For each site, perform leave-one-out validation
    for held_out_idx, held_out_site in enumerate(site_ids):
        print(f"Validating site {held_out_site} ({held_out_idx + 1}/{n_sites})")

        # Create a mask excluding this site
        mask = np.ones(n_sites, dtype=bool)
        mask[held_out_idx] = False

        # Filter POD results to exclude this site
        loo_site_ids = [site for i, site in enumerate(site_ids) if i != held_out_idx]

        # For each mode
        mode_predictions = []
        mode_uncertainties = []
        mode_actuals = []

        for mode_idx in range(n_modes):
            # Get original eigenvector and temporal coefficient for this site/mode
            orig_eigenvector = pod_results["eigenvectors"][held_out_idx, mode_idx]
            temporal_coeff = pod_results["temporal_coefficients"][:, mode_idx]

            # Calculate actual values (what we're trying to predict)
            actual_values = orig_eigenvector * temporal_coeff

            # Create temporary POD results excluding this site
            temp_pod = {
                "eigenvectors": pod_results["eigenvectors"][mask, :],
                "temporal_coefficients": pod_results["temporal_coefficients"],
                "timestamps": pod_results["timestamps"],
                "site_ids": loo_site_ids,
                "explained_variance": pod_results["explained_variance"],
            }

            # Build spatiotemporal model
            model = build_spatiotemporal_gpr_model(
                temp_pod, station_coords, imf_idx, mode_idx
            )

            # Get held-out site coordinates
            held_out_coords = station_coords[
                station_coords["station_id"] == held_out_site
            ]
            if len(held_out_coords) > 0:
                held_out_lat = held_out_coords["lat"].values[0]
                held_out_lon = held_out_coords["lon"].values[0]

                # Predict at held-out location
                predictions = predict_with_spatiotemporal_gpr(
                    model,
                    times=pod_results["timestamps"],
                    locations=[(held_out_lat, held_out_lon)],
                    return_std=True,
                )

                # Extract predictions and uncertainty
                pred_values = predictions["values"][:, 0]  # First (only) location
                pred_std = (
                    predictions["std"][:, 0] if predictions["std"] is not None else None
                )

                # Store results
                mode_predictions.append(pred_values)
                mode_uncertainties.append(pred_std)
                mode_actuals.append(actual_values)
            else:
                print(
                    f"Warning: No coordinates found for {held_out_site}, skipping validation"
                )
                continue

        # Combine predictions across modes
        if mode_predictions:
            # Sum predictions from all modes
            combined_pred = np.sum(mode_predictions, axis=0)
            combined_actual = np.sum(mode_actuals, axis=0)

            # Sum variances for uncertainty
            combined_uncertainty = np.sqrt(
                np.sum([std**2 for std in mode_uncertainties], axis=0)
            )

            # Calculate error metrics
            errors = combined_pred - combined_actual
            mse = np.mean(errors**2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(errors))

            # Calculate R² score
            ss_total = np.sum((combined_actual - np.mean(combined_actual)) ** 2)
            ss_residual = np.sum(errors**2)
            r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0

            # Store metrics
            site_errors[held_out_site] = {
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "errors": errors,
            }
            site_r2_scores[held_out_site] = r2
            site_predictions[held_out_site] = combined_pred
            site_actuals[held_out_site] = combined_actual
            site_uncertainties[held_out_site] = combined_uncertainty

    return {
        "site_errors": site_errors,
        "site_r2_scores": site_r2_scores,
        "site_predictions": site_predictions,
        "site_actuals": site_actuals,
        "site_uncertainties": site_uncertainties,
        "mean_rmse": np.mean([metrics["rmse"] for metrics in site_errors.values()]),
        "mean_r2": np.mean(list(site_r2_scores.values())),
    }


def compute_variograms_for_imf_pod(pod_results, station_coords, imf_idx=0, n_modes=3):
    """Compute empirical and fitted variograms for spatial patterns in IMF-POD analysis.

    Args:
        pod_results: POD result dict from :func:`perform_gappy_pod_on_imf`.
        station_coords: DataFrame with 'latitude' and 'longitude' per station.
        imf_idx: IMF level to analyse.
        n_modes: Number of leading spatial modes to include.

    Returns:
        Dict with 'variograms', 'variogram_models', and 'variogram_parameters'.
    """
    import skgstat as skg  # Requires scikit-gstat package

    site_ids = pod_results["site_ids"]
    eigenvectors = pod_results["eigenvectors"]
    explained_var = pod_results["explained_variance"]

    # Store results
    variograms = {}
    variogram_models = {}
    variogram_parameters = {}

    # For each mode
    for mode_idx in range(min(n_modes, eigenvectors.shape[1])):
        # Extract spatial pattern (eigenvector)
        spatial_pattern = eigenvectors[:, mode_idx]

        # Get coordinates for each station
        coords = []
        values = []
        for i, site in enumerate(site_ids):
            site_data = station_coords[site]
            if len(site_data) > 0:
                lat = site_data["latitude"]
                lon = site_data["longitude"]
                if (
                    not np.isnan(lat)
                    and not np.isnan(lon)
                    and not np.isnan(spatial_pattern[i])
                ):
                    coords.append([lon, lat])  # Note: skgstat expects [x, y] format
                    values.append(spatial_pattern[i])

        coords = np.array(coords)
        values = np.array(values)

        if len(coords) < 5:
            print(
                f"Warning: Not enough valid coordinates for IMF {imf_idx}, Mode {mode_idx + 1}"
            )
            continue

        # Compute empirical variogram
        try:
            # Test for anisotropy
            for azimuth in [0, 45, 90, 135]:
                vario = skg.Variogram(
                    coords,
                    values,
                    maxlag="median",  # Use half of the median distance
                    n_lags=10,
                    azimuth=azimuth,
                    tolerance=45.0,
                )

                # Store empirical variogram
                key = f"azimuth_{azimuth}"
                if mode_idx not in variograms:
                    variograms[mode_idx] = {}

                variograms[mode_idx][key] = {
                    "bins": vario.bins,
                    "experimental": vario.experimental,
                    "n_lags": vario.n_lags,
                    "azimuth": azimuth,
                }

            # Main omnidirectional variogram for modeling
            vario = skg.Variogram(coords, values, maxlag="median", n_lags=10)

            # Store omnidirectional variogram
            variograms[mode_idx]["omnidirectional"] = {
                "bins": vario.bins,
                "experimental": vario.experimental,
                "n_lags": vario.n_lags,
            }

            # Fit models and store parameters
            models_to_try = ["spherical", "exponential", "gaussian", "matern"]
            model_results = {}

            for model_type in models_to_try:
                vario.model = model_type
                model_results[model_type] = {
                    "range": vario.parameters[0],
                    "sill": vario.parameters[1],
                    "nugget": vario.parameters[2],
                    "rmse": vario.rmse,
                    "r2": vario.describe()["r2"],
                }

            # Find best model
            best_model = min(model_results.items(), key=lambda x: x[1]["rmse"])[0]

            # Store model results
            variogram_models[mode_idx] = model_results
            variogram_parameters[mode_idx] = {
                "best_model": best_model,
                "parameters": model_results[best_model],
                "explained_variance": explained_var[mode_idx],
            }

            print(f"Computed variogram for IMF {imf_idx}, Mode {mode_idx + 1}")
            print(
                f"Best model: {best_model}, Range: {model_results[best_model]['range']: .2f}, "
                f"Sill: {model_results[best_model]['sill']: .4f}, Nugget: {model_results[best_model]['nugget']: .4f}"
            )

        except Exception as e:
            print(
                f"Error computing variogram for IMF {imf_idx}, Mode {mode_idx + 1}: {e}"
            )

    return {
        "imf_level": imf_idx,
        "variograms": variograms,
        "variogram_models": variogram_models,
        "variogram_parameters": variogram_parameters,
    }
