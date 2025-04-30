"""
GPR Functions
Last Updated: April 30, 2025
This script contains the Gaussian Process Regression (GPR)
functions for spatial and temporal interpolation.
@author: markjcampmier
"""
# Import Packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels

# Define Functions


def build_gpr_model_for_imf_pod(pod_results, imf_idx, mode_idx=0):
    """
    Build a Gaussian Process Regression model for a specific IMF-POD mode.
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


def predict_with_gpr_model(model, timestamps=None, n_points=100, return_std=True):
    """
    Make predictions with the GPR model.
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
    """
    Visualize GPR predictions.
    """
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
    """
    Reconstruct data from GPR models for multiple IMF-POD modes.
    """
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
