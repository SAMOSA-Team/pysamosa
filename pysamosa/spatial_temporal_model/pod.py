# flake8: noqa
"""
POD Objects and Functions
Last Updated: April 30, 2025
This script contains the Proper Orthogonal Decomposition (POD) objects
and functions for spatial and temporal analysis.
@author: markjcampmier
"""
# Import Packages
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from scipy.linalg import svd, eigh

# Define GappyPOD Class


class GappyPOD:
    def __init__(self, n_modes=None, max_iter=100, tol=1e-6):
        self.n_modes = n_modes
        self.max_iter = max_iter
        self.tol = tol
        self.spatial_modes = None
        self.temporal_modes = None
        self.singular_values = None
        self.columns = None
        self.index = None

    def _dataframe_to_array(self, df):
        self.columns = df.columns
        self.index = df.index
        data = df.values
        mask = ~np.isnan(data)
        return data, mask

    def _array_to_dataframe(self, arr):
        return pd.DataFrame(arr, index=self.index, columns=self.columns)

    def fit_transform(
        self, ds, return_reconstruction=False, value="pa_campmier_delhi_mean"
    ):

        df = ds.to_dataframe().pivot_table(index="time", columns="site", values=value)

        data, mask = self._dataframe_to_array(df)

        # Initial guess for missing values
        filled_data = data.copy()
        for col in range(data.shape[1]):
            col_mask = mask[:, col]
            if np.any(col_mask):
                filled_data[~col_mask, col] = np.mean(data[col_mask, col])
            else:
                filled_data[:, col] = np.nanmean(data)

        # Initialize convergence tracking
        prev_error = np.inf
        self.reconstruction_error = []

        for iter in range(self.max_iter):
            # Compute POD
            U, s, Vt = self._compute_pod(filled_data)

            # Reconstruct data
            reconstruction = U @ np.diag(s) @ Vt

            # Update only missing values
            filled_data[~mask] = reconstruction[~mask]

            # Check convergence on known values
            error = np.mean((reconstruction[mask] - data[mask]) ** 2)
            self.reconstruction_error.append(error)

            if abs(error - prev_error) < self.tol:
                print(f"Converged after {iter + 1} iterations")
                break

            prev_error = error

        # Store final modes
        self.spatial_modes = U
        self.singular_values = s
        self.temporal_modes = Vt.T

        # Convert reconstruction back to DataFrame
        if return_reconstruction:
            return self._array_to_dataframe(reconstruction)

    def _compute_pod(self, data):
        U, s, Vt = svd(data, full_matrices=False)

        if self.n_modes is None:
            # Use energy criterion (95% of total energy)
            energy = np.cumsum(s**2) / np.sum(s**2)
            self.n_modes = np.argmax(energy >= 0.95) + 1

        return U[:, : self.n_modes], s[: self.n_modes], Vt[: self.n_modes, :]


# Define Spatial POD Class


class SpatialPOD:
    def __init__(self, n_modes=None):
        self.n_modes = n_modes
        self.spatial_modes = None
        self.singular_values = None
        self.band_modes = None

    def _compute_pod(self, data):
        data = data.values
        U, s, Vt = svd(data, full_matrices=False)

        if self.n_modes is None:
            energy = np.cumsum(s**2) / np.sum(s**2)
            self.n_modes = np.argmax(energy >= 0.95) + 1

        return U[:, : self.n_modes], s[: self.n_modes], Vt[: self.n_modes, :]

    def fit_transform(self, ds):
        df = ds.to_dataframe()

        self.location_index = df.index
        self.band_cols = df.columns

        U, s, Vt = self._compute_pod(df)

        self.spatial_modes = Vt.T
        self.singular_values = s
        self.band_modes = U


# Define Functions


def perform_gappy_pod_on_imf(
    imfs, imf_idx, timestamps=None, site_ids=None, max_iterations=10, tol=1e-6
):
    """
    Perform Gappy POD analysis on a specific IMF level from MEMD output.
    Handles missing data (NaN values).

    Parameters:
    -----------
    imfs : ndarray
        IMFs array with shape (m, n, p) where:
        - m is the number of IMF levels
        - n is the number of time points
        - p is the number of stations/locations
    imf_idx : int
        Index of the IMF level to analyze (0 for highest frequency)
    timestamps : array-like, optional
        Timestamps corresponding to the time dimension
    site_ids : list, optional
        List of site identifiers for each station
    max_iterations : int
        Maximum number of iterations for Gappy POD convergence
    tol : float
        Tolerance for convergence

    Returns:
    --------
    pod_results : dict
        Dictionary containing POD analysis results
    """
    # Extract dimensions
    n_imfs, n_times, n_stations = imfs.shape

    # Ensure imf_idx is valid
    if imf_idx >= n_imfs:
        raise ValueError(
            f"IMF index {imf_idx} is out of range. Available IMFs: 0-{n_imfs-1}"
        )

    # Create default site IDs if not provided
    if site_ids is None:
        site_ids = [f"Station_{i+1}" for i in range(n_stations)]

    # Create default timestamps if not provided
    if timestamps is None:
        timestamps = np.arange(n_times)

    print(
        f"Performing Gappy POD analysis on IMF level {imf_idx} across {n_stations} stations"
    )

    # Extract the selected IMF level
    # This gives us a matrix of shape (n_times, n_stations)
    X = imfs[imf_idx].copy()

    # Check for missing data
    mask = np.isnan(X)
    has_missing_data = np.any(mask)

    if has_missing_data:
        print(
            f"Found missing data: {np.sum(mask)} NaN values out of {X.size} total points"
        )

        # Create a mask matrix (1 for valid data, 0 for missing data)
        valid_mask = ~mask

        # Initialize the filled matrix with column means for missing values
        X_filled = X.copy()

        for col in range(n_stations):
            # Get valid values for this column
            valid_vals = X[:, col][valid_mask[:, col]]
            if len(valid_vals) > 0:
                # Fill missing values with column mean
                col_mean = np.nanmean(X[:, col])
                X_filled[mask[:, col], col] = col_mean

        # Gappy POD iteration
        prev_norm = np.linalg.norm(X_filled)
        converged = False

        for iteration in range(max_iterations):
            # Step 1: Compute correlation matrix and POD modes using current filled data
            C = np.corrcoef(X_filled.T)
            eigenvalues, eigenvectors = eigh(C)

            # Sort in descending order
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Step 2: Determine number of modes to use (explain 99% variance)
            explained_var = eigenvalues / np.sum(eigenvalues)
            cum_var = np.cumsum(explained_var)
            n_modes = np.searchsorted(cum_var, 0.99) + 1
            n_modes = max(
                n_modes, min(3, n_stations)
            )  # Use at least 3 modes if available

            print(
                f"  Iteration {iteration+1}: Using {n_modes} modes explaining {cum_var[n_modes-1]*100:.1f}% of variance"
            )

            # Step 3: Compute temporal coefficients
            modes = eigenvectors[:, :n_modes]
            temporal_coeff = X_filled @ modes

            # Step 4: Reconstruct the field
            X_recon = temporal_coeff @ modes.T

            # Step 5: Update only the missing data points
            X_filled[mask] = X_recon[mask]

            # Check convergence
            current_norm = np.linalg.norm(X_filled)
            rel_change = np.abs(current_norm - prev_norm) / prev_norm

            if rel_change < tol:
                converged = True
                print(
                    f"  Converged after {iteration+1} iterations (relative change: {rel_change:.2e})"
                )
                break

            prev_norm = current_norm

        if not converged:
            print(
                f"  Did not converge after {max_iterations} iterations (final relative change: {rel_change:.2e})"
            )

        # Use the filled data matrix for POD analysis
        X_analysis = X_filled
    else:
        print("No missing data found, proceeding with standard POD analysis")
        X_analysis = X

    # Compute correlation matrix for final analysis
    C = np.corrcoef(X_analysis.T)

    # Eigendecomposition to extract spatial modes
    eigenvalues, eigenvectors = eigh(C)

    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Normalize eigenvalues to show percentage of variance explained
    explained_var = (eigenvalues / np.sum(eigenvalues)) * 100

    # Project data onto spatial modes to get temporal coefficients
    # Shape: (n_times, n_stations)
    temporal_coeff = np.zeros((n_times, n_stations))
    for i in range(n_stations):
        # For columns with missing data, use only valid points for projection
        valid_times = ~mask[:, i] if has_missing_data else np.ones(n_times, dtype=bool)
        if np.any(valid_times):
            for j in range(n_stations):
                # Project using valid data points
                if np.sum(valid_times) > 0:
                    temporal_coeff[valid_times, j] = (
                        X_analysis[valid_times, :] @ eigenvectors[:, j]
                    )
                else:
                    # If all data points are missing for this column, set to zero
                    temporal_coeff[:, j] = 0

    # Save both the original data and the filled data
    return {
        "imf_level": imf_idx,
        "correlation_matrix": C,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "explained_variance": explained_var,
        "temporal_coefficients": temporal_coeff,
        "site_ids": site_ids,
        "timestamps": timestamps,
        "original_data": X,
        "filled_data": X_analysis if has_missing_data else X,
        "missing_mask": mask if has_missing_data else np.zeros_like(X, dtype=bool),
    }


def visualize_pod_results(pod_results, n_modes=3, show_data_reconstruction=False):
    """
    Visualize the results of POD analysis.
    """
    # Extract data from POD results
    imf_level = pod_results["imf_level"]
    C = pod_results["correlation_matrix"]
    eigenvalues = pod_results["eigenvalues"]
    eigenvectors = pod_results["eigenvectors"]
    explained_var = pod_results["explained_variance"]
    temporal_coeff = pod_results["temporal_coefficients"]
    site_ids = pod_results["site_ids"]
    timestamps = pod_results["timestamps"]
    missing_mask = pod_results.get("missing_mask", None)

    n_stations = len(site_ids)
    n_modes = min(n_modes, n_stations)

    # Determine if we have missing data
    has_missing_data = missing_mask is not None and np.any(missing_mask)

    # Determine number of rows based on whether to show data reconstruction
    n_rows = 4 if show_data_reconstruction else 3

    # Create figure
    fig = plt.figure(figsize=(16, 4 * n_rows))

    # Plot 1: Spatial correlation matrix
    ax1 = plt.subplot2grid((n_rows, 3), (0, 0), colspan=1, rowspan=1)
    im = ax1.imshow(C, cmap="cmc.roma_r", vmin=-1, vmax=1)
    ax1.set_xticks(np.arange(n_stations))
    ax1.set_yticks(np.arange(n_stations))
    ax1.set_xticklabels(site_ids, rotation=45, ha="right")
    ax1.set_yticklabels(site_ids)
    ax1.set_title(f"Spatial Correlation Matrix - IMF {imf_level}")
    plt.colorbar(im, ax=ax1)

    # Plot 2: Eigenvalue spectrum (explained variance)
    ax2 = plt.subplot2grid((n_rows, 3), (0, 1), colspan=2, rowspan=1)
    ax2.bar(range(1, n_stations + 1), explained_var, color="skyblue")
    ax2.plot(range(1, n_stations + 1), np.cumsum(explained_var), "ro-", alpha=0.7)
    ax2.set_xlabel("Mode Number")
    ax2.set_ylabel("Explained Variance (%)")
    ax2.set_title("Eigenvalue Spectrum")
    ax2.set_xticks(range(1, n_stations + 1))
    ax2.axhline(y=95, color="gray", linestyle="--", alpha=0.7)

    # Add cumulative variance label
    cumul_var_labels = [f"{x:.1f}%" for x in np.cumsum(explained_var)]
    for i, txt in enumerate(cumul_var_labels):
        if i < 5:  # Only label first few points to avoid clutter
            ax2.annotate(
                txt,
                (i + 1, np.cumsum(explained_var)[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

    # Plot 3: Spatial modes visualization
    ax3 = plt.subplot2grid((n_rows, 3), (1, 0), colspan=3, rowspan=1)

    # Bar plot of spatial modes
    width = 0.8 / n_modes
    for i in range(n_modes):
        positions = np.arange(n_stations) + (i - n_modes / 2 + 0.5) * width
        ax3.bar(
            positions,
            eigenvectors[:, i],
            width=width,
            label=f"Mode {i+1} ({explained_var[i]:.1f}%)",
        )

    ax3.set_xticks(np.arange(n_stations))
    ax3.set_xticklabels(site_ids, rotation=45, ha="right")
    ax3.set_ylabel("Mode Coefficient")
    ax3.set_title(f"Top {n_modes} Spatial Modes")
    ax3.legend()
    ax3.axhline(y=0, color="black", linestyle="-", alpha=0.3)

    # Plot 4: Temporal evolution of modes
    ax4 = plt.subplot2grid((n_rows, 3), (2, 0), colspan=3, rowspan=1)

    # Plot temporal coefficients
    for i in range(n_modes):
        ax4.plot(timestamps, temporal_coeff[:, i], label=f"Mode {i+1}", alpha=0.7)

    ax4.set_xlabel("Time")
    ax4.set_ylabel("Amplitude")
    ax4.set_title("Temporal Evolution of Spatial Modes")
    ax4.legend()

    # Format time axis if timestamps are datetime objects
    if isinstance(timestamps[0], (pd.Timestamp, np.datetime64)):
        ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

    # Plot 5 & 6: Original Data vs Reconstruction (if requested)
    if show_data_reconstruction and n_rows > 3:
        # Get data
        orig_data = pod_results["original_data"]
        filled_data = pod_results["filled_data"]

        # Plot original data with missing values shown
        ax5 = plt.subplot2grid((n_rows, 3), (3, 0), colspan=1, rowspan=1)
        im = ax5.imshow(orig_data.T, aspect="auto", cmap="viridis")
        ax5.set_xlabel("Time")
        ax5.set_ylabel("Station")
        ax5.set_yticks(np.arange(n_stations))
        ax5.set_yticklabels(site_ids)
        ax5.set_title("Original Data (white = missing)")
        plt.colorbar(im, ax=ax5)

        # Highlight missing data
        if has_missing_data:
            # Create a masked array to show missing data
            masked_data = np.ma.array(orig_data.T, mask=~missing_mask.T)
            ax5.imshow(
                masked_data, aspect="auto", cmap="binary", vmin=0, vmax=1, alpha=0.3
            )

        # Plot reconstructed/filled data
        ax6 = plt.subplot2grid((n_rows, 3), (3, 1), colspan=2, rowspan=1)
        im = ax6.imshow(filled_data.T, aspect="auto", cmap="viridis")
        ax6.set_xlabel("Time")
        ax6.set_ylabel("Station")
        ax6.set_yticks(np.arange(n_stations))
        ax6.set_yticklabels(site_ids)
        ax6.set_title("Reconstructed Data (Gappy POD filled)")
        plt.colorbar(im, ax=ax6)

    plt.tight_layout()
    return fig


def analyze_all_imf_levels(
    imfs,
    timestamps=None,
    site_ids=None,
    levels_to_analyze=None,
    show_reconstruction=False,
):
    """ """
    n_imfs, _, _ = imfs.shape

    # Default: analyze all IMF levels
    if levels_to_analyze is None:
        levels_to_analyze = range(n_imfs)

    # Validate IMF levels
    levels_to_analyze = [level for level in levels_to_analyze if level < n_imfs]

    # Perform POD for each level
    all_results = []
    for level in levels_to_analyze:
        print(f"Analyzing IMF level {level}")
        results = perform_gappy_pod_on_imf(imfs, level, timestamps, site_ids)
        all_results.append(results)

        # Visualize results
        fig = visualize_pod_results(
            results, show_data_reconstruction=show_reconstruction
        )
        plt.show()

    return all_results


def compare_imf_modes(all_pod_results, mode_idx=0):
    """
    Compare the same mode across different IMF levels.
    """
    n_levels = len(all_pod_results)

    # Extract site IDs (assuming all results have the same sites)
    site_ids = all_pod_results[0]["site_ids"]
    n_stations = len(site_ids)

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Mode patterns across IMF levels
    ax = axes[0]

    # Set width for grouped bars
    width = 0.8 / n_levels

    # Get data for each IMF level
    for i, results in enumerate(all_pod_results):
        level = results["imf_level"]
        eigenvectors = results["eigenvectors"]
        explained_var = results["explained_variance"]

        # Position bars for this level
        positions = np.arange(n_stations) + (i - n_levels / 2 + 0.5) * width

        # Plot the selected mode
        ax.bar(
            positions,
            eigenvectors[:, mode_idx],
            width=width,
            label=f"IMF {level} ({explained_var[mode_idx]:.1f}%)",
        )

    ax.set_xticks(np.arange(n_stations))
    ax.set_xticklabels(site_ids, rotation=45, ha="right")
    ax.set_ylabel("Mode Coefficient")
    ax.set_title(f"Mode {mode_idx+1} Patterns Across IMF Levels")
    ax.legend()
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

    # Plot 2: Explained variance for this mode across IMF levels
    ax = axes[1]

    levels = [results["imf_level"] for results in all_pod_results]
    variances = [results["explained_variance"][mode_idx] for results in all_pod_results]

    ax.bar(levels, variances, color="skyblue")
    ax.set_xlabel("IMF Level")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title(f"Importance of Mode {mode_idx+1} Across IMF Levels")
    ax.set_xticks(levels)

    plt.tight_layout()
    return fig
