"""Proper Orthogonal Decomposition (POD) classes for spatial and temporal analysis."""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # noqa: F401
import cmcrameri

from scipy.linalg import svd, eigh


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


class GappyPOD:
    """
    Enhanced Gappy POD with uncertainty tracking and iterative refinement.

    Key improvements:
    - Tracks original vs reconstructed values
    - Uses uncertainty weighting in iterations
    - Limited iterations (3-5) to prevent overfitting
    - Compatible with MEMD output format
    """

    def __init__(
        self,
        n_modes: int | None = None,
        max_iter: int = 5,
        tol: float = 1e-3,
        energy_threshold: float = 0.95,
        weight_decay: float = 0.8,
    ):
        """Args:
        n_modes: Number of POD modes to retain; auto-selected via *energy_threshold* if None.
        max_iter: Maximum Gappy POD iterations.
        tol: Convergence tolerance on mode changes.
        energy_threshold: Cumulative energy fraction for automatic mode selection.
        weight_decay: Blend weight for new reconstruction vs. previous (0–1).
        """
        self.n_modes = n_modes
        self.max_iter = max_iter
        self.tol = tol
        self.energy_threshold = energy_threshold
        self.weight_decay = weight_decay

        # Results storage
        self.spatial_modes = None
        self.temporal_modes = None
        self.singular_values = None
        self.mean_field = None

        # Metadata
        self.columns = None
        self.index = None
        self.is_original = None
        self.uncertainty = None
        self.reconstruction_history = []
        self.mode_history = []
        self.n_iterations_actual = 0

    def fit_transform(
        self,
        data: pd.DataFrame | np.ndarray | object,
        return_reconstruction: bool = False,
        return_components: bool = False,
        value: str = "pa_campmier_delhi_mean",
    ) -> pd.DataFrame | dict:
        """Fit Gappy POD to data with missing values and optionally return reconstruction.

        Args:
            data: Input data — DataFrame, ndarray, or xarray Dataset with missing values.
            return_reconstruction: Return the reconstructed DataFrame if True.
            return_components: Return a full components dict (for MEMD workflow) if True.
            value: Variable name when *data* is an xarray Dataset.

        Returns:
            Reconstructed data or components dict depending on flags; self otherwise.
        """
        # Convert input to numpy array
        if hasattr(data, "to_dataframe"):
            # xarray Dataset
            df = data.to_dataframe().pivot_table(
                index="time", columns="site", values=value
            )
            data_array, mask = self._dataframe_to_array(df)
        elif isinstance(data, pd.DataFrame):
            data_array, mask = self._dataframe_to_array(data)
        else:
            # Assume numpy array
            data_array = np.asarray(data)
            if data_array.ndim == 1:
                data_array = data_array.reshape(-1, 1)
            mask = ~np.isnan(data_array)
            self.index = np.arange(data_array.shape[0])
            self.columns = [f"Var_{i}" for i in range(data_array.shape[1])]

        # Store original data mask
        self.is_original = mask.copy()

        # Remove mean field
        self.mean_field = np.nanmean(data_array, axis=0)
        data_centered = data_array - self.mean_field

        # Initialize filled data
        filled_data = self._initialize_missing_values(data_centered, mask)

        # Initialize uncertainty
        self.uncertainty = (~mask).astype(float)  # 1 for missing, 0 for original

        # Iterative refinement
        self.reconstruction_history = []
        self.mode_history = []
        prev_modes = None

        for iteration in range(self.max_iter):
            # Compute weighted POD
            U, s, Vt = self._compute_weighted_pod(filled_data, self.uncertainty)

            # Store modes history
            self.mode_history.append(Vt.copy())

            # Reconstruct
            reconstruction = U @ np.diag(s) @ Vt

            # Update missing values with weighting
            if iteration > 0:
                # Weighted update for missing values
                for i in range(data_array.shape[0]):
                    for j in range(data_array.shape[1]):
                        if not mask[i, j]:
                            old_val = filled_data[i, j]
                            new_val = reconstruction[i, j]
                            filled_data[i, j] = (
                                self.weight_decay * new_val
                                + (1 - self.weight_decay) * old_val
                            )
                            # Update uncertainty (decreases with iterations)
                            self.uncertainty[i, j] = self.uncertainty[i, j] * 0.9
            else:
                # First iteration - direct update
                filled_data[~mask] = reconstruction[~mask]

            # Track reconstruction error on known values
            error = np.sqrt(np.mean((reconstruction[mask] - data_centered[mask]) ** 2))
            self.reconstruction_history.append(
                {"iteration": iteration, "rmse": error, "n_modes": len(s)}
            )

            # Check convergence based on mode changes
            if prev_modes is not None:
                mode_change = np.mean(np.abs(Vt - prev_modes))
                if mode_change < self.tol:
                    print(
                        f"Gappy POD converged after {iteration + 1} iterations (mode change: {mode_change:.6f})"
                    )
                    break

            prev_modes = Vt.copy()

        self.n_iterations_actual = iteration + 1

        # Store final results
        self.spatial_modes = (
            Vt.T
        )  # Transpose to match expected format (n_stations, n_modes)
        self.temporal_modes = U @ np.diag(s)  # (n_time, n_modes)
        self.singular_values = s

        # Add back mean field for final reconstruction
        final_reconstruction = reconstruction + self.mean_field

        if return_components:
            # Return dictionary for MEMD workflow
            return {
                "spatial_modes": self.spatial_modes,
                "temporal_coefficients": self.temporal_modes,
                "singular_values": self.singular_values,
                "mean_field": self.mean_field,
                "is_original": self.is_original,
                "uncertainty": self.uncertainty,
                "reconstruction": final_reconstruction,
                "n_iterations": self.n_iterations_actual,
                "reconstruction_history": self.reconstruction_history,
            }
        elif return_reconstruction:
            # Return as DataFrame if requested
            if self.columns is not None:
                return self._array_to_dataframe(final_reconstruction)
            else:
                return final_reconstruction
        else:
            # Default: return self for method chaining
            return self

    def _dataframe_to_array(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Convert DataFrame to array and extract mask."""
        self.columns = df.columns
        self.index = df.index
        data = df.values
        mask = ~np.isnan(data)
        return data, mask

    def _array_to_dataframe(self, arr: np.ndarray) -> pd.DataFrame:
        """Convert array back to DataFrame."""
        return pd.DataFrame(arr, index=self.index, columns=self.columns)

    def _initialize_missing_values(
        self, data: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """
        Initialize missing values with station means or global mean.
        """
        filled_data = data.copy()

        for col in range(data.shape[1]):
            col_mask = mask[:, col]
            if np.any(col_mask):
                # Use column mean for missing values
                col_mean = np.mean(data[col_mask, col])
                filled_data[~col_mask, col] = col_mean
            else:
                # If entire column is missing, use global mean
                global_mean = np.nanmean(data)
                filled_data[:, col] = global_mean if not np.isnan(global_mean) else 0

        return filled_data

    def _compute_weighted_pod(
        self, data: np.ndarray, weights: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute POD with optional weighting for uncertainty.

        For now, uses standard SVD but could be extended to weighted SVD.
        """
        # Standard SVD
        U, s, Vt = svd(data, full_matrices=False)

        # Determine number of modes
        if self.n_modes is None:
            # Use energy criterion
            energy = np.cumsum(s**2) / np.sum(s**2)
            n_modes = np.argmax(energy >= self.energy_threshold) + 1
            n_modes = min(n_modes, len(s))  # Don't exceed available modes
        else:
            n_modes = min(self.n_modes, len(s))

        return U[:, :n_modes], s[:n_modes], Vt[:n_modes, :]

    def transform(self, new_data: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Project new data onto existing POD modes.
        """
        if self.spatial_modes is None:
            raise ValueError("Must fit before transform. Call fit_transform first.")

        # Convert to array
        if isinstance(new_data, pd.DataFrame):
            data_array = new_data.values
        else:
            data_array = np.asarray(new_data)

        # Center using stored mean
        data_centered = data_array - self.mean_field

        # Project onto spatial modes
        temporal_coeffs = data_centered @ self.spatial_modes

        return temporal_coeffs

    def inverse_transform(self, temporal_coeffs: np.ndarray) -> np.ndarray:
        """
        Reconstruct data from temporal coefficients.
        """
        if self.spatial_modes is None:
            raise ValueError("Must fit before inverse_transform.")

        # Reconstruct centered data
        reconstruction = temporal_coeffs @ self.spatial_modes.T

        # Add back mean field
        reconstruction += self.mean_field

        return reconstruction

    def get_variance_explained(self) -> np.ndarray:
        """
        Get variance explained by each mode.
        """
        if self.singular_values is None:
            raise ValueError("Must fit first.")

        variance = self.singular_values**2
        return variance / np.sum(variance)

    def get_cumulative_variance(self) -> np.ndarray:
        """
        Get cumulative variance explained.
        """
        return np.cumsum(self.get_variance_explained())

    def plot_convergence(self):
        """
        Plot convergence history.
        """
        if not self.reconstruction_history:
            print("No convergence history available.")
            return

        import matplotlib.pyplot as plt

        iterations = [h["iteration"] for h in self.reconstruction_history]
        rmse = [h["rmse"] for h in self.reconstruction_history]

        plt.figure(figsize=(8, 5))
        plt.plot(iterations, rmse, "o-")
        plt.xlabel("Iteration")
        plt.ylabel("RMSE on Known Values")
        plt.title("Gappy POD Convergence")
        plt.grid(True, alpha=0.3)
        plt.show()

    def get_uncertainty_summary(self) -> dict:
        """
        Get summary of uncertainty in reconstruction.
        """
        if self.uncertainty is None:
            return {}

        n_total = self.uncertainty.size
        n_original = np.sum(self.is_original)
        n_reconstructed = n_total - n_original

        return {
            "total_values": n_total,
            "original_values": n_original,
            "reconstructed_values": n_reconstructed,
            "percent_reconstructed": 100 * n_reconstructed / n_total,
            "mean_uncertainty": np.mean(self.uncertainty),
            "max_uncertainty": np.max(self.uncertainty),
            "uncertainty_by_station": np.mean(self.uncertainty, axis=0),
            "uncertainty_by_time": np.mean(self.uncertainty, axis=1),
        }


def perform_gappy_pod_on_imf(
    imfs, imf_idx, timestamps=None, site_ids=None, max_iterations=10, tol=1e-6
):
    """Perform Gappy POD analysis on a single IMF level from MEMD output.

    Args:
        imfs: IMF array of shape (n_imfs, n_times, n_stations).
        imf_idx: IMF level index to analyse (0 = highest frequency).
        timestamps: Time axis labels; auto-generated integer range if None.
        site_ids: Station identifiers; auto-generated labels if None.
        max_iterations: Maximum Gappy POD iterations.
        tol: Relative change tolerance for convergence.

    Returns:
        Dictionary with POD analysis results (eigenvalues, eigenvectors, temporal coefficients, etc.).
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


def visualize_pod_results(
    pod_results, n_modes=3, show_data_reconstruction=False, figsize=(18, 16)
):

    cmap_correlation = cmcrameri.cm.vik  # Diverging colormap for correlation (-1 to 1)
    cmap_modes = (
        cmcrameri.cm.broc
    )  # Diverging colormap for modes (negative to positive)
    cmap_data = cmcrameri.cm.batlow  # Sequential colormap for data visualization

    # Extract data from POD results
    imf_level = pod_results["imf_level"]
    C = pod_results["correlation_matrix"]
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

    # Determine number of rows and adjust layout for large number of stations
    n_rows = 4 if show_data_reconstruction else 3

    # Create figure
    fig = plt.figure(figsize=figsize)

    # Define grid layout
    if n_stations > 20:
        # For many stations, give more space to correlation matrix and mode visualization
        gs = plt.GridSpec(
            n_rows, 3, height_ratios=[1, 4, 1] if n_rows == 3 else [1, 1.5, 1]
        )
    else:
        gs = plt.GridSpec(n_rows, 3)

    # Plot 1: Spatial correlation matrix (larger size)
    ax1 = plt.subplot(gs[0, :2])
    im = ax1.imshow(C, cmap=cmap_correlation, vmin=-1, vmax=1)
    ax1.set_axis_off()
    ax1.set_title(f"IMF {imf_level} - Correlation Matrix", fontsize=14)
    cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
    cbar.set_label("Correlation")

    # Plot 2: Eigenvalue spectrum (explained variance)
    ax2 = plt.subplot(gs[0, 2:])
    ax2.bar(
        range(1, min(21, n_stations)),
        explained_var[: min(20, n_stations)],
        color="#5799c5",
    )
    ax2.plot(
        range(1, min(21, n_stations)),
        np.cumsum(explained_var)[: min(20, n_stations)],
        color="#a63603",
        marker="o",
        alpha=0.7,
    )
    ax2.set_xlabel("Mode Number")
    ax2.set_ylabel("Explained Variance (%)")
    ax2.set_title("Eigenvalue Spectrum (top 20 modes)", fontsize=14)
    ax2.set_xticks(range(1, min(21, n_stations) + 1, 2))  # Show fewer ticks
    ax2.axhline(y=95, color="gray", linestyle="--", alpha=0.7)
    ax2.grid(alpha=0.3)

    # Add cumulative variance label for key points
    cumul_var = np.cumsum(explained_var)[: min(20, n_stations)]
    # Label at mode 1, 2, 5, 10, 20 or the last mode if less
    label_indices = [0, 1, 4, 9, 19]
    label_indices = [i for i in label_indices if i < len(cumul_var)]

    for i in label_indices:
        ax2.annotate(
            f"{cumul_var[i]:.1f}%",
            (i + 1, cumul_var[i]),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
        )

    # Plot 3: Spatial modes visualization as heatmap
    ax3 = plt.subplot(gs[1, 1:])

    # Create a data matrix for the heatmap - extracting just the top n_modes
    mode_matrix = eigenvectors[:, :n_modes].copy()

    # Find the maximum absolute value for symmetric colormap
    vmax = 1
    vmin = -1

    # Find most extreme values for each mode to help with sorting
    extremes = np.zeros(n_stations)
    for i in range(n_stations):
        mode_vals = mode_matrix[i, :n_modes]
        # Use the most extreme value (positive or negative)
        max_abs_idx = np.argmax(np.abs(mode_vals))
        extremes[i] = mode_vals[max_abs_idx]

    # Sort stations by their most extreme values
    sort_idx = np.argsort(extremes)

    # Apply sorting
    mode_matrix = mode_matrix[sort_idx, :]
    sorted_site_ids = [site_ids[i] for i in sort_idx]

    # Create heatmap
    im = ax3.imshow(mode_matrix, cmap=cmap_modes, aspect="auto", vmin=vmin, vmax=vmax)

    # Add y-axis labels (station IDs)
    ax3.set_yticks(np.arange(len(sorted_site_ids)))
    ax3.set_yticklabels(sorted_site_ids, fontsize=14)

    # Add x-axis labels (mode numbers)
    ax3.set_xticks(np.arange(n_modes))
    ax3.set_xticklabels(
        [f"Mode {i + 1}\n({explained_var[i]:.1f}%)" for i in range(n_modes)]
    )

    ax3.grid()
    # Add title
    ax3.set_title(f"Top {n_modes} Spatial Modes - IMF {imf_level}", fontsize=14)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
    cbar.set_label("Mode Coefficient")

    # Plot 4: Temporal evolution of modes
    ax4 = plt.subplot(gs[2, 1:])

    # Use a color cycle that works well with the Crameri colormaps
    colors = plt.cm.get_cmap("tab10", n_modes)

    # Plot temporal coefficients with improved styling
    for i in range(n_modes):
        ax4.plot(
            timestamps,
            temporal_coeff[:, i],
            color=colors(i),
            label=f"Mode {i + 1}",
            linewidth=2,
            alpha=0.8,
        )

    ax4.set_xlabel("Time")
    ax4.set_ylabel("Amplitude")
    ax4.set_title("Temporal Evolution of Spatial Modes", fontsize=14)
    ax4.legend(ncol=min(5, n_modes))  # Multi-column legend for many modes
    ax4.grid(alpha=0.3)

    # Format time axis if timestamps are datetime objects
    if isinstance(timestamps[0], (pd.Timestamp, np.datetime64)):
        ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

    # Plot 5 & 6: Original Data vs Reconstruction (if requested)
    if show_data_reconstruction and n_rows > 3:
        # Get data
        orig_data = pod_results["original_data"]
        filled_data = pod_results["filled_data"]

        # Create common colormap limits for both plots
        vmin = min(np.nanmin(orig_data), np.nanmin(filled_data))
        vmax = max(np.nanmax(orig_data), np.nanmax(filled_data))

        # Plot original data with missing values shown
        ax5 = plt.subplot(gs[3, :2])
        im = ax5.imshow(
            orig_data.T, aspect="auto", cmap=cmap_data, vmin=vmin, vmax=vmax
        )
        ax5.set_xlabel("Time Index")
        ax5.set_title("Original Data (white = missing)", fontsize=14)

        # For many stations, don't show all station labels
        if n_stations > 20:
            # Show a subset of labels
            step = max(1, n_stations // 10)
            ax5.set_yticks(np.arange(0, n_stations, step))
            ax5.set_yticklabels([site_ids[i] for i in range(0, n_stations, step)])
        else:
            ax5.set_yticks(np.arange(n_stations))
            ax5.set_yticklabels(site_ids)

        ax5.set_ylabel("Station")
        plt.colorbar(im, ax=ax5)

        # Highlight missing data
        if has_missing_data:
            # Create a masked array to show missing data
            masked_data = np.ma.array(orig_data.T, mask=~missing_mask.T)
            ax5.imshow(
                masked_data, aspect="auto", cmap="binary", vmin=0, vmax=1, alpha=0.3
            )

        # Plot reconstructed/filled data
        ax6 = plt.subplot(gs[3, 2:])
        im = ax6.imshow(
            filled_data.T, aspect="auto", cmap=cmap_data, vmin=vmin, vmax=vmax
        )
        ax6.set_xlabel("Time Index")
        ax6.set_title("Reconstructed Data (Gappy POD filled)", fontsize=14)

        # Match y-ticks with the original data plot
        if n_stations > 20:
            ax6.set_yticks(np.arange(0, n_stations, step))
            ax6.set_yticklabels([site_ids[i] for i in range(0, n_stations, step)])
        else:
            ax6.set_yticks(np.arange(n_stations))
            ax6.set_yticklabels(site_ids)

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
        visualize_pod_results(results, show_data_reconstruction=show_reconstruction)
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
