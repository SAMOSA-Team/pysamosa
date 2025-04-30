# flake8: noqa
"""
MEMD Object and Functions
Last Updated: April 30, 2025
This script contains the Multivariate Empirical Mode Decomposition (MEMD)
class, and associated functions.
@author: markjcampmier
"""
# Import Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import signal
from scipy.interpolate import Akima1DInterpolator
from scipy.stats import iqr

# Define MEMD Class


class MEMD:
    """
    Multi-Empirical Mode Decomposition (MEMD) for multivariate signals.

    This implementation is based on the paper:
    "Multivariate empirical mode decomposition" by N. Rehman and D. P. Mandic,
    Proc. Royal Society A, Vol. 466, No. 2117, pp. 1291-1302, 2010.
    """

    def __init__(self, n_directions=64, n_iterations=40, stopping_criterion=0.075):
        self.n_directions = n_directions
        self.n_iterations = n_iterations
        self.stopping_criterion = stopping_criterion
        self.imfs = None
        self.imf_energies = None
        self.variable_names = None
        self.time_index = None
        self.selected_imfs = None

    def _generate_direction_vectors(self, n_variables):
        if n_variables == 2:
            # For bivariate signals, use points on a unit circle
            angles = np.linspace(0, 2 * np.pi, self.n_directions, endpoint=False)
            directions = np.column_stack((np.cos(angles), np.sin(angles)))
        else:
            # For higher dimensions, use points on a unit hypersphere
            # This is based on the Hammersley sequence for uniform sampling
            directions = np.zeros((self.n_directions, n_variables))

            # First direction is [1, 0, 0, ..., 0]
            directions[0, 0] = 1

            for i in range(1, self.n_directions):
                # Generate quasi-random points on a hypersphere
                v = np.random.randn(n_variables)
                v = v / np.linalg.norm(v)
                directions[i] = v

        return directions

    def _project_signal(self, signal, direction):
        n_samples = signal.shape[0]
        projected_signal = np.zeros(n_samples)

        for i in range(n_samples):
            # Create a mask for non-NaN values
            valid_mask = ~np.isnan(signal[i, :])

            # Skip if all values are NaN
            if not np.any(valid_mask):
                projected_signal[i] = np.nan
                continue

            # Get valid parts of the signal and direction
            valid_signal = signal[i, valid_mask]
            valid_direction = direction[valid_mask]

            # Normalize valid direction to ensure unit length projection
            norm = np.linalg.norm(valid_direction)
            if norm > 1e-10:  # Avoid division by zero
                valid_direction = valid_direction / norm

            # Calculate projection
            projected_signal[i] = np.sum(valid_signal * valid_direction)

        return projected_signal

    def _find_extrema(self, projected_signal):
        # Handle NaN values by identifying valid segments of data
        valid_mask = ~np.isnan(projected_signal)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) < 3:  # Need at least 3 points to find extrema
            return np.array([], dtype=int), np.array([], dtype=int)

        # Extract valid segments
        segments = []
        current_segment = [valid_indices[0]]

        for i in range(1, len(valid_indices)):
            if valid_indices[i] == valid_indices[i - 1] + 1:
                current_segment.append(valid_indices[i])
            else:
                if len(current_segment) >= 3:  # Need at least 3 points in a segment
                    segments.append(current_segment)
                current_segment = [valid_indices[i]]

        if len(current_segment) >= 3:
            segments.append(current_segment)

        # Initialize empty arrays for extrema indices
        max_indices = np.array([], dtype=int)
        min_indices = np.array([], dtype=int)

        # Process each valid segment
        for segment in segments:
            segment_signal = projected_signal[segment]

            # Difference of the signal in this segment
            diff_signal = np.diff(segment_signal)

            # Find where the difference changes sign
            seg_max_idx = (
                np.where((diff_signal[:-1] > 0) & (diff_signal[1:] < 0))[0] + 1
            )
            seg_min_idx = (
                np.where((diff_signal[:-1] < 0) & (diff_signal[1:] > 0))[0] + 1
            )

            # Convert to original signal indices
            seg_max_idx = np.array([segment[idx] for idx in seg_max_idx])
            seg_min_idx = np.array([segment[idx] for idx in seg_min_idx])

            # Check if the segment endpoints are extrema
            if len(segment) > 2:
                start_idx = segment[0]
                end_idx = segment[-1]

                if projected_signal[start_idx] > projected_signal[start_idx + 1]:
                    seg_max_idx = np.insert(seg_max_idx, 0, start_idx)
                elif projected_signal[start_idx] < projected_signal[start_idx + 1]:
                    seg_min_idx = np.insert(seg_min_idx, 0, start_idx)

                if projected_signal[end_idx] > projected_signal[end_idx - 1]:
                    seg_max_idx = np.append(seg_max_idx, end_idx)
                elif projected_signal[end_idx] < projected_signal[end_idx - 1]:
                    seg_min_idx = np.append(seg_min_idx, end_idx)

            # Combine with overall extrema
            max_indices = np.append(max_indices, seg_max_idx)
            min_indices = np.append(min_indices, seg_min_idx)

        return max_indices, min_indices

    def _compute_envelopes(self, signal, time_points):
        n_samples, n_variables = signal.shape
        directions = self._generate_direction_vectors(n_variables)

        # Initialize the sum of envelopes and counter for valid directions
        sum_envelopes = np.zeros((n_samples, n_variables))
        valid_directions = 0

        for direction in directions:
            # Project the signal onto the direction, with proper NaN handling
            projected_signal = self._project_signal(signal, direction)

            # Skip if projection is all NaN
            if np.all(np.isnan(projected_signal)):
                continue

            # Find local extrema
            max_indices, min_indices = self._find_extrema(projected_signal)

            # Skip if not enough extrema are found
            if len(max_indices) < 2 or len(min_indices) < 2:
                continue

            # Compute upper and lower envelopes using interpolation
            max_env = np.full((n_samples, n_variables), np.nan)
            min_env = np.full((n_samples, n_variables), np.nan)

            envelope_is_valid = False

            for v in range(n_variables):
                # Skip variables with all NaN values
                if np.all(np.isnan(signal[:, v])):
                    continue

                # Filter out NaN values at extrema points for this variable
                valid_max_indices = np.array(
                    [idx for idx in max_indices if not np.isnan(signal[idx, v])]
                )
                valid_min_indices = np.array(
                    [idx for idx in min_indices if not np.isnan(signal[idx, v])]
                )

                # If not enough valid extrema, skip this variable
                if len(valid_max_indices) < 2 or len(valid_min_indices) < 2:
                    continue

                try:
                    # Upper envelope interpolation
                    if len(valid_max_indices) >= 4:  # Akima needs at least 4 points
                        try:
                            max_interpolator = Akima1DInterpolator(
                                time_points[valid_max_indices],
                                signal[valid_max_indices, v],
                            )
                            max_env[:, v] = max_interpolator(time_points)
                        except:
                            # Fall back to linear interpolation if Akima fails
                            max_env[:, v] = np.interp(
                                time_points,
                                time_points[valid_max_indices],
                                signal[valid_max_indices, v],
                            )
                    else:
                        # Linear interpolation for fewer points
                        max_env[:, v] = np.interp(
                            time_points,
                            time_points[valid_max_indices],
                            signal[valid_max_indices, v],
                        )

                    # Lower envelope interpolation
                    if len(valid_min_indices) >= 4:  # Akima needs at least 4 points
                        try:
                            min_interpolator = Akima1DInterpolator(
                                time_points[valid_min_indices],
                                signal[valid_min_indices, v],
                            )
                            min_env[:, v] = min_interpolator(time_points)
                        except:
                            # Fall back to linear interpolation if Akima fails
                            min_env[:, v] = np.interp(
                                time_points,
                                time_points[valid_min_indices],
                                signal[valid_min_indices, v],
                            )
                    else:
                        # Linear interpolation for fewer points
                        min_env[:, v] = np.interp(
                            time_points,
                            time_points[valid_min_indices],
                            signal[valid_min_indices, v],
                        )

                    # Mark as valid if we successfully computed at least one envelope
                    envelope_is_valid = True
                except Exception as e:
                    print(
                        f"Warning: Envelope interpolation failed for variable {v}: {e}"
                    )
                    # Keep NaN values for this variable

            # Only add this direction's envelopes if at least one variable had valid envelopes
            if envelope_is_valid:
                # Calculate mean envelope for this direction
                dir_env_mean = (max_env + min_env) / 2

                # Add to sum of envelopes, replacing NaNs with original signal values
                for v in range(n_variables):
                    valid_mask = ~np.isnan(dir_env_mean[:, v])
                    if np.any(valid_mask):
                        # Only update values that are valid in this direction's envelope
                        if valid_directions == 0:
                            # First valid direction - initialize values
                            sum_envelopes[:, v][valid_mask] = dir_env_mean[:, v][
                                valid_mask
                            ]
                        else:
                            # Add to existing values
                            sum_envelopes[:, v][valid_mask] += dir_env_mean[:, v][
                                valid_mask
                            ]

                valid_directions += 1

        # If no valid directions found, return the original signal
        if valid_directions == 0:
            return signal.copy()

        # Compute mean envelope
        mean_envelope = np.zeros_like(signal)
        for v in range(n_variables):
            valid_mask = ~np.isnan(sum_envelopes[:, v])
            if np.any(valid_mask):
                # Divide valid values by the count of valid directions
                mean_envelope[:, v][valid_mask] = (
                    sum_envelopes[:, v][valid_mask] / valid_directions
                )
            else:
                # If no valid values for this variable, use original signal
                mean_envelope[:, v] = signal[:, v]

        return mean_envelope

    def _check_imf(self, mode, mean_envelope):
        # Calculate the stopping criterion, handling NaN values
        epsilon = 0
        valid_vars = 0

        for v in range(mode.shape[1]):
            # Skip variables with all NaN values
            if np.all(np.isnan(mode[:, v])) or np.all(np.isnan(mean_envelope[:, v])):
                continue

            # Create masks for valid data points
            valid_mask = ~(np.isnan(mode[:, v]) | np.isnan(mean_envelope[:, v]))

            if not np.any(valid_mask):
                continue

            # Calculate amplitude using only valid data points
            valid_mode = mode[valid_mask, v]
            valid_envelope = mean_envelope[valid_mask, v]

            if len(valid_mode) == 0:
                continue

            amplitude = np.max(np.abs(valid_mode))

            # Avoid division by zero
            if amplitude > 1e-10:
                ratio = np.abs(valid_envelope) / amplitude
                epsilon += np.mean(ratio)
                valid_vars += 1

        # If no valid variables, return True to avoid infinite sifting
        if valid_vars == 0:
            return True

        epsilon /= valid_vars

        # Mode is an IMF if the stopping criterion is satisfied
        return epsilon < self.stopping_criterion

    def _sift(self, signal, time_points):
        n_samples, n_variables = signal.shape

        # Initialize mode and debug counters
        mode = signal.copy()
        iteration_count = 0

        for iteration in range(self.n_iterations):
            iteration_count += 1

            # Compute mean envelope
            mean_envelope = self._compute_envelopes(mode, time_points)

            # Check if mean_envelope contains any NaN values
            nan_in_envelope = np.any(np.isnan(mean_envelope))

            # Subtract mean envelope from mode
            h = mode - mean_envelope

            # Debug information for troubleshooting
            if iteration == 0 or iteration == 1:
                non_nan_count = np.sum(~np.isnan(h))
                non_zero_count = np.sum(np.abs(h) > 1e-10)
                print(
                    f"Sifting iteration {iteration}: Non-NaN values: {non_nan_count}, Non-zero values: {non_zero_count}"
                )
                if nan_in_envelope:
                    print(
                        f"  Warning: Mean envelope contains {np.sum(np.isnan(mean_envelope))} NaN values"
                    )

            # Check if h is all zeros or all NaNs (sifting has failed)
            if np.all(np.isnan(h)) or np.all(np.abs(h) < 1e-10):
                print(
                    f"Sifting terminated early at iteration {iteration}: All NaN or zero values"
                )
                # Return a copy of the original signal with small perturbation to avoid returning all zeros
                # This helps debug the issue while letting the algorithm continue
                perturbed = signal.copy()
                valid_mask = ~np.isnan(perturbed)
                if np.any(valid_mask):
                    perturbed[valid_mask] += np.random.normal(
                        0, 1e-5, np.sum(valid_mask)
                    )
                return perturbed, np.zeros_like(signal)

            # Check if h satisfies the IMF criteria
            if self._check_imf(h, mean_envelope):
                print(f"IMF criteria satisfied after {iteration+1} iterations")
                return h, signal - h

            # Update mode
            mode = h

        print(f"Maximum sifting iterations ({self.n_iterations}) reached")
        # Return the mode as IMF even if it doesn't satisfy the criteria
        return mode, signal - mode

    def decompose(self, signal, max_imfs=10, debug=False):
        # Check if signal is a pandas DataFrame
        if isinstance(signal, pd.DataFrame):
            # Store variable names and time index for later use
            self.variable_names = signal.columns.tolist()
            self.time_index = signal.index.values

            # Convert to numpy array
            signal = signal.values
            time_points = self.time_index
        else:
            # Reset variable names and time index
            self.variable_names = None
            self.time_index = None

            # Ensure signal is a numpy array
            signal = np.asarray(signal)

            # Generate time points
            time_points = np.arange(signal.shape[0])

        # Check if the signal is multivariate
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)

        n_samples, n_variables = signal.shape

        # If no variable names are provided, generate default ones
        if self.variable_names is None:
            self.variable_names = [f"Variable {i+1}" for i in range(n_variables)]

        # If no time index is provided, use the generated time points
        if self.time_index is None:
            self.time_index = time_points

        # Check for NaN values and warn if present
        nan_count = np.isnan(signal).sum()
        if nan_count > 0:
            print(
                f"Warning: Input signal contains {nan_count} NaN values out of {signal.size} total values."
            )
            print("NaN values will be handled during decomposition.")

        # Check for all-zero signal
        if np.all(np.abs(signal[~np.isnan(signal)]) < 1e-10):
            print("Warning: Input signal contains only zeros or NaNs.")
            self.imfs = np.zeros((1, n_samples, n_variables))
            self.selected_imfs = self.imfs  # Initialize selected IMFs
            return self.imfs

        # Initialize IMFs and residue
        imfs = []
        residue = signal.copy()

        # Extract IMFs
        for i in range(max_imfs):
            if debug:
                print(f"\nExtracting IMF {i+1}...")

            # Check residue
            non_nan_mask = ~np.isnan(residue)
            if not np.any(non_nan_mask):
                if debug:
                    print(f"Stopping: Residue contains only NaN values")
                break

            # Check if residue amplitude is very small
            if np.max(np.abs(residue[non_nan_mask])) < 1e-8:
                if debug:
                    print(f"Stopping: Residue amplitude too small (<1e-8)")
                break

            # Extract an IMF
            imf, residue = self._sift(residue, time_points)

            # Check if IMF extraction failed (all zeros or NaNs)
            if np.all(np.isnan(imf)) or np.all(np.abs(imf[~np.isnan(imf)]) < 1e-8):
                if debug:
                    print(f"IMF {i+1} extraction failed: All NaNs or near-zero values")
                if i == 0:  # If first IMF fails, something is wrong with the input
                    imfs.append(signal.copy())  # Return original signal as IMF
                break

            # Add the IMF to the list
            imfs.append(imf)

            if debug:
                print(f"IMF {i+1} successfully extracted")
                print(f"Residue max amplitude: {np.nanmax(np.abs(residue))}")

            # If residue has at most one extremum, stop
            # Calculate mean across variables, ignoring NaNs
            proj_residue = np.nanmean(residue, axis=1)

            # If all projection values are NaN, break
            if np.all(np.isnan(proj_residue)):
                if debug:
                    print(f"Stopping: Projected residue contains only NaN values")
                break

            # Find extrema in projection
            max_indices, min_indices = self._find_extrema(proj_residue)
            if len(max_indices) + len(min_indices) <= 2:
                if debug:
                    print(
                        f"Stopping: Residue has ≤ 2 extrema ({len(max_indices)} maxima, {len(min_indices)} minima)"
                    )
                break

        # Add the final residue as the last IMF only if it's not all zeros
        if not np.all(np.abs(residue[~np.isnan(residue)]) < 1e-8):
            imfs.append(residue)
            if debug:
                print(f"Adding final residue as IMF {len(imfs)}")

        # Convert list of IMFs to 3D array
        if not imfs:
            # If no IMFs were extracted, return original signal as a single IMF
            self.imfs = np.array([signal.copy()])
        else:
            self.imfs = np.array(imfs)

        # Calculate energy of each IMF
        self._calculate_imf_energies()

        # Initialize selected IMFs to all IMFs by default
        self.selected_imfs = self.imfs

        if debug:
            print(f"\nExtracted {len(imfs)} IMFs with shape {self.imfs.shape}")
            self.plot_imf_energies()

        return self.imfs

    def _calculate_imf_energies(self):
        if self.imfs is None:
            raise ValueError("No IMFs to calculate energy. Call decompose() first.")

        n_imfs, n_samples, n_variables = self.imfs.shape

        # Initialize energy arrays
        energies = np.zeros((n_imfs, n_variables))
        total_energies = np.zeros(n_imfs)

        # Calculate energy for each IMF and variable
        for i in range(n_imfs):
            for v in range(n_variables):
                # Skip NaN values when calculating energy
                valid_mask = ~np.isnan(self.imfs[i, :, v])
                if np.any(valid_mask):
                    energies[i, v] = np.sum(self.imfs[i, valid_mask, v] ** 2)

            # Total energy across all variables
            total_energies[i] = np.sum(energies[i])

        # Calculate percentages
        total_energy = np.sum(total_energies)
        if total_energy > 0:
            energy_percentages = (total_energies / total_energy) * 100
            cumulative_energy = np.cumsum(energy_percentages)
        else:
            energy_percentages = np.zeros(n_imfs)
            cumulative_energy = np.zeros(n_imfs)

        # Store energy information
        self.imf_energies = {
            "per_variable": energies,
            "total": total_energies,
            "percentage": energy_percentages,
            "cumulative": cumulative_energy,
        }

        return self.imf_energies

    def select_imfs(self, indices=None, energy_threshold=None):
        if self.imfs is None:
            raise ValueError("No IMFs to select. Call decompose() first.")

        # If both are None, keep all IMFs
        if indices is None and energy_threshold is None:
            self.selected_imfs = self.imfs
            return self.selected_imfs

        # Select by energy threshold
        if energy_threshold is not None:
            if self.imf_energies is None:
                self._calculate_imf_energies()

            # Find minimum number of IMFs to exceed threshold
            cumulative = self.imf_energies["cumulative"]
            n_imfs = len(cumulative)

            for i in range(n_imfs):
                if cumulative[i] >= energy_threshold:
                    indices = list(range(i + 1))  # Include IMFs up to this point
                    break
            else:
                # If threshold is never reached, include all IMFs
                indices = list(range(n_imfs))

            print(
                f"Selected {len(indices)} IMFs based on {energy_threshold}% energy threshold"
            )

        # Select by indices
        if indices is not None:
            self.selected_imfs = self.imfs[indices]
        else:
            self.selected_imfs = self.imfs

        return self.selected_imfs

    def get_reconstructed_signal(self):
        if self.selected_imfs is None:
            raise ValueError("No selected IMFs. Call select_imfs() first.")

        # Sum all selected IMFs
        reconstructed = np.nansum(self.selected_imfs, axis=0)

        return reconstructed

    def plot_imf_energies(self):
        if self.imf_energies is None:
            self._calculate_imf_energies()

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Number of IMFs
        n_imfs = len(self.imf_energies["percentage"])
        imf_labels = [f"IMF {i+1}" for i in range(n_imfs)]

        # Plot individual energy percentages
        ax1.bar(imf_labels, self.imf_energies["percentage"])
        ax1.set_title("IMF Energy Distribution")
        ax1.set_ylabel("Energy (%)")
        ax1.set_xlabel("IMF")
        ax1.grid(True, axis="y")

        # Plot cumulative energy
        ax2.step(imf_labels, self.imf_energies["cumulative"], where="mid", marker="o")
        ax2.set_title("Cumulative IMF Energy")
        ax2.set_ylabel("Cumulative Energy (%)")
        ax2.set_xlabel("IMF")
        ax2.grid(True)
        ax2.set_ylim(0, 105)  # Leave a little space at the top

        # Add threshold lines at common percentages
        for threshold in [50, 80, 90, 95, 99]:
            ax2.axhline(y=threshold, linestyle="--", alpha=0.5, color="gray")
            ax2.text(
                0, threshold + 1, f"{threshold}%", va="bottom", ha="left", fontsize=8
            )

        plt.tight_layout()
        plt.show()

    def plot_imfs(self, time_points=None, use_selected=True):
        # Choose which set of IMFs to plot
        if use_selected and self.selected_imfs is not None:
            imfs_to_plot = self.selected_imfs
        elif self.imfs is not None:
            imfs_to_plot = self.imfs
        else:
            raise ValueError("No IMFs to plot. Call decompose() first.")

        n_imfs, n_samples, n_variables = imfs_to_plot.shape

        # Use provided time points, or time index, or generate default
        if time_points is None:
            if self.time_index is not None:
                time_points = self.time_index
            else:
                time_points = np.arange(n_samples)

        # Create a figure
        fig, axes = plt.subplots(n_imfs, 1, figsize=(10, 2 * n_imfs), sharex=True)

        if n_imfs == 1:
            axes = [axes]

        # Plot each IMF
        for i, imf in enumerate(imfs_to_plot):
            ax = axes[i]
            for v in range(n_variables):
                var_name = (
                    self.variable_names[v] if self.variable_names else f"Variable {v+1}"
                )
                ax.plot(time_points, imf[:, v], label=var_name)

            # Get original IMF index if using selected
            if (
                use_selected
                and self.selected_imfs is not None
                and self.imfs is not None
            ):
                # Find which original IMF this corresponds to
                for j in range(len(self.imfs)):
                    if np.array_equal(imf, self.imfs[j]):
                        ax.set_ylabel(f"IMF {j+1}")
                        break
                else:
                    ax.set_ylabel(f"IMF {i+1}")
            else:
                ax.set_ylabel(f"IMF {i+1}")

            # ax.legend()
            ax.grid(True)

        axes[-1].set_xlabel("Time")
        plt.tight_layout()
        plt.show()

    def plot_3d_decomposition(self, signal=None, time_points=None, use_selected=True):
        # Choose which set of IMFs to plot
        if use_selected and self.selected_imfs is not None:
            imfs_to_plot = self.selected_imfs
        elif self.imfs is not None:
            imfs_to_plot = self.imfs
        else:
            raise ValueError("No IMFs to plot. Call decompose() first.")

        n_imfs, n_samples, n_variables = imfs_to_plot.shape

        if n_variables != 3:
            raise ValueError(
                "This method is only for trivariate signals (3 variables)."
            )

        # Use provided time points, or time index, or generate default
        if time_points is None:
            if self.time_index is not None:
                time_points = self.time_index
            else:
                time_points = np.arange(n_samples)

        # Handle signal data
        if signal is None:
            # If no signal is provided, reconstruct from all IMFs
            signal = np.nansum(self.imfs, axis=0)
        elif isinstance(signal, pd.DataFrame):
            # If DataFrame, convert to numpy array
            signal = signal.values

        # Create a figure
        fig = plt.figure(figsize=(15, 10))

        # Get variable names for axis labels
        var_names = []
        for i in range(3):
            var_names.append(
                self.variable_names[i] if self.variable_names else f"Variable {i+1}"
            )

        # Plot original signal
        ax_orig = fig.add_subplot(2, 2, 1, projection="3d")
        ax_orig.plot(signal[:, 0], signal[:, 1], signal[:, 2])
        ax_orig.set_title("Original Signal")
        ax_orig.set_xlabel(var_names[0])
        ax_orig.set_ylabel(var_names[1])
        ax_orig.set_zlabel(var_names[2])

        # Plot IMFs (up to 3)
        for i in range(min(3, n_imfs)):
            ax = fig.add_subplot(2, 2, i + 2, projection="3d")
            ax.plot(imfs_to_plot[i, :, 0], imfs_to_plot[i, :, 1], imfs_to_plot[i, :, 2])

            # Get original IMF index if using selected
            if (
                use_selected
                and self.selected_imfs is not None
                and self.imfs is not None
            ):
                # Find which original IMF this corresponds to
                for j in range(len(self.imfs)):
                    if np.array_equal(imfs_to_plot[i], self.imfs[j]):
                        ax.set_title(f"IMF {j+1}")
                        break
                else:
                    ax.set_title(f"IMF {i+1}")
            else:
                ax.set_title(f"IMF {i+1}")

            ax.set_xlabel(var_names[0])
            ax.set_ylabel(var_names[1])
            ax.set_zlabel(var_names[2])

        plt.tight_layout()
        plt.show()


# Define Functions


def compare_imf_across_locations(imfs_array, imf_index=0):
    m, n, p = imfs_array.shape
    selected_imf = imfs_array[:, :, imf_index]  # shape: (m, n) - time × locations

    # Calculate metrics for each location
    location_metrics = {
        "std_dev": np.std(
            selected_imf, axis=0
        ),  # Array of length n (one value per location)
        "iqr": np.array([iqr(selected_imf[:, loc]) for loc in range(n)]),
        "energy": np.sum(selected_imf**2, axis=0),
        "abs_mean": np.mean(np.abs(selected_imf), axis=0),
        "rms": np.sqrt(np.mean(selected_imf**2, axis=0)),
    }

    # Calculate frequency domain metrics (more complex)
    dominant_freqs = np.zeros(n)
    spectral_energy = np.zeros(n)

    for loc in range(n):
        freqs, powers = signal.welch(selected_imf[:, loc], fs=1.0, nperseg=min(256, m))
        dominant_freqs[loc] = freqs[np.argmax(powers)]
        spectral_energy[loc] = np.sum(powers)

    location_metrics["dominant_frequency"] = dominant_freqs
    location_metrics["spectral_energy"] = spectral_energy

    return location_metrics


def visualize_location_comparison(metrics, metric_name, location_names=None):
    if metric_name not in metrics:
        raise ValueError(f"Metric '{metric_name}' not found in metrics dictionary")

    values = metrics[metric_name]
    n_locations = len(values)

    if location_names is None:
        location_names = [f"Location {i}" for i in range(n_locations)]

    plt.figure(figsize=(10, 6))
    plt.bar(location_names, values)
    plt.title(f"{metric_name.replace('_', ' ').title()} of IMF Across Locations")
    plt.ylabel(metric_name.replace("_", " ").title())
    plt.xlabel("Location")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
