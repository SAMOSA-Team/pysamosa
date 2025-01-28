import numpy as np
import pandas as pd
from scipy.linalg import svd


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

    def fit_transform(self, ds):

        df = ds.to_dataframe().pivot_table(
            index="time", columns="site", values="pa_campmier_delhi_mean"
        )

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
        df_reconstructed = self._array_to_dataframe(reconstruction)

        return df_reconstructed

    def _compute_pod(self, data):
        U, s, Vt = svd(data, full_matrices=False)

        if self.n_modes is None:
            # Use energy criterion (95% of total energy)
            energy = np.cumsum(s**2) / np.sum(s**2)
            self.n_modes = np.argmax(energy >= 0.95) + 1

        return U[:, : self.n_modes], s[: self.n_modes], Vt[: self.n_modes, :]


class SpatialPOD:
    def __init__(self, n_modes=None):
        self.n_modes = n_modes
        self.spatial_modes = None
        self.singular_values = None
        self.temporal_modes = None

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
        self.temporal_modes = U

        return self.spatial_modes
