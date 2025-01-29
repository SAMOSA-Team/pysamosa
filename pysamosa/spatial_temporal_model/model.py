import numpy as np
import pandas as pd
import xarray as xr

import pysamosa.spatial_temporal_model.pod as pod
import pysamosa.spatial_temporal_model.sampling as sampling
import pysamosa.spatial_temporal_model.reprojection as reprojection


class SpatialTemporalModel:
    def __init__(self, loc_points, time_index):
        self.time_index = time_index
        self.loc_points = loc_points
        self.gpod = None
        self.spod = None
        self.phi = None
        self.C = None

    def _compute_gpod(self, ds_ground, value="pa_campmier_delhi_mean"):
        self.gpod = pod.GappyPOD()
        self.gpod.fit_transform(ds_ground, value=value)

    def _compute_spod(self, ds_remote):
        self.spod = pod.SpatialPOD()
        self.spod.fit_transform(ds_remote)

    def _compute_sampling_matrix(self, ds_ground, ds_remote):
        self.phi = sampling.create_sampling_matrix(ds_ground, ds_remote)

    def _compute_reprojection_matrix(self):
        self.C = reprojection.compute_reprojection_mp(
            self.spod.band_modes, self.gpod.temporal_modes, self.phi
        )

    def _compute_reconstruction_matrix(self):
        s = np.diag(self.gpod.singular_values)
        return (self.spod.band_modes @ self.C @ s @ self.gpod.spatial_modes.T).T

    def fit_transform(self, return_validation=True):
        self._compute_reprojection_matrix()

        reconstruction = self._compute_reconstruction_matrix()

        df = pd.DataFrame(reconstruction)

        df.loc[:, "time"] = self.time_index
        df = df.set_index("time")

        df_validate = df.loc[:, pd.DataFrame(self.phi).sum().astype(bool).values]

        df = df.reset_index().melt(id_vars=["time"], var_name="location")

        df.loc[:, "latitude"] = self.loc_points.loc[df.location, "latitude"].values
        df.loc[:, "longitude"] = self.loc_points.loc[df.location, "longitude"].values

        df = df.drop(["location"], axis=1)

        df = df.set_index(["time", "latitude", "longitude"])
        ds = df.to_xarray()

        if return_validation:
            return ds, df_validate
        else:
            return ds

    def export_reconstruction(self, data_var_name="reconstruction"):
        # Get reconstruction
        reconstruction = self._compute_reconstruction_matrix()

        # Get unique lat/lon values
        unique_lats = np.unique(self.loc_points["latitude"].values)
        unique_lons = np.unique(self.loc_points["longitude"].values)

        # Initialize the 3D array (time, lat, lon)
        data_3d = np.full(
            (len(self.time_index), len(unique_lats), len(unique_lons)), np.nan
        )

        # Fill the 3D array
        for i, loc in enumerate(range(reconstruction.shape[1])):
            lat_idx = np.where(unique_lats == self.loc_points["latitude"].iloc[i])[0][0]
            lon_idx = np.where(unique_lons == self.loc_points["longitude"].iloc[i])[0][
                0
            ]
            data_3d[:, lat_idx, lon_idx] = reconstruction[:, i]

        # Get sampling information for training locations
        sampling_points = pd.DataFrame(self.phi).sum().astype(bool)
        training_mask = np.full((len(unique_lats), len(unique_lons)), False)
        for i, is_training in enumerate(sampling_points):
            if is_training:
                lat_idx = np.where(unique_lats == self.loc_points["latitude"].iloc[i])[
                    0
                ][0]
                lon_idx = np.where(unique_lons == self.loc_points["longitude"].iloc[i])[
                    0
                ][0]
                training_mask[lat_idx, lon_idx] = True

        # Create coordinates
        coords = {
            "time": self.time_index,
            "latitude": unique_lats,
            "longitude": unique_lons,
        }

        # Create data variables
        data_vars = {
            data_var_name: (["time", "latitude", "longitude"], data_3d),
            "is_training": (["latitude", "longitude"], training_mask),
        }

        # Create dataset
        ds = xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs={
                "description": f"Reconstructed Data Field ({data_var_name})",
                "creation_time": pd.Timestamp.now().isoformat(),
            },
        )

        # Add metadata
        ds[data_var_name].attrs["description"] = "Reconstructed values"
        ds.latitude.attrs["units"] = "degrees_north"
        ds.longitude.attrs["units"] = "degrees_east"
        ds.is_training.attrs[
            "description"
        ] = "Boolean mask indicating training locations"

        return ds

    def export_gpod(self):

        n_modes = len(self.gpod.singular_values)

        # Create coordinates
        coords = {
            "mode": np.arange(n_modes),
            "spatial_index": np.arange(self.gpod.spatial_modes.shape[0]),
            "temporal_index": np.arange(self.gpod.temporal_modes.shape[0]),
        }

        # Create data variables
        data_vars = {
            "singular_values": ("mode", self.gpod.singular_values),
            "spatial_modes": (["spatial_index", "mode"], self.gpod.spatial_modes),
            "temporal_modes": (["temporal_index", "mode"], self.gpod.temporal_modes),
        }

        # Create dataset
        ds = xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs={
                "description": "Gappy POD Components",
                "creation_time": pd.Timestamp.now().isoformat(),
            },
        )

        # Add metadata
        ds.singular_values.attrs[
            "description"
        ] = "Singular values from POD decomposition"
        ds.spatial_modes.attrs["description"] = "Spatial modes from POD decomposition"
        ds.temporal_modes.attrs["description"] = "Temporal modes from POD decomposition"

        return ds

    def export_spod(self):

        n_modes = self.spod.band_modes.shape[1]

        # Create coordinates
        coords = {
            "mode": np.arange(n_modes),
            "location": np.arange(self.loc_points.shape[0]),
        }

        # Create data variables
        data_vars = {
            "band_modes": (["location", "mode"], self.spod.band_modes),
            "latitude": ("location", self.loc_points["latitude"].values),
            "longitude": ("location", self.loc_points["longitude"].values),
        }

        # Create dataset
        ds = xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs={
                "description": "Spatial POD Components",
                "creation_time": pd.Timestamp.now().isoformat(),
            },
        )

        # Add metadata
        ds.band_modes.attrs["description"] = "Band modes from spatial POD"
        ds.latitude.attrs["units"] = "degrees_north"
        ds.longitude.attrs["units"] = "degrees_east"

        return ds

    def export_model_params(self):
        # Convert matrices to dataframes
        # Sampling matrix
        df_phi = pd.DataFrame(
            self.phi,
            columns=[f"loc_{i}" for i in range(self.phi.shape[1])],
            index=[f"spatial_{i}" for i in range(self.phi.shape[0])],
        ).add_prefix("phi_")

        # Reprojection matrix
        df_c = pd.DataFrame(
            self.C,
            columns=[f"mode_out_{i}" for i in range(self.C.shape[1])],
            index=[f"mode_in_{i}" for i in range(self.C.shape[0])],
        ).add_prefix("C_")

        # Reset indices to columns
        df_phi = df_phi.reset_index().rename(columns={"index": "spatial_index"})
        df_c = df_c.reset_index().rename(columns={"index": "mode_index"})

        return df_phi, df_c
