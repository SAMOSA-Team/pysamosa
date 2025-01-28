import numpy as np
import pandas as pd

# import pod
# import sampling
import reprojection


class SpatialTemporalModel:
    def __init__(self, loc_points, time_index):
        self.time_index = time_index
        self.loc_points = loc_points
        self.gpod = None
        self.spod = None
        self.phi = None
        self.C = None

    def _compute_sampling_matrix(self):
        self.phi = [0]

    def _compute_reprojection_matrix(self):
        self.C = reprojection.compute_reprojection_mp(
            self.spod.spatial_modes, self.gpod.temporal_modes, self.phi
        )

    def _compute_reconstruction_matrix(self):
        s = np.diag(self.gpod.singular_values)
        return (self.spod.temporal_modes @ self.C @ s @ self.gpod.spatial_modes.T).T

    def fit_transform(self):
        self._compute_sampling_matrix()
        self._compute_reprojection_matrix()

        reconstruction = self._compute_reconstruction_matrix()

        df = pd.DataFrame(reconstruction.T)

        df.loc[:, "time"] = self.time_index
        df = df.melt(id_vars=["time"], var_name="location")

        df.loc[:, "latitude"] = self.loc_points.loc[df.location, "latitude"].values
        df.loc[:, "longitude"] = self.loc_points.loc[df.location, "longitude"].values

        df = df.drop(["location"], axis=1)

        df = df.set_index(["time", "latitude", "longitude"])
        ds = df.to_xarray()

        return ds

    def validate(self, ds_reconstruction):
        return
