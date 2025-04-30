import numpy as np
from scipy.linalg import pinv


def compute_reprojection_mp(H_remote, W_ground, sampling_matrix):
    # Apply sampling matrix - Note: sampling_matrix should be (n_locations, n_stations)

    # No need to transpose sampling_matrix as it's already in the correct orientation
    H_sampled = H_remote.T @ sampling_matrix
    C = pinv(H_sampled).T @ W_ground

    return C


def compute_reprojection_tr(H_remote, W_ground, sampling_matrix, lambda_reg=0.9):
    # Ensure consistent approach with compute_reprojection_mp
    # sampling_matrix should be (n_locations, n_stations)
    H_sampled = sampling_matrix @ H_remote
    HTH = H_sampled.T @ H_sampled
    n = HTH.shape[0]
    C = np.linalg.solve(HTH + lambda_reg * np.eye(n), H_sampled.T @ W_ground)
    return C
