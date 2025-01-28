import numpy as np
from scipy.linalg import pinv


def compute_reprojection_mp(H_remote, W_ground, sampling_matrix):
    # First get the typical ratio between satellite and PM2.5 at training locations
    # Apply sampling matrix
    H_sampled = sampling_matrix @ H_remote

    # Incorporate unit conversion into reprojection
    H_sampled_converted = H_sampled

    # Compute C with unit-adjusted data
    C = pinv(H_sampled_converted) @ W_ground

    return C


def compute_reprojection_tr(H_remote, W_ground, sampling_matrix, lambda_reg=0.9):
    H_sampled = sampling_matrix @ H_remote
    HTH = H_sampled.T @ H_sampled
    n = HTH.shape[0]
    C = np.linalg.solve(HTH + lambda_reg * np.eye(n), H_sampled.T @ W_ground)
    return C
