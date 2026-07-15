"""Reprojection matrix computation for POD-based spatial reconstruction."""

import numpy as np
from scipy.linalg import pinv


def compute_reprojection_mp(H_remote, W_ground, sampling_matrix):
    """Compute the reprojection matrix via Moore-Penrose pseudo-inverse.

    Args:
        H_remote: Spatial band modes of shape (n_locations, r1).
        W_ground: Temporal ground modes of shape (n_times, r2).
        sampling_matrix: Sampling matrix of shape (n_locations, n_stations).

    Returns:
        Reprojection matrix C of shape (r1, r2).
    """
    H_sampled = H_remote.T @ sampling_matrix
    C = pinv(H_sampled).T @ W_ground
    return C


def compute_reprojection_tr(H_remote, W_ground, sampling_matrix, lambda_reg=0.9):
    """Compute the reprojection matrix via Tikhonov-regularised least squares.

    Args:
        H_remote: Spatial band modes of shape (n_locations, r1).
        W_ground: Temporal ground modes of shape (n_times, r2).
        sampling_matrix: Sampling matrix of shape (n_locations, n_stations).
        lambda_reg: Tikhonov regularisation coefficient.

    Returns:
        Reprojection matrix C.
    """
    H_sampled = sampling_matrix @ H_remote
    HTH = H_sampled.T @ H_sampled
    n = HTH.shape[0]
    C = np.linalg.solve(HTH + lambda_reg * np.eye(n), H_sampled.T @ W_ground)
    return C
