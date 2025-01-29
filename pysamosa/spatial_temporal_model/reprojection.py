import numpy as np
from scipy.linalg import pinv


def compute_reprojection_mp(H_remote, W_ground, sampling_matrix):
    # Apply sampling matrix
    H_sampled = sampling_matrix @ H_remote

    print(H_remote.shape)
    print(sampling_matrix.shape)
    print(H_sampled.shape)
    print(W_ground.shape)

    C = pinv(H_sampled) @ W_ground

    return C


def compute_reprojection_tr(H_remote, W_ground, sampling_matrix, lambda_reg=0.9):
    H_sampled = sampling_matrix @ H_remote
    HTH = H_sampled.T @ H_sampled
    n = HTH.shape[0]
    C = np.linalg.solve(HTH + lambda_reg * np.eye(n), H_sampled.T @ W_ground)
    return C
