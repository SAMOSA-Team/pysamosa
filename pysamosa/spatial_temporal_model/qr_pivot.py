"""
QR Pivot Functions
Last Updated: April 30, 2025
This script contains the QR Pivoting (QR) method for selecting optimal sensor locations.
QR refers to Q (Orthogonal) and R or U (Upper Triangle) matrices
arising from the eponymous QR (or QU) decomposition.
@author: markjcampmier
"""
# Import Packages
import numpy as np
from scipy.linalg import qr

# Define Functions


def find_optimal_sensors(pod_results, n_modes=3, n_sensors=None):
    """
    Find optimal sensor locations using QR pivoting on POD modes.
    """
    # Extract eigenvectors (spatial modes)
    eigenvectors = pod_results["eigenvectors"][:, :n_modes]

    # Transpose to get mode matrix (rows=modes, columns=locations)
    mode_matrix = eigenvectors.T

    # Apply QR decomposition with column pivoting
    _, _, p = qr(mode_matrix, pivoting=True)

    # Convert to list and limit to requested number of sensors
    sensor_ranking = p.tolist()
    if n_sensors is not None:
        sensor_ranking = sensor_ranking[:n_sensors]

    return sensor_ranking


def interpret_sensor_ranking(pod_results, sensor_ranking):
    """
    Interpret the sensor ranking in terms of site IDs and importance.
    """
    site_ids = pod_results["site_ids"]
    explained_var = pod_results["explained_variance"]

    # Calculate cumulative information captured
    cumulative_info = np.zeros(len(sensor_ranking))
    for i, idx in enumerate(sensor_ranking):
        # Create a mask with only the first i+1 sensors
        mask = np.zeros(len(site_ids), dtype=bool)
        mask[sensor_ranking[: i + 1]] = True

        # Calculate information captured by these sensors
        # (This is a simplified approach - a more rigorous method would
        # reconstruct the full field from the selected sensors)
        cumulative_info[i] = np.sum(explained_var[: i + 1])

    # Calculate incremental importance of each sensor
    importance = np.zeros(len(sensor_ranking))
    importance[0] = cumulative_info[0]
    for i in range(1, len(importance)):
        importance[i] = cumulative_info[i] - cumulative_info[i - 1]

    # Create list of (site_id, importance) tuples
    ranked_sites = [
        (site_ids[idx], importance[i]) for i, idx in enumerate(sensor_ranking)
    ]

    return ranked_sites


def optimize_sensor_network_for_imf(all_pod_results, imf_idx, n_modes=3):
    """
    Optimize sensor network for a specific IMF level.
    """
    pod_result = all_pod_results[imf_idx]
    sensor_ranking = find_optimal_sensors(pod_result, n_modes=n_modes)
    ranked_sites = interpret_sensor_ranking(pod_result, sensor_ranking)

    print(f"Optimal sensor locations for IMF {imf_idx}: ")
    for i, (site, importance) in enumerate(ranked_sites):
        print(f"{i + 1}. {site} (information gain: {importance: .2f}%)")

    return ranked_sites
