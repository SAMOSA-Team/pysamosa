"""Sampling matrix construction for mapping remote-sensing locations to ground stations."""

import numpy as np


def create_sampling_matrix(ground_ds, remote_ds):
    """Build a binary sampling matrix mapping flattened remote-sensing locations to ground stations.

    Args:
        ground_ds: Ground station dataset with 'latitude', 'longitude', and 'site' coords.
        remote_ds: Remote-sensing dataset with 'latitude' and 'longitude' coords.

    Returns:
        Sampling matrix of shape (n_locations, n_stations) with 1 at the nearest cell.
    """
    ground_lats = ground_ds.latitude.values
    ground_lons = ground_ds.longitude.values

    lats = remote_ds.latitude.values
    lons = remote_ds.longitude.values

    n_stations = len(ground_ds.site)
    n_locations = len(lats) * len(lons)
    phi = np.zeros((n_locations, n_stations))

    for i, (lat, lon) in enumerate(zip(ground_lats, ground_lons)):
        lat_idx = np.abs(lats - lat).argmin()
        lon_idx = np.abs(lons - lon).argmin()
        flat_idx = lat_idx * len(lons) + lon_idx
        phi[flat_idx, i] = 1

    return phi


def create_weighted_sampling_matrix(ground_ds, remote_ds, radius_km=2, sigma_km=5):
    """Build a Gaussian-weighted sampling matrix within a radius around each ground station.

    Args:
        ground_ds: Ground station dataset with 'latitude', 'longitude', and 'site' coords.
        remote_ds: Remote-sensing dataset with 'latitude' and 'longitude' coords.
        radius_km: Search radius in km; only remote-sensing cells within this radius receive weight.
        sigma_km: Gaussian kernel standard deviation in km.

    Returns:
        Weighted sampling matrix of shape (n_locations, n_stations), column-normalised.
    """
    ground_lats = ground_ds.latitude.values
    ground_lons = ground_ds.longitude.values

    lons, lats = np.meshgrid(remote_ds.longitude, remote_ds.latitude)
    n_locations = len(lats.flatten())
    n_stations = len(ground_ds.site)
    weighted_phi = np.zeros((n_locations, n_stations))

    deg_per_km = 1 / 111
    radius_deg = radius_km * deg_per_km
    sigma_deg = sigma_km * deg_per_km

    for i in range(n_stations):
        distances = np.sqrt(
            (lats.flatten() - ground_lats[i]) ** 2
            + (lons.flatten() - ground_lons[i]) ** 2
        )
        mask = distances <= radius_deg
        if np.any(mask):
            weighted_phi[mask, i] = np.exp(-0.5 * (distances[mask] / sigma_deg) ** 2)
            station_sum = np.sum(weighted_phi[:, i])
            if station_sum > 0:
                weighted_phi[:, i] = weighted_phi[:, i] / station_sum
            else:
                nearest_idx = np.argmin(distances)
                weighted_phi[nearest_idx, i] = 1.0

    return np.nan_to_num(weighted_phi, nan=0.0, posinf=0.0, neginf=0.0)
