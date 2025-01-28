import numpy as np


def create_sampling_matrix(ground_ds, remote_ds, remote_df):
    # Get ground station coordinates
    ground_lats = ground_ds.latitude.values
    ground_lons = ground_ds.longitude.values

    # Get original remote sensing grid
    lats = remote_ds.latitude.values
    lons = remote_ds.longitude.values

    # Initialize sampling matrix
    n_stations = len(ground_ds.site)
    n_locations = len(remote_df)  # This matches our POD input
    phi = np.zeros((n_stations, n_locations))

    # For each ground station, find nearest remote sensing point
    for i, (lat, lon) in enumerate(zip(ground_lats, ground_lons)):
        # Find nearest point indices in original grid
        lat_idx = np.abs(lats - lat).argmin()
        lon_idx = np.abs(lons - lon).argmin()

        # Convert to flattened index matching our POD format
        flat_idx = lat_idx * len(lons) + lon_idx

        # Set corresponding element to 1
        phi[i, flat_idx] = 1

    return phi


def create_weighted_sampling_matrix(ground_ds, remote_ds, radius_km=2, sigma_km=5):
    # Extract coordinates
    ground_lats = ground_ds.latitude.values
    ground_lons = ground_ds.longitude.values

    # Create meshgrid of remote sensing coordinates
    lons, lats = np.meshgrid(remote_ds.longitude, remote_ds.latitude)
    n_locations = len(lats.flatten())

    # Initialize weighted matrix
    n_stations = len(ground_ds.site)
    weighted_phi = np.zeros((n_stations, n_locations))

    # Convert radius and sigma to degrees (approximate)
    deg_per_km = 1 / 111  # rough conversion
    radius_deg = radius_km * deg_per_km
    sigma_deg = sigma_km * deg_per_km

    # For each ground station
    for i in range(n_stations):
        # Calculate distances to all remote sensing points
        distances = np.sqrt(
            (lats.flatten() - ground_lats[i]) ** 2
            + (lons.flatten() - ground_lons[i]) ** 2
        )

        # Apply Gaussian weighting within radius
        mask = distances <= radius_deg
        if np.any(mask):  # Check if we have any points within radius
            weighted_phi[i, mask] = np.exp(-0.5 * (distances[mask] / sigma_deg) ** 2)

            # Normalize weights for this station if any weights exist
            station_sum = np.sum(weighted_phi[i, :])
            if station_sum > 0:
                weighted_phi[i, :] = weighted_phi[i, :] / station_sum
            else:
                # If no weights, use nearest point
                nearest_idx = np.argmin(distances)
                weighted_phi[i, nearest_idx] = 1.0

    # Add final check for any remaining NaN or inf
    weighted_phi = np.nan_to_num(weighted_phi, nan=0.0, posinf=0.0, neginf=0.0)

    return weighted_phi
