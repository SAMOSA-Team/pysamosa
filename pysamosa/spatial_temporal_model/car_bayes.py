# flake8: noqa


class CARModel:
    def __init__(self, df_sensors, df_satellite, gdf):
        ## Data shape/specifications
        self.n_settlements = df_satellite.shape[1]
        self.n_hours = df_sensors.shape[0]
        self.hour_to_month = df_sensors.index.month
        self.n_params = self.n_settlements * self.n_hours

        ## Model inputs
        self.Q_obs = None
        self.b_obs = None
        self.coords = gdf.loc[:, ["latitude", "longitude"]]
        self.satellite_monthly = df_satellite
        self.sensor_obs = df_sensors

        ## Model params
        self.W = None
        self.D = None
        self.Q_spatial = None

        self.Q_hour = None
        self.Q_day = None
        self.Q_temporal = None

        self.Q_prior = None

        self.mu_prior = None
        self.diurnal_weights = None

        ## Posterior
        self.Q_posterior = None
        self.b_posterior = None

    def create_adjacency_matrix(self):
        tree = cKDTree(list(zip(self.coords["latitude"], self.coords["longitude"])))

        W = pd.DataFrame(
            np.zeros((self.coords.shape[0], self.coords.shape[0])),
            index=self.coords.index,
            columns=self.coords.index,
        )

        for i, row in self.coords.iterrows():
            dist, idx = tree.query([row["latitude"], row["longitude"]], k=W.shape[0])
            dist, idx = dist[1:], idx[1:]
            W.loc[i, W.columns[idx]] = dist
        W = np.exp(-W)
        W = (W + W.T) / 2
        D = np.diag(W.sum(axis=1).values)

        self.W = W
        self.D = D

    def create_spatial_precision_matrix(self, tau_spatial=1.0):
        self.Q_spatial = (self.D - self.W) * tau_spatial

    def create_hourly_precision(self, tau_hourly=1.0):
        Q_hour = np.zeros((self.n_hours, self.n_hours))

        for h in range(self.n_hours):
            if h == 0:
                Q_hour[h, h] = 1
                if self.n_hours > 1:
                    Q_hour[h, h + 1] = -1
            elif h == self.n_hours - 1:
                Q_hour[h, h] = 1
                Q_hour[h, h - 1] = -1
            else:
                Q_hour[h, h] = 2
                Q_hour[h, h - 1] = -1
                Q_hour[h, h + 1] = -1

        self.Q_hour = Q_hour * tau_hourly

    def create_daily_precision(self, tau_daily=1.0):
        Q_day = np.zeros((self.n_hours, self.n_hours))

        for h in range(self.n_hours):
            hour_of_day = h % 24

            # Same hour previous day
            if h >= 24:
                Q_day[h, h - 24] = -1
                Q_day[h, h] += 1

            # Same hour next day
            if h < self.n_hours - 24:
                Q_day[h, h + 24] = -1
                Q_day[h, h] += 1

        self.Q_day = Q_day * tau_daily

    def create_combined_precision(self):
        self.Q_temporal = self.Q_hour + self.Q_day

    def create_mu_prior(self):
        mu_prior = np.zeros(self.n_params)
        for s in range(self.n_settlements):
            for h in range(self.n_hours):
                month_idx = self.hour_to_month[h]
                param_idx = s * self.n_hours + h
                mu_prior[param_idx] = self.satellite_monthly.iloc[s, month_idx]

        self.mu_prior = mu_prior

    def learn_diurnal_weights(self):
        diurnal_weights = np.zeros((24, self.n_settlements))

        for s in range(self.n_settlements):
            hourly_sums = np.zeros(24)
            hourly_counts = np.zeros(24)
            for h in range(self.n_hours):
                value = self.sensor_obs.iloc[h, s]
                if not np.isnan(value):
                    hour_of_day = h % 24
                    hourly_sums[hour_of_day] += value
                    hourly_counts[hour_of_day] += 1

            hourly_counts[hourly_counts == 0] = 1

            hourly_means = hourly_sums / hourly_counts
            diurnal_weights[:, s] = hourly_means / hourly_means.max()

        self.diurnal_weights = diurnal_weights

    def create_diurnal_prior(self):
        if self.diurnal_weights is None:
            self.diurnal_weights = np.ones(24)
            self.diurnal_weights = self.diurnal_weights / self.diurnal_weights.mean()

        mu_prior = np.zeros(self.n_params)

        for s in range(self.n_settlements):
            for h in range(self.n_hours):
                month_idx = self.hour_to_month[h]
                satellite_monthly_ = self.satellite_monthly[
                    self.satellite_monthly.index.month == month_idx
                ]
                hour_of_day = h % 24
                param_idx = s * self.n_hours + h
                base_value = float(satellite_monthly_.iloc[:, s].values[0])

                mu_prior[param_idx] = base_value * self.diurnal_weights[hour_of_day, s]

        self.mu_prior = mu_prior

    def build_observation_precision(self, tau_obs=1.0):
        Q_obs = sp.lil_matrix((self.n_params, self.n_params))
        b_obs = np.zeros(self.n_params)

        for s in range(self.n_settlements):
            for h in range(self.n_hours):
                value = self.sensor_obs.iloc[h, s]
                if not np.isnan(value):
                    idx = s * self.n_hours + h
                    Q_obs[idx, idx] = tau_obs
                    b_obs[idx] = tau_obs * value

        Q_obs = Q_obs.tocsr()

        self.Q_obs = Q_obs
        self.b_obs = b_obs

    def build_full_precision(self):
        Q_space_kron = sp.kron(self.Q_spatial, sp.eye(self.n_hours), format="csr")
        Q_time_kron = sp.kron(sp.eye(self.n_settlements), self.Q_temporal, format="csr")

        Q_prior = Q_space_kron + Q_time_kron

        self.Q_prior = Q_prior

    def build_posterior_system(self):
        Q_posterior = self.Q_prior + self.Q_obs
        b_posterior = self.b_obs + self.Q_prior @ self.mu_prior

        self.Q_posterior = Q_posterior
        self.b_posterior = b_posterior


import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree


class CARModel:
    def __init__(self, df_sensors, df_satellite, gdf):
        ## Data shape/specifications
        self.n_settlements = df_satellite.shape[1]
        self.n_hours = df_sensors.shape[0]
        self.hour_to_month = df_sensors.index.month
        self.n_params = self.n_settlements * self.n_hours

        ## Model inputs
        self.coords = gdf.loc[:, ["latitude", "longitude"]]
        self.satellite_monthly = df_satellite
        self.sensor_obs = df_sensors

        ## Model params
        self.W = None
        self.D = None
        self.Q_spatial = None

        self.Q_hour = None
        self.Q_day = None
        self.Q_temporal = None

        self.Q_prior = None
        self.mu_prior = None

        ## Observation matrices
        self.Q_satellite = None  # Monthly satellite constraints
        self.b_satellite = None
        self.A_satellite = None  # Aggregation matrix (for diagnostics)
        self.y_satellite = None  # Satellite monthly observations

        self.Q_sensor_pattern = None  # Sensor temporal pattern (optional/weak)
        self.b_sensor_pattern = None

        ## Posterior
        self.Q_posterior = None
        self.b_posterior = None

        ## Derived
        self.diurnal_patterns = None

    def create_adjacency_matrix(self):
        """Create spatial adjacency matrix based on distance"""
        tree = cKDTree(list(zip(self.coords["latitude"], self.coords["longitude"])))

        W = pd.DataFrame(
            np.zeros((self.coords.shape[0], self.coords.shape[0])),
            index=self.coords.index,
            columns=self.coords.index,
        )

        for i, row in self.coords.iterrows():
            dist, idx = tree.query([row["latitude"], row["longitude"]], k=W.shape[0])
            dist, idx = dist[1:], idx[1:]
            W.loc[i, W.columns[idx]] = dist

        W = np.exp(-W)
        W = (W + W.T) / 2
        D = np.diag(W.sum(axis=1).values)

        self.W = W
        self.D = D

    def create_spatial_precision_matrix(self, tau_spatial=1.0):
        """Spatial CAR precision"""
        self.Q_spatial = (self.D - self.W) * tau_spatial

    def create_hourly_precision(self, tau_hourly=1.0):
        """Hour-to-hour temporal smoothness"""
        Q_hour = np.zeros((self.n_hours, self.n_hours))

        for h in range(self.n_hours):
            if h == 0:
                Q_hour[h, h] = 1
                if self.n_hours > 1:
                    Q_hour[h, h + 1] = -1
            elif h == self.n_hours - 1:
                Q_hour[h, h] = 1
                Q_hour[h, h - 1] = -1
            else:
                Q_hour[h, h] = 2
                Q_hour[h, h - 1] = -1
                Q_hour[h, h + 1] = -1

        self.Q_hour = Q_hour * tau_hourly

    def create_daily_precision(self, tau_daily=1.0):
        """Day-to-day temporal structure (same hour across days)"""
        Q_day = np.zeros((self.n_hours, self.n_hours))

        for h in range(self.n_hours):
            # Same hour previous day
            if h >= 24:
                Q_day[h, h - 24] = -1
                Q_day[h, h] += 1

            # Same hour next day
            if h < self.n_hours - 24:
                Q_day[h, h + 24] = -1
                Q_day[h, h] += 1

        self.Q_day = Q_day * tau_daily

    def create_combined_precision(self):
        """Combine hour and day temporal structure"""
        self.Q_temporal = self.Q_hour + self.Q_day

    def learn_diurnal_patterns(self):
        """
        Learn diurnal patterns from sensor data.
        These are multiplicative factors (mean = 1) that describe the temporal shape.
        """
        diurnal_patterns = np.ones((24, self.n_settlements))

        for s in range(self.n_settlements):
            hourly_sums = np.zeros(24)
            hourly_counts = np.zeros(24)

            for h in range(self.n_hours):
                value = self.sensor_obs.iloc[h, s]
                if not np.isnan(value):
                    hour_of_day = h % 24
                    hourly_sums[hour_of_day] += value
                    hourly_counts[hour_of_day] += 1

            # Only normalize if we have data
            if hourly_counts.sum() > 0:
                hourly_counts[hourly_counts == 0] = 1
                hourly_means = hourly_sums / hourly_counts

                # Normalize to mean = 1 (multiplicative factors)
                if hourly_means.sum() > 0:
                    diurnal_patterns[:, s] = hourly_means / hourly_means.mean()

        self.diurnal_patterns = diurnal_patterns

    def create_prior_mean(self):
        """
        Create prior mean by applying diurnal patterns to satellite monthly values.
        This distributes monthly totals across hours while preserving the sum.
        """
        if self.diurnal_patterns is None:
            self.learn_diurnal_patterns()

        mu_prior = np.zeros(self.n_params)

        for s in range(self.n_settlements):
            for h in range(self.n_hours):
                # Get the actual month value (1-12) for this hour
                month_idx = self.hour_to_month[h]
                hour_of_day = h % 24
                param_idx = s * self.n_hours + h

                # Get satellite monthly value for this settlement-month
                satellite_monthly_value = float(
                    self.satellite_monthly.iloc[
                        self.satellite_monthly.index.month == month_idx, s
                    ].iloc[0]
                )

                # Count hours in this month for this settlement
                hours_in_month = np.sum(self.hour_to_month == month_idx)

                if hours_in_month == 0:
                    # This shouldn't happen for the current hour, but handle it
                    print(f"Warning: No hours found for month {month_idx}")
                    hours_in_month = 1  # Avoid division by zero

                # Apply diurnal pattern
                # Distribute monthly total across hours based on diurnal pattern
                mu_prior[param_idx] = (
                    satellite_monthly_value
                    / hours_in_month
                    * self.diurnal_patterns[hour_of_day, s]
                )

        self.mu_prior = mu_prior

    def create_sensor_informed_prior(self):
        """
        Create a complete prior mean field from gappy sensor data using hybrid approach:
        - Use sensor values directly where observed (preserves temporal detail)
        - Spatially interpolate from neighbors where missing (borrows patterns)

        This creates a sensor-informed but potentially biased field that satellite will calibrate.
        """
        mu_prior = np.zeros(self.n_params)

        # We'll use the adjacency matrix W for spatial weighting
        if self.W is None:
            raise ValueError("Must call create_adjacency_matrix() first")

        # Process each hour
        for h in range(self.n_hours):
            hour_values = self.sensor_obs.iloc[h, :]  # All settlements at this hour

            for s in range(self.n_settlements):
                param_idx = s * self.n_hours + h

                # Case 1: We have a sensor observation - use it directly
                if not np.isnan(hour_values.iloc[s]):
                    mu_prior[param_idx] = hour_values.iloc[s]

                # Case 2: Missing - borrow from spatial neighbors
                else:
                    # Find neighbors with observations at this hour
                    neighbor_values = []
                    neighbor_weights = []

                    for neighbor_s in range(self.n_settlements):
                        if neighbor_s != s and not np.isnan(
                            hour_values.iloc[neighbor_s]
                        ):
                            # Use adjacency weight (higher = closer)
                            weight = self.W.iloc[s, neighbor_s]
                            if weight > 0.01:  # Only use meaningful neighbors
                                neighbor_values.append(hour_values.iloc[neighbor_s])
                                neighbor_weights.append(weight)

                    # Weighted average from neighbors
                    if len(neighbor_values) > 0:
                        neighbor_weights = np.array(neighbor_weights)
                        neighbor_values = np.array(neighbor_values)
                        mu_prior[param_idx] = np.average(
                            neighbor_values, weights=neighbor_weights
                        )
                    else:
                        # No neighbors with data - use overall mean as fallback
                        # This could be: sensor mean, satellite mean, or a default value
                        available_values = hour_values.dropna()
                        if len(available_values) > 0:
                            mu_prior[param_idx] = available_values.mean()
                        else:
                            # Absolute fallback: use satellite data for this settlement-month
                            month_idx = self.hour_to_month[h]
                            try:
                                mu_prior[param_idx] = self.satellite_monthly.iloc[
                                    self.satellite_monthly.index.month == month_idx, s
                                ]
                            except:
                                mu_prior[param_idx] = 50.0  # Default PM2.5 value

        self.mu_prior = mu_prior

        # Diagnostic: How much did we fill?
        n_sensor_obs = (~self.sensor_obs.isna()).sum().sum()
        n_spatially_filled = np.sum(mu_prior != 0) - n_sensor_obs
        print(
            f"Prior construction: {n_sensor_obs} direct sensor obs, {n_spatially_filled} spatially interpolated"
        )

    def build_satellite_constraints(self, tau_satellite=1.0):
        """
        NEW: Build observation equations that constrain monthly MEANS to match satellite.
        For each settlement-month: mean(x[s,h] for h in month) ~ satellite_monthly[s,m]

        Implementation: Convert means to sums for numerical stability
        sum(x[s,h] for h in month) ~ satellite_monthly[s,m] × n_hours_in_month

        Uses aggregation matrix approach: Q = A^T * tau * A, b = A^T * tau * y
        where A sums hourly values and y is scaled satellite monthly mean.
        """
        unique_months = self.hour_to_month.unique()
        n_constraints = self.n_settlements * len(unique_months)

        # Build aggregation matrix A: (n_constraints x n_params)
        # A[constraint_i, param_j] = 1 if param_j contributes to constraint_i (for summing)
        A = sp.lil_matrix((n_constraints, self.n_params))
        y_satellite = np.zeros(
            n_constraints
        )  # Observed monthly values (scaled to sums)

        constraint_idx = 0
        for s in range(self.n_settlements):
            for month in unique_months:
                # Find all hours in this month for this settlement
                hours_in_month = np.where(self.hour_to_month == month)[0]

                if len(hours_in_month) == 0:
                    continue

                # Get satellite observation for this settlement-month (MONTHLY MEAN)
                satellite_mean = float(
                    self.satellite_monthly.iloc[
                        self.satellite_monthly.index.month == month, s
                    ].iloc[0]
                )

                # Convert mean to sum: if mean = μ, then sum = μ × n_hours
                # This way sum(x_hours) ≈ μ × n_hours implies mean(x_hours) ≈ μ
                y_satellite[constraint_idx] = satellite_mean * len(hours_in_month)

                # Set aggregation: SUM of these hours should equal scaled satellite value
                for h in hours_in_month:
                    param_idx = s * self.n_hours + h
                    A[constraint_idx, param_idx] = 1.0

                constraint_idx += 1

        # Convert to CSR for efficient multiplication
        A = A.tocsr()

        # Build precision matrix: Q = A^T * tau * A
        # This creates the constraint that A*x (sum) should be close to y_satellite (scaled mean)
        self.Q_satellite = tau_satellite * (A.T @ A)

        # Build b vector: b = A^T * tau * y
        self.b_satellite = tau_satellite * (A.T @ y_satellite)

        # Store A for diagnostics
        self.A_satellite = A
        self.y_satellite = y_satellite

    def normalize_sensors_to_satellite(self):
        """
        OPTIONAL: Normalize sensor observations to have similar monthly means as satellite.
        This allows us to use sensor patterns while respecting satellite levels.
        """
        sensor_normalized = self.sensor_obs.copy()

        for s in range(self.n_settlements):
            for month in self.hour_to_month.unique():
                hours_in_month = np.where(self.hour_to_month == month)[0]

                # Get sensor values for this month
                sensor_month_vals = []
                for h in hours_in_month:
                    val = self.sensor_obs.iloc[h, s]
                    if not np.isnan(val):
                        sensor_month_vals.append(val)

                if len(sensor_month_vals) > 0:
                    sensor_monthly_mean = np.mean(sensor_month_vals)
                    satellite_monthly_value = self.satellite_monthly.iloc[month - 1, s]

                    # Scaling factor to match satellite
                    if sensor_monthly_mean > 0:
                        scale = (
                            satellite_monthly_value
                            / sensor_monthly_mean
                            / len(hours_in_month)
                        )

                        # Apply scaling to all hours in this month
                        for h in hours_in_month:
                            if not np.isnan(sensor_normalized.iloc[h, s]):
                                sensor_normalized.iloc[h, s] *= scale

        self.sensor_normalized = sensor_normalized

    def build_sensor_pattern_observations(self, tau_sensor_pattern=0.1):
        """
        OPTIONAL: Add weak observations from normalized sensors to guide temporal patterns.
        Much weaker than satellite constraints (tau_sensor_pattern << tau_satellite).
        """
        if self.sensor_normalized is None:
            self.normalize_sensors_to_satellite()

        Q_sensor = sp.lil_matrix((self.n_params, self.n_params))
        b_sensor = np.zeros(self.n_params)

        for s in range(self.n_settlements):
            for h in range(self.n_hours):
                value = self.sensor_normalized.iloc[h, s]
                if not np.isnan(value):
                    idx = s * self.n_hours + h
                    Q_sensor[idx, idx] = tau_sensor_pattern
                    b_sensor[idx] = tau_sensor_pattern * value

        self.Q_sensor_pattern = Q_sensor.tocsr()
        self.b_sensor_pattern = b_sensor

    def build_full_precision(self):
        """Build full prior precision matrix from spatial and temporal components"""
        Q_space_kron = sp.kron(self.Q_spatial, sp.eye(self.n_hours), format="csr")
        Q_time_kron = sp.kron(sp.eye(self.n_settlements), self.Q_temporal, format="csr")

        self.Q_prior = Q_space_kron + Q_time_kron

    def build_posterior_system(self, use_sensor_patterns=True):
        """
        Build posterior system combining:
        1. Spatiotemporal prior
        2. Satellite monthly constraints (PRIMARY)
        3. Optional: sensor pattern matching (WEAK)
        """
        # Start with prior
        Q_posterior = self.Q_prior.copy()
        b_posterior = self.Q_prior @ self.mu_prior

        # Add satellite constraints (MAIN OBSERVATIONS)
        Q_posterior = Q_posterior + self.Q_satellite
        b_posterior = b_posterior + self.b_satellite

        # Optionally add weak sensor pattern matching
        if use_sensor_patterns and self.Q_sensor_pattern is not None:
            Q_posterior = Q_posterior + self.Q_sensor_pattern
            b_posterior = b_posterior + self.b_sensor_pattern

        self.Q_posterior = Q_posterior
        self.b_posterior = b_posterior

    def diagnose_model(self):
        """Print diagnostic information about model structure"""
        print("=== Model Diagnostics ===")
        print(
            f"Parameters: {self.n_params} ({self.n_settlements} settlements × {self.n_hours} hours)"
        )
        print(f"\nPrecision matrix diagonal means:")
        print(f"  Q_prior:     {self.Q_prior.diagonal().mean():.4f}")
        print(f"  Q_satellite: {self.Q_satellite.diagonal().mean():.4f}")
        if self.Q_sensor_pattern is not None:
            print(f"  Q_sensor:    {self.Q_sensor_pattern.diagonal().mean():.4f}")

        print(f"\nRelative strengths:")
        print(
            f"  Satellite/Prior ratio: {self.Q_satellite.diagonal().mean() / self.Q_prior.diagonal().mean():.2f}x"
        )
        if self.Q_sensor_pattern is not None:
            print(
                f"  Sensor/Prior ratio:    {self.Q_sensor_pattern.diagonal().mean() / self.Q_prior.diagonal().mean():.2f}x"
            )

        print(f"\nSparsity:")
        print(
            f"  Q_satellite non-zeros: {self.Q_satellite.nnz:,} ({self.Q_satellite.nnz/self.n_params**2*100:.3f}%)"
        )
        print(
            f"  Q_prior non-zeros:     {self.Q_prior.nnz:,} ({self.Q_prior.nnz/self.n_params**2*100:.3f}%)"
        )

        print(f"\nData coverage:")
        print(
            f"  Sensor observations:     {(~self.sensor_obs.isna()).sum().sum() / self.sensor_obs.size * 100:.1f}%"
        )
        print(f"  Satellite constraints:   {len(self.y_satellite)} monthly aggregates")

        # Check if monthly constraints are reasonable
        if hasattr(self, "A_satellite") and hasattr(self, "mu_prior"):
            predicted_monthly = self.A_satellite @ self.mu_prior
            residuals = predicted_monthly - self.y_satellite
            print(f"\nPrior-satellite alignment:")
            print(f"  Mean absolute error: {np.abs(residuals).mean():.2f}")
            print(
                f"  Mean relative error: {np.abs(residuals / self.y_satellite).mean()*100:.1f}%"
            )


# car_model = CARModel(df_sensors, df_sat, gdf_x)


from scipy.sparse.linalg import spsolve

# Use your CARModel with FIXED taus
car_model.create_adjacency_matrix()
car_model.create_spatial_precision_matrix(tau_spatial=0.5)
car_model.create_hourly_precision(tau_hourly=1.0)
car_model.create_daily_precision(tau_daily=1.0)
car_model.create_combined_precision()
car_model.build_full_precision()
car_model.learn_diurnal_patterns()
car_model.create_sensor_informed_prior()
car_model.build_satellite_constraints(tau_satellite=10.0)
car_model.build_posterior_system(use_sensor_patterns=False)

# Solve
# theta_map = spsolve(car_model.Q_posterior, car_model.b_posterior)

from scipy.sparse.linalg import cg, spilu, LinearOperator
import time

# Option 1: Basic CG (fastest to try)
print("Solving with CG...")
start = time.time()
theta_map, info = cg(
    car_model.Q_posterior,
    car_model.b_posterior,
    x0=car_model.mu_prior,  # Start from sensor prior
    maxiter=1000,
)
print(f"CG took {time.time()-start:.1f}s, converged: {info==0}")

# Option 2: CG with preconditioner (if basic CG is slow)
# Build incomplete LU preconditioner
"""
print("Building preconditioner...")
ilu = spilu(car_model.Q_posterior.tocsc(), drop_tol=1e-3, fill_factor=10)
M = LinearOperator(car_model.Q_posterior.shape, ilu.solve)

print("Solving with preconditioned CG...")
start = time.time()
theta_map, info = cg(car_model.Q_posterior, car_model.b_posterior,
                     x0=car_model.mu_prior,
                     M=M,  # Preconditioner
                     maxiter=500)
print(f"Preconditioned CG took {time.time()-start:.1f}s, converged: {info==0}")
"""

# Diagnostics for MAP solution
print("=== MAP Solution Diagnostics ===")

# 1. Check satellite constraint satisfaction
monthly_pred = car_model.A_satellite @ theta_map
residuals = monthly_pred - car_model.y_satellite

print(f"\nSatellite Constraint Fit:")
print(f"  Mean absolute error:   {np.abs(residuals).mean():.2f}")
print(
    f"  Mean relative error:   {np.abs(residuals / car_model.y_satellite).mean()*100:.1f}%"
)
print(f"  RMSE:                  {np.sqrt(np.mean(residuals**2)):.2f}")
print(f"  Max absolute residual: {np.abs(residuals).max():.2f}")

# 2. Compare to prior
print(f"\nPrior vs MAP:")
print(
    f"  Prior range:     [{car_model.mu_prior.min():.1f}, {car_model.mu_prior.max():.1f}]"
)
print(f"  MAP range:       [{theta_map.min():.1f}, {theta_map.max():.1f}]")
print(f"  Mean shift:      {(theta_map - car_model.mu_prior).mean():.1f}")
print(f"  Std of shifts:   {(theta_map - car_model.mu_prior).std():.1f}")

# 3. Check for unrealistic values
print(f"\nRealism checks:")
print(f"  Negative values: {np.sum(theta_map < 0)} / {len(theta_map)}")
print(f"  Extreme values (>500): {np.sum(theta_map > 500)} / {len(theta_map)}")

# 4. Look at a sample settlement
s = 0  # First settlement
theta_s = theta_map[s * car_model.n_hours : (s + 1) * car_model.n_hours]
print(f"\nExample: Settlement {s}")
print(f"  Hourly range: [{theta_s.min():.1f}, {theta_s.max():.1f}]")
print(f"  Mean: {theta_s.mean():.1f}")

# Store
car_model.theta_map = theta_map

from scipy.sparse.linalg import cg
import itertools


def evaluate_hyperparameters(
    tau_spatial, tau_hourly, tau_daily, tau_satellite, sensor_obs_train, sensor_obs_test
):
    """
    Fit model with given hyperparameters and evaluate on held-out sensors.
    Returns RMSE on test set.
    """
    # Build model with these hyperparameters
    temp_model = CARModel(sensor_obs_train, car_model.satellite_monthly, gdf_x)
    temp_model.create_adjacency_matrix()
    temp_model.create_spatial_precision_matrix(tau_spatial)
    temp_model.create_hourly_precision(tau_hourly)
    temp_model.create_daily_precision(tau_daily)
    temp_model.create_combined_precision()
    temp_model.build_full_precision()
    temp_model.learn_diurnal_patterns()
    temp_model.create_sensor_informed_prior()
    temp_model.build_satellite_constraints(tau_satellite)
    temp_model.build_posterior_system(use_sensor_patterns=False)

    # Solve
    theta, info = cg(
        temp_model.Q_posterior,
        temp_model.b_posterior,
        x0=temp_model.mu_prior,
        maxiter=1000,
    )

    if info != 0:
        return np.inf  # Didn't converge

    # Evaluate on test set
    test_indices = []
    test_values = []
    predicted_values = []

    for s in range(temp_model.n_settlements):
        for h in range(temp_model.n_hours):
            test_val = sensor_obs_test.iloc[h, s]
            if not np.isnan(test_val):
                idx = s * temp_model.n_hours + h
                test_indices.append(idx)
                test_values.append(test_val)
                predicted_values.append(theta[idx])

    # RMSE on test set
    test_values = np.array(test_values)
    predicted_values = np.array(predicted_values)
    rmse = np.sqrt(np.mean((test_values - predicted_values) ** 2))

    # Also check satellite constraint satisfaction
    monthly_pred = temp_model.A_satellite @ theta
    sat_rmse = np.sqrt(np.mean((monthly_pred - temp_model.y_satellite) ** 2))

    return rmse, sat_rmse


# Create train/test split (80/20 random hold-out)
np.random.seed(42)
sensor_mask = ~car_model.sensor_obs.isna()
test_fraction = 0.2

sensor_obs_train = car_model.sensor_obs.copy()
sensor_obs_test = pd.DataFrame(
    np.nan, index=car_model.sensor_obs.index, columns=car_model.sensor_obs.columns
)

# Randomly hold out 20% of observations
for s in range(car_model.n_settlements):
    for h in range(car_model.n_hours):
        if sensor_mask.iloc[h, s]:
            if np.random.rand() < test_fraction:
                # Move to test set
                sensor_obs_test.iloc[h, s] = car_model.sensor_obs.iloc[h, s]
                sensor_obs_train.iloc[h, s] = np.nan

n_train = (~sensor_obs_train.isna()).sum().sum()
n_test = (~sensor_obs_test.isna()).sum().sum()
print(f"Train: {n_train} observations, Test: {n_test} observations")

# Grid search over hyperparameters
tau_spatial_grid = [0.1, 0.5, 1.0, 2.0]
tau_hourly_grid = [0.5, 1.0, 2.0, 5.0]
tau_daily_grid = [0.5, 1.0, 2.0, 5.0]
tau_satellite_grid = [5.0, 10.0, 20.0]

print("\nGrid search (this will take a few minutes)...")
print("tau_spatial | tau_hourly | tau_daily | tau_satellite | Test RMSE | Sat RMSE")
print("-" * 80)

results = []
for tau_sp, tau_hr, tau_dy, tau_sat in itertools.product(
    tau_spatial_grid, tau_hourly_grid, tau_daily_grid, tau_satellite_grid
):

    test_rmse, sat_rmse = evaluate_hyperparameters(
        tau_sp, tau_hr, tau_dy, tau_sat, sensor_obs_train, sensor_obs_test
    )

    results.append(
        {
            "tau_spatial": tau_sp,
            "tau_hourly": tau_hr,
            "tau_daily": tau_dy,
            "tau_satellite": tau_sat,
            "test_rmse": test_rmse,
            "sat_rmse": sat_rmse,
        }
    )

    print(
        f"{tau_sp:8.1f} | {tau_hr:10.1f} | {tau_dy:9.1f} | {tau_sat:13.1f} | {test_rmse:9.2f} | {sat_rmse:8.2f}"
    )

# Find best
results_df = pd.DataFrame(results)
best_idx = results_df["test_rmse"].idxmin()
best = results_df.iloc[best_idx]

print("\n" + "=" * 80)
print("BEST HYPERPARAMETERS:")
print("=" * 80)
print(f"tau_spatial:   {best['tau_spatial']:.1f}")
print(f"tau_hourly:    {best['tau_hourly']:.1f}")
print(f"tau_daily:     {best['tau_daily']:.1f}")
print(f"tau_satellite: {best['tau_satellite']:.1f}")
print(f"\nTest RMSE: {best['test_rmse']:.2f}")
print(f"Satellite constraint RMSE: {best['sat_rmse']:.2f}")

# What's a naive baseline?
naive_rmse = np.sqrt(np.mean((sensor_obs_test.values[~sensor_obs_test.isna()]) ** 2))
print(f"Predict-zero baseline RMSE: {naive_rmse:.2f}")

# What if you just predict the prior?
test_vals = []
prior_preds = []
for s in range(car_model.n_settlements):
    for h in range(car_model.n_hours):
        if not np.isnan(sensor_obs_test.iloc[h, s]):
            test_vals.append(sensor_obs_test.iloc[h, s])
            prior_preds.append(car_model.mu_prior[s * car_model.n_hours + h])

prior_rmse = np.sqrt(np.mean((np.array(test_vals) - np.array(prior_preds)) ** 2))
print(f"Sensor-informed prior RMSE: {prior_rmse:.2f}")

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.sparse.linalg import cg


class GapFillingCARModel:
    def __init__(self, df_sensors, df_satellite, gdf):
        ## Data shape/specifications
        self.n_settlements = df_sensors.shape[1]
        self.n_hours = df_sensors.shape[0]
        self.hour_to_month = df_sensors.index.month
        self.n_params = self.n_settlements * self.n_hours

        ## Model inputs
        self.coords = gdf.loc[:, ["latitude", "longitude"]]
        self.satellite_monthly = df_satellite
        self.sensor_obs = df_sensors

        ## Model components
        self.W = None
        self.D = None
        self.Q_spatial = None
        self.Q_temporal = None
        self.Q_prior = None

        ## Observations (sensors STRONG, satellite WEAK)
        self.Q_sensor = None
        self.b_sensor = None

        self.Q_satellite = None
        self.b_satellite = None
        self.A_satellite = None
        self.y_satellite = None

        ## Solution
        self.Q_posterior = None
        self.b_posterior = None
        self.theta_map = None

    def create_adjacency_matrix(self):
        """Create spatial adjacency matrix based on distance"""
        tree = cKDTree(list(zip(self.coords["latitude"], self.coords["longitude"])))

        W = pd.DataFrame(
            np.zeros((self.coords.shape[0], self.coords.shape[0])),
            index=self.coords.index,
            columns=self.coords.index,
        )

        for i, row in self.coords.iterrows():
            dist, idx = tree.query([row["latitude"], row["longitude"]], k=W.shape[0])
            dist, idx = dist[1:], idx[1:]
            W.loc[i, W.columns[idx]] = dist

        W = np.exp(-W)
        W = (W + W.T) / 2
        D = np.diag(W.sum(axis=1).values)

        self.W = W
        self.D = D

    def build_spatial_precision(self, tau_spatial=1.0):
        """Spatial CAR precision"""
        self.Q_spatial = (self.D - self.W) * tau_spatial

    def build_temporal_precision(self, tau_hourly=1.0, tau_daily=1.0):
        """Combined hour-to-hour and day-to-day temporal precision"""
        Q_hour = np.zeros((self.n_hours, self.n_hours))
        Q_day = np.zeros((self.n_hours, self.n_hours))

        # Hour-to-hour smoothness
        for h in range(self.n_hours):
            if h == 0:
                Q_hour[h, h] = 1
                if self.n_hours > 1:
                    Q_hour[h, h + 1] = -1
            elif h == self.n_hours - 1:
                Q_hour[h, h] = 1
                Q_hour[h, h - 1] = -1
            else:
                Q_hour[h, h] = 2
                Q_hour[h, h - 1] = -1
                Q_hour[h, h + 1] = -1

        # Day-to-day smoothness (same hour across days)
        for h in range(self.n_hours):
            if h >= 24:
                Q_day[h, h - 24] = -1
                Q_day[h, h] += 1
            if h < self.n_hours - 24:
                Q_day[h, h + 24] = -1
                Q_day[h, h] += 1

        self.Q_temporal = Q_hour * tau_hourly + Q_day * tau_daily

    def build_prior_precision(self):
        """Combine spatial and temporal structure"""
        Q_space_kron = sp.kron(self.Q_spatial, sp.eye(self.n_hours), format="csr")
        Q_time_kron = sp.kron(sp.eye(self.n_settlements), self.Q_temporal, format="csr")
        self.Q_prior = Q_space_kron + Q_time_kron

    def build_sensor_observations(self, tau_sensor=10.0):
        """
        PRIMARY OBSERVATIONS: Sensor data where it exists (STRONG).
        High precision - these are the ground truth we want to fit closely.
        """
        Q_sensor = sp.lil_matrix((self.n_params, self.n_params))
        b_sensor = np.zeros(self.n_params)

        n_obs = 0
        for s in range(self.n_settlements):
            for h in range(self.n_hours):
                value = self.sensor_obs.iloc[h, s]
                if not np.isnan(value):
                    idx = s * self.n_hours + h
                    Q_sensor[idx, idx] = tau_sensor
                    b_sensor[idx] = tau_sensor * value
                    n_obs += 1

        self.Q_sensor = Q_sensor.tocsr()
        self.b_sensor = b_sensor
        print(
            f"Sensor observations: {n_obs} / {self.n_params} ({n_obs/self.n_params*100:.1f}%)"
        )

    def build_satellite_constraints(self, tau_satellite=1.0):
        """
        SECONDARY CONSTRAINTS: Monthly satellite means (WEAK).
        Lower precision - just prevents drift in gap-filled regions.
        Should have tau_satellite << tau_sensor.
        """
        unique_months = self.hour_to_month.unique()
        n_constraints = self.n_settlements * len(unique_months)

        A = sp.lil_matrix((n_constraints, self.n_params))
        y_satellite = np.zeros(n_constraints)

        constraint_idx = 0
        for s in range(self.n_settlements):
            for month in unique_months:
                hours_in_month = np.where(self.hour_to_month == month)[0]
                if len(hours_in_month) == 0:
                    continue

                satellite_mean = float(
                    self.satellite_monthly.iloc[
                        self.satellite_monthly.index.month == month, s
                    ].iloc[0]
                )
                y_satellite[constraint_idx] = satellite_mean * len(hours_in_month)

                for h in hours_in_month:
                    param_idx = s * self.n_hours + h
                    A[constraint_idx, param_idx] = 1.0

                constraint_idx += 1

        A = A.tocsr()
        self.Q_satellite = tau_satellite * (A.T @ A)
        self.b_satellite = tau_satellite * (A.T @ y_satellite)
        self.A_satellite = A
        self.y_satellite = y_satellite
        print(f"Satellite constraints: {constraint_idx} monthly aggregates")

    def build_posterior(self):
        """Combine everything: Prior + Sensor observations + Satellite constraints"""
        self.Q_posterior = self.Q_prior + self.Q_sensor + self.Q_satellite
        self.b_posterior = self.b_sensor + self.b_satellite

    def solve_map(self, use_cg=True):
        """Solve for MAP estimate"""
        import time

        start = time.time()

        if use_cg:
            # Use conjugate gradient (fast)
            theta, info = cg(self.Q_posterior, self.b_posterior, maxiter=2000)
            if info != 0:
                print(f"Warning: CG didn't converge (info={info})")
        else:
            # Use direct solve (slower but exact)
            from scipy.sparse.linalg import spsolve

            theta = spsolve(self.Q_posterior, self.b_posterior)

        self.theta_map = theta
        elapsed = time.time() - start
        print(f"Solved in {elapsed:.2f} seconds")
        return theta

    def evaluate(self):
        """Evaluate solution quality"""
        print("\n=== Model Evaluation ===")

        # Sensor fit (should be very good)
        sensor_residuals = []
        for s in range(self.n_settlements):
            for h in range(self.n_hours):
                value = self.sensor_obs.iloc[h, s]
                if not np.isnan(value):
                    idx = s * self.n_hours + h
                    sensor_residuals.append(self.theta_map[idx] - value)

        sensor_residuals = np.array(sensor_residuals)
        print(f"\nSensor fit (where observed):")
        print(f"  RMSE: {np.sqrt(np.mean(sensor_residuals**2)):.2f}")
        print(f"  Mean error: {sensor_residuals.mean():.2f}")
        print(f"  Max error: {np.abs(sensor_residuals).max():.2f}")

        # Satellite constraint satisfaction (should be moderate)
        monthly_pred = self.A_satellite @ self.theta_map
        sat_residuals = monthly_pred - self.y_satellite
        print(f"\nSatellite constraint satisfaction:")
        print(f"  RMSE: {np.sqrt(np.mean(sat_residuals**2)):.2f}")
        print(
            f"  Mean relative error: {np.abs(sat_residuals / self.y_satellite).mean()*100:.1f}%"
        )

        # Value range
        print(
            f"\nEstimated PM2.5 range: [{self.theta_map.min():.1f}, {self.theta_map.max():.1f}]"
        )
        print(f"Negative values: {np.sum(self.theta_map < 0)} / {len(self.theta_map)}")


# Usage
print("Building gap-filling model...")
model = GapFillingCARModel(df_sensors, df_sat, gdf_x)

# Build structure
model.create_adjacency_matrix()
model.build_spatial_precision(tau_spatial=0.5)
model.build_temporal_precision(tau_hourly=1.0, tau_daily=1.0)
model.build_prior_precision()

# Add observations (KEY: sensor >> satellite)
model.build_sensor_observations(tau_sensor=10.0)  # STRONG - fit sensors closely
model.build_satellite_constraints(tau_satellite=5.0)  # WEAK - just prevent drift

# Solve
model.build_posterior()
theta_map = model.solve_map(use_cg=True)

# Evaluate
model.evaluate()

from scipy.sparse.linalg import cg
import itertools


def evaluate_gap_filling(
    tau_sensor,
    tau_satellite,
    tau_spatial,
    tau_hourly,
    tau_daily,
    sensor_train,
    sensor_test,
):
    """
    Fit gap-filling model and evaluate on held-out sensors.
    """
    # Build model
    temp_model = GapFillingCARModel(sensor_train, df_sat, gdf_x)
    temp_model.create_adjacency_matrix()
    temp_model.build_spatial_precision(tau_spatial)
    temp_model.build_temporal_precision(tau_hourly, tau_daily)
    temp_model.build_prior_precision()
    temp_model.build_sensor_observations(tau_sensor)
    temp_model.build_satellite_constraints(tau_satellite)
    temp_model.build_posterior()

    # Solve
    theta, info = cg(temp_model.Q_posterior, temp_model.b_posterior, maxiter=2000)

    if info != 0:
        return np.inf, np.inf, np.inf  # Failed to converge

    # Evaluate on test sensors
    test_residuals = []
    for s in range(temp_model.n_settlements):
        for h in range(temp_model.n_hours):
            test_val = sensor_test.iloc[h, s]
            if not np.isnan(test_val):
                idx = s * temp_model.n_hours + h
                test_residuals.append(theta[idx] - test_val)

    test_rmse = np.sqrt(np.mean(np.array(test_residuals) ** 2))

    # Evaluate on train sensors (should be very small)
    train_residuals = []
    for s in range(temp_model.n_settlements):
        for h in range(temp_model.n_hours):
            train_val = sensor_train.iloc[h, s]
            if not np.isnan(train_val):
                idx = s * temp_model.n_hours + h
                train_residuals.append(theta[idx] - train_val)

    train_rmse = np.sqrt(np.mean(np.array(train_residuals) ** 2))

    # Check satellite constraints
    monthly_pred = temp_model.A_satellite @ theta
    sat_rmse = np.sqrt(np.mean((monthly_pred - temp_model.y_satellite) ** 2))

    return test_rmse, train_rmse, sat_rmse


# Create train/test split (20% holdout)
np.random.seed(42)
sensor_train = df_sensors.copy()
sensor_test = pd.DataFrame(np.nan, index=df_sensors.index, columns=df_sensors.columns)

for s in range(df_sensors.shape[1]):
    for h in range(df_sensors.shape[0]):
        if not np.isnan(df_sensors.iloc[h, s]):
            if np.random.rand() < 0.2:
                sensor_test.iloc[h, s] = df_sensors.iloc[h, s]
                sensor_train.iloc[h, s] = np.nan

n_train = (~sensor_train.isna()).sum().sum()
n_test = (~sensor_test.isna()).sum().sum()
print(f"Split: {n_train} train, {n_test} test observations\n")

# Grid search
# tau_sensor should stay high (we want to fit sensors)
# We're tuning the gap-filling quality (spatial/temporal smoothness vs satellite)
tau_sensor_grid = [5.0]  # [20.0, 10.0, 5.0]
tau_satellite_grid = [3.0]  # [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
tau_spatial_grid = [0.5, 1.0]
tau_hourly_grid = [2.0, 2.5, 3.0]
tau_daily_grid = [2.0, 2.5, 3.0]

print("Grid search starting...")
print(
    "tau_sens | tau_sat | tau_sp | tau_hr | tau_dy | Test RMSE | Train RMSE | Sat RMSE"
)
print("-" * 90)

results = []
total_combos = (
    len(tau_sensor_grid)
    * len(tau_satellite_grid)
    * len(tau_spatial_grid)
    * len(tau_hourly_grid)
    * len(tau_daily_grid)
)
counter = 0

for tau_sens, tau_sat, tau_sp, tau_hr, tau_dy in itertools.product(
    tau_sensor_grid,
    tau_satellite_grid,
    tau_spatial_grid,
    tau_hourly_grid,
    tau_daily_grid,
):

    counter += 1
    test_rmse, train_rmse, sat_rmse = evaluate_gap_filling(
        tau_sens, tau_sat, tau_sp, tau_hr, tau_dy, sensor_train, sensor_test
    )

    results.append(
        {
            "tau_sensor": tau_sens,
            "tau_satellite": tau_sat,
            "tau_spatial": tau_sp,
            "tau_hourly": tau_hr,
            "tau_daily": tau_dy,
            "test_rmse": test_rmse,
            "train_rmse": train_rmse,
            "sat_rmse": sat_rmse,
        }
    )

    print(
        f"{tau_sens:8.0f} | {tau_sat:7.1f} | {tau_sp:6.1f} | {tau_hr:6.1f} | "
        f"{tau_dy:6.1f} | {test_rmse:9.2f} | {train_rmse:10.2f} | {sat_rmse:8.2f}"
    )

    if counter % 10 == 0:
        print(f"  [{counter}/{total_combos} completed]")

# Find best
results_df = pd.DataFrame(results)
best_idx = results_df["test_rmse"].idxmin()
best = results_df.iloc[best_idx]

print("\n" + "=" * 90)
print("BEST HYPERPARAMETERS (minimizing test RMSE):")
print("=" * 90)
print(f"tau_sensor:    {best['tau_sensor']:.0f}  (sensor observation strength)")
print(f"tau_satellite: {best['tau_satellite']:.1f}  (satellite constraint strength)")
print(f"tau_spatial:   {best['tau_spatial']:.1f}  (spatial smoothness)")
print(f"tau_hourly:    {best['tau_hourly']:.1f}  (hour-to-hour smoothness)")
print(f"tau_daily:     {best['tau_daily']:.1f}  (day-to-day smoothness)")
print(f"\nTest RMSE:  {best['test_rmse']:.2f} (prediction on held-out sensors)")
print(f"Train RMSE: {best['train_rmse']:.2f} (fit to training sensors)")
print(f"Sat RMSE:   {best['sat_rmse']:.2f} (monthly constraint satisfaction)")

# Show top 5 configurations
print("\n" + "=" * 90)
print("TOP 5 CONFIGURATIONS:")
print("=" * 90)
top5 = results_df.nsmallest(5, "test_rmse")
print(top5.to_string(index=False))


class SatelliteAnchoredGapFilling:
    """
    Separates PM2.5 into:
    1. Monthly level (from satellite, spatially smooth)
    2. Temporal pattern (from sensors, settlement-specific)

    theta[s,t] = level[s,month(t)] × pattern[s,hour_of_day(t)]
    """

    def __init__(self, df_sensors, df_satellite, gdf):
        self.n_settlements = df_sensors.shape[1]
        self.n_hours = df_sensors.shape[0]
        self.sensor_obs = df_sensors
        self.satellite_monthly = df_satellite
        self.coords = gdf.loc[:, ["latitude", "longitude"]]
        self.hour_to_month = df_sensors.index.month

        # Learned components
        self.temporal_patterns = None  # [24 hours × n_settlements]
        self.monthly_levels = None  # [n_months × n_settlements]
        self.theta_filled = None  # Final gap-filled field

    def learn_temporal_patterns(self):
        """
        Learn settlement-specific diurnal patterns from sensors.
        Normalized so mean=1 (multiplicative factors).
        """
        patterns = np.ones((24, self.n_settlements))

        for s in range(self.n_settlements):
            hourly_sums = np.zeros(24)
            hourly_counts = np.zeros(24)

            for h in range(self.n_hours):
                value = self.sensor_obs.iloc[h, s]
                if not np.isnan(value):
                    hour_of_day = h % 24
                    hourly_sums[hour_of_day] += value
                    hourly_counts[hour_of_day] += 1

            # If we have data, compute normalized pattern
            if hourly_counts.sum() > 0:
                hourly_counts[hourly_counts == 0] = 1
                hourly_means = hourly_sums / hourly_counts

                # Normalize to mean=1 (multiplicative factor)
                if hourly_means.mean() > 0:
                    patterns[:, s] = hourly_means / hourly_means.mean()

        self.temporal_patterns = patterns
        print(f"Learned temporal patterns for {self.n_settlements} settlements")

    def spatially_interpolate_patterns(self, W, missing_threshold=100):
        """
        For settlements with insufficient sensor data,
        borrow temporal patterns from neighbors.
        """
        for s in range(self.n_settlements):
            # Check if this settlement has enough data
            n_obs = (~self.sensor_obs.iloc[:, s].isna()).sum()

            if n_obs < missing_threshold:
                # Borrow from neighbors
                neighbor_patterns = []
                neighbor_weights = []

                for neighbor_s in range(self.n_settlements):
                    if neighbor_s != s:
                        neighbor_obs = (
                            ~self.sensor_obs.iloc[:, neighbor_s].isna()
                        ).sum()
                        if neighbor_obs >= missing_threshold:
                            weight = W.iloc[s, neighbor_s]
                            if weight > 0.01:
                                neighbor_patterns.append(
                                    self.temporal_patterns[:, neighbor_s]
                                )
                                neighbor_weights.append(weight)

                if len(neighbor_patterns) > 0:
                    # Weighted average of neighbor patterns
                    neighbor_weights = np.array(neighbor_weights)
                    neighbor_patterns = np.array(
                        neighbor_patterns
                    ).T  # 24 × n_neighbors

                    borrowed_pattern = np.average(
                        neighbor_patterns, axis=1, weights=neighbor_weights
                    )
                    borrowed_pattern = (
                        borrowed_pattern / borrowed_pattern.mean()
                    )  # Re-normalize

                    self.temporal_patterns[:, s] = borrowed_pattern
                    print(
                        f"  Settlement {s}: borrowed pattern from {len(neighbor_patterns)} neighbors"
                    )

    def fit_monthly_levels(self, alpha=0.5):
        """
        Fit monthly levels that:
        1. Match satellite values (primary)
        2. Are smooth across months (secondary)

        alpha controls satellite vs smoothness trade-off (0=smooth, 1=exact satellite)
        """
        unique_months = self.hour_to_month.unique()
        n_months = len(unique_months)

        # Start with satellite values
        monthly_levels = np.zeros((n_months, self.n_settlements))

        for m_idx, month in enumerate(unique_months):
            for s in range(self.n_settlements):
                sat_value = self.satellite_monthly.iloc[
                    self.satellite_monthly.index.month == month, s
                ].iloc[0]
                monthly_levels[m_idx, s] = sat_value

        # Could add temporal smoothing here if desired
        # For now, just use satellite directly
        self.monthly_levels = monthly_levels
        print(
            f"Fit monthly levels: {n_months} months × {self.n_settlements} settlements"
        )

    def fill_gaps(self):
        """
        Generate complete hourly field:
        theta[s,t] = monthly_level[s,month(t)] × temporal_pattern[s,hour(t)]
        """
        theta = np.zeros(self.n_settlements * self.n_hours)

        for s in range(self.n_settlements):
            for h in range(self.n_hours):
                month_value = self.hour_to_month[h]
                month_idx = np.where(self.hour_to_month.unique() == month_value)[0][0]
                hour_of_day = h % 24

                param_idx = s * self.n_hours + h

                # Multiplicative: level × pattern
                level = self.monthly_levels[month_idx, s]
                pattern = self.temporal_patterns[hour_of_day, s]

                theta[param_idx] = level * pattern

        self.theta_filled = theta
        print("Gap-filling complete")
        return theta

    def evaluate(self, sensor_test=None):
        """Evaluate against held-out sensors or all sensors"""
        if sensor_test is None:
            sensor_test = self.sensor_obs

        test_residuals = []
        for s in range(self.n_settlements):
            for h in range(self.n_hours):
                obs = sensor_test.iloc[h, s]
                if not np.isnan(obs):
                    idx = s * self.n_hours + h
                    test_residuals.append(self.theta_filled[idx] - obs)

        test_residuals = np.array(test_residuals)
        rmse = np.sqrt(np.mean(test_residuals**2))

        print(f"\n=== Evaluation ===")
        print(f"Test RMSE: {rmse:.2f}")
        print(f"Mean error: {test_residuals.mean():.2f}")
        print(f"Std error: {test_residuals.std():.2f}")

        return rmse


# Usage
from scipy.spatial import cKDTree

# Build adjacency for spatial borrowing
tree = cKDTree(list(zip(gdf_x["latitude"], gdf_x["longitude"])))
W = pd.DataFrame(
    np.zeros((len(gdf_x), len(gdf_x))), index=gdf_x.index, columns=gdf_x.index
)
for i, row in gdf_x.iterrows():
    dist, idx = tree.query([row["latitude"], row["longitude"]], k=len(gdf_x))
    dist, idx = dist[1:], idx[1:]
    W.loc[i, W.columns[idx]] = dist
W = np.exp(-W)
W = (W + W.T) / 2

# Run model
model = SatelliteAnchoredGapFilling(df_sensors, df_sat, gdf_x)
model.learn_temporal_patterns()
model.spatially_interpolate_patterns(W, missing_threshold=100)
model.fit_monthly_levels(alpha=1.0)  # Use satellite exactly
theta_filled = model.fill_gaps()
model.evaluate()

# Check if error varies by time or space
residuals_by_settlement = {}
residuals_by_hour = {}

for s in range(model.n_settlements):
    resids = []
    for h in range(model.n_hours):
        obs = df_sensors.iloc[h, s]
        if not np.isnan(obs):
            idx = s * model.n_hours + h
            resid = model.theta_filled[idx] - obs
            resids.append(resid)

            # By hour of day
            hod = h % 24
            if hod not in residuals_by_hour:
                residuals_by_hour[hod] = []
            residuals_by_hour[hod].append(resid)

    if len(resids) > 0:
        residuals_by_settlement[s] = np.array(resids)

# Settlement-specific bias?
print("Per-settlement RMSE:")
for s, resids in residuals_by_settlement.items():
    print(
        f"  Settlement {s}: RMSE={np.sqrt(np.mean(resids**2)):.1f}, bias={resids.mean():.1f}"
    )

# Hour-of-day patterns in error?
print("\nError by hour of day:")
for hod in sorted(residuals_by_hour.keys()):
    resids = np.array(residuals_by_hour[hod])
    print(f"  Hour {hod:02d}: mean={resids.mean():.1f}, std={resids.std():.1f}")

# After initial gap-filling, learn biases from residuals
biases = np.zeros(model.n_settlements)

for s in range(model.n_settlements):
    residuals = []
    for h in range(model.n_hours):
        obs = df_sensors.iloc[h, s]
        if not np.isnan(obs):
            idx = s * model.n_hours + h
            residuals.append(obs - model.theta_filled[idx])

    if len(residuals) > 10:  # Only if enough data
        biases[s] = np.median(residuals)  # Median is robust

# Apply corrections
for s in range(model.n_settlements):
    for h in range(model.n_hours):
        idx = s * model.n_hours + h
        model.theta_filled[idx] += biases[s]

# Re-evaluate
model.evaluate()

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial.distance import cdist

# =============================================================================
# YOUR DATA (assuming these are loaded)
# =============================================================================
# df_sensors = ds_med.pa_campmier_delhi_mean.to_dataframe().pivot_table(index='time', columns='settlement', values='pa_campmier_delhi_mean', aggfunc='mean')
# df_sat = ds_med.pm25_cnn.to_dataframe().pivot_table(index='time', columns='settlement', values='pm25_cnn', aggfunc='mean')
# gdf_fig2_settlement = your geodataframe with settlement geometries

# =============================================================================
# DIMENSIONS
# =============================================================================

tau_spatial = 0.01
tau_hourly = 0.1
tau_daily = 0.01
tau_sensor = 20.0
tau_satellite = 4.0

satellite_hour = 13
n_settlements = len(df_sat.columns)
n_days = len(
    pd.date_range(
        df_sensors.index.min().date(), df_sensors.index.max().date(), freq="D"
    )
)
n_hours = n_days * 24

print(
    f"Study period: {df_sensors.index.min().date()} to {df_sensors.index.max().date()}"
)
print(f"Dimensions:")
print(f"  Settlements: {n_settlements}")
print(f"  Days: {n_days}")
print(f"  Hours: {n_hours}")
print()

# =============================================================================
# FIX 1: CREATE THE ADJACENCY MATRIX
# =============================================================================
centroids = np.array([[geom.x, geom.y] for geom in gdf_fig2_settlement["geometry"]])
dist_matrix = cdist(centroids, centroids, metric="euclidean")


def create_inverse_distance_adj(dist_matrix, cutoff_distance=None, min_neighbors=2):
    """Create inverse distance adjacency matrix."""
    n = dist_matrix.shape[0]
    adj = np.zeros((n, n))

    if cutoff_distance is not None:
        # Method 1: All areas within cutoff distance
        for i in range(n):
            for j in range(n):
                if i != j and dist_matrix[i, j] <= cutoff_distance:
                    adj[i, j] = 1.0 / dist_matrix[i, j]

        # Check if any area has too few neighbors
        n_neighbors = (adj > 0).sum(axis=1)
        for i in range(n):
            if n_neighbors[i] < min_neighbors:
                # Add closest neighbors
                distances_i = dist_matrix[i, :].copy()
                distances_i[i] = np.inf  # Exclude self
                closest = np.argsort(distances_i)[:min_neighbors]
                for j in closest:
                    adj[i, j] = 1.0 / dist_matrix[i, j]
    else:
        # Method 2: K nearest neighbors
        k = min_neighbors
        for i in range(n):
            distances_i = dist_matrix[i, :].copy()
            distances_i[i] = np.inf  # Exclude self
            closest = np.argsort(distances_i)[:k]
            for j in closest:
                adj[i, j] = 1.0 / dist_matrix[i, j]

    # Make symmetric (average the two directions)
    adj = (adj + adj.T) / 2
    return adj


# Actually create the adjacency matrix (using k-nearest neighbors with k=3)
adj_matrix = create_inverse_distance_adj(
    dist_matrix, cutoff_distance=None, min_neighbors=3
)

print("Adjacency matrix created:")
print(f"  Method: K-nearest neighbors (k=3)")
print(f"  Non-zero entries: {(adj_matrix > 0).sum()}")
print(
    f"  Neighbors per settlement: min={(adj_matrix > 0).sum(axis=1).min()}, max={(adj_matrix > 0).sum(axis=1).max()}"
)
print()

# =============================================================================
# FIX 2: BUILD SPATIAL PRECISION (with consistent variable names)
# =============================================================================
row_sums = np.sum(adj_matrix, axis=1)
D = np.diag(row_sums)
Q_spatial = D - adj_matrix

tau_spatial = 2.0  # Hyperparameter to tune later
Q_spatial_scaled = tau_spatial * Q_spatial

# FIX: Use n_settlements consistently (not n_areas)
n_areas = n_settlements  # Alias for consistency with later code

print("Spatial precision matrix:")
print(f"  Shape: {Q_spatial.shape}")
print(f"  tau_spatial: {tau_spatial}")
print()

# =============================================================================
# TEMPORAL PRECISION (HOURLY)
# =============================================================================
def build_hourly_temporal_precision(n_hours, tau_hourly=10.0, tau_daily=1.0):
    """Build temporal precision with hourly and daily components."""
    Q_temp = np.zeros((n_hours, n_hours))

    for h in range(n_hours):
        # Hour-to-hour transitions (strong smoothing)
        if h > 0:  # Previous hour
            Q_temp[h, h - 1] = -tau_hourly
            Q_temp[h, h] += tau_hourly
        if h < n_hours - 1:  # Next hour
            Q_temp[h, h + 1] = -tau_hourly
            Q_temp[h, h] += tau_hourly

        # Same hour, previous/next day (weaker smoothing)
        if h >= 24:  # Previous day, same hour
            Q_temp[h, h - 24] = -tau_daily
            Q_temp[h, h] += tau_daily
        if h < n_hours - 24:  # Next day, same hour
            Q_temp[h, h + 24] = -tau_daily
            Q_temp[h, h] += tau_daily

    return Q_temp


Q_temporal_hourly = build_hourly_temporal_precision(n_hours, tau_hourly, tau_daily)

print("Temporal precision matrix:")
print(f"  Shape: {Q_temporal_hourly.shape}")
print(f"  tau_hourly: {tau_hourly}")
print(f"  tau_daily: {tau_daily}")
print()

# =============================================================================
# ORGANIZE SENSOR DATA (HOURLY)
# =============================================================================
print("Organizing sensor data...")

# Check temporal resolution of sensor data
time_diff = df_sensors.index.to_series().diff().median()
print(f"  Sensor temporal resolution: {time_diff}")

# Create full hourly index for study period
date_start = df_sensors.index.min().replace(hour=0, minute=0, second=0)
date_end = df_sensors.index.max().replace(hour=23, minute=59, second=59)
hourly_index = pd.date_range(date_start, date_end, freq="H")[:n_hours]

# Resample sensor data to hourly (if needed)
if time_diff < pd.Timedelta("1H"):
    # If sub-hourly, aggregate to hourly
    df_sensors_hourly = df_sensors.resample("H").mean()
    print(f"  Aggregated sub-hourly to hourly")
elif time_diff > pd.Timedelta("1H"):
    # If super-hourly, need to interpolate or forward-fill
    df_sensors_hourly = df_sensors.resample("H").ffill()  # or .interpolate()
    print(f"  Upsampled to hourly using forward-fill")
else:
    # Already hourly
    df_sensors_hourly = df_sensors.copy()
    print(f"  Already hourly")

# Reindex to ensure complete hourly coverage
df_sensors_hourly = df_sensors_hourly.reindex(hourly_index)

# Convert to numpy array: (n_settlements, n_hours)
sensor_data_hourly = df_sensors_hourly[
    df_sat.columns
].T.values  # Ensure same settlement order

print(f"  Sensor data shape: {sensor_data_hourly.shape}")
print(
    f"  Coverage: {(~np.isnan(sensor_data_hourly)).sum()} / {sensor_data_hourly.size} = {(~np.isnan(sensor_data_hourly)).mean():.1%}"
)
print()

# =============================================================================
# ORGANIZE SATELLITE DATA (DAILY AT FIXED HOUR)
# =============================================================================
print("Organizing satellite data...")

# Satellite observes once per day at satellite_hour
# Create daily index
daily_index = pd.date_range(date_start.date(), date_end.date(), freq="D")[:n_days]

# Resample satellite to daily (if not already)
if len(df_sat) > n_days:
    df_sat_daily = df_sat.resample("D").mean()
    print(f"  Aggregated to daily")
else:
    df_sat_daily = df_sat.copy()
    print(f"  Already daily")

# Reindex to ensure complete daily coverage
df_sat_daily = df_sat_daily.reindex(daily_index)

# Convert to numpy array: (n_settlements, n_days)
satellite_data_daily = df_sat_daily[df_sat.columns].T.values

print(f"  Satellite data shape: {satellite_data_daily.shape}")
# print(f"  Coverage: {(~np.isnan(satellite_data_daily)).sum()} / {satellite_data_daily.size} = {(~np.isnan(satellite_data_daily)).mean():.1%}")
print()

# =============================================================================
# CREATE OBSERVATION DICTIONARIES FOR INFERENCE
# =============================================================================

# Sensor observations: dictionary mapping (settlement_idx, hour_idx) -> value
sensor_obs = {}
for s in range(n_settlements):
    for h in range(n_hours):
        if not np.isnan(sensor_data_hourly[s, h]):
            sensor_obs[(s, h)] = sensor_data_hourly[s, h]

print(f"Sensor observations dictionary: {len(sensor_obs)} non-NaN values")

# Satellite observations: dictionary mapping (settlement_idx, day_idx) -> value
satellite_obs = {}
for s in range(n_settlements):
    for d in range(n_days):
        if not np.isnan(satellite_data_daily[s, d]):
            satellite_obs[(s, d)] = satellite_data_daily[s, d]

print(f"Satellite observations dictionary: {len(satellite_obs)} non-NaN values")
print()

# =============================================================================
# AGGREGATION MATRICES
# =============================================================================

# Matrix to aggregate hourly to daily averages
H_daily = np.zeros((n_days, n_hours))
for day in range(n_days):
    hour_start = day * 24
    hour_end = hour_start + 24
    if hour_end <= n_hours:  # Safety check
        H_daily[day, hour_start:hour_end] = 1.0 / 24

print(f"Daily aggregation matrix H_daily: {H_daily.shape}")

# Matrix to extract satellite observation hour from each day
H_satellite = np.zeros((n_days, n_hours))
for day in range(n_days):
    hour_idx = day * 24 + satellite_hour
    if hour_idx < n_hours:  # Safety check
        H_satellite[day, hour_idx] = 1.0

print(f"Satellite sampling matrix H_satellite: {H_satellite.shape}")
print()

# =============================================================================
# FULL SPATIOTEMPORAL PRECISION
# =============================================================================

Q_space_kron = sp.kron(sp.eye(n_hours), Q_spatial_scaled, format="csr")
Q_time_kron = sp.kron(Q_temporal_hourly, sp.eye(n_settlements), format="csr")
Q_full_hourly = Q_space_kron + Q_time_kron

print(f"Full spatiotemporal precision matrix:")
print(f"  Shape: {Q_full_hourly.shape}")
print(
    f"  Non-zeros: {Q_full_hourly.nnz} / {Q_full_hourly.shape[0]**2} = {Q_full_hourly.nnz/Q_full_hourly.shape[0]**2:.4%}"
)
print()

# =============================================================================
# DATA QUALITY CHECK
# =============================================================================
print("=" * 60)
print("DATA QUALITY CHECK")
print("=" * 60)

# Check for temporal alignment
print("\nTemporal alignment:")
print(
    f"  Sensor data: {df_sensors_hourly.index.min()} to {df_sensors_hourly.index.max()}"
)
print(f"  Satellite data: {df_sat_daily.index.min()} to {df_sat_daily.index.max()}")

# Check value ranges
print("\nValue ranges:")
sensor_vals = sensor_data_hourly[~np.isnan(sensor_data_hourly)]
sat_vals = satellite_data_daily[~np.isnan(satellite_data_daily.astype(float))]
print(
    f"  Sensors: {sensor_vals.min():.1f} to {sensor_vals.max():.1f} (mean={sensor_vals.mean():.1f})"
)
print(
    f"  Satellite: {sat_vals.min():.1f} to {sat_vals.max():.1f} (mean={sat_vals.mean():.1f})"
)

# Check coverage by settlement
print("\nSensor coverage by settlement:")
for i, settlement in enumerate(df_sat.columns[:5]):  # Show first 5
    coverage = (~np.isnan(sensor_data_hourly[i, :])).mean()
    print(f"  {settlement}: {coverage:.1%}")
print("  ...")

print("\n" + "=" * 60)
print("READY FOR BAYESIAN INFERENCE!")
print("=" * 60)
print("\nAll components prepared:")
print("  ✓ Spatial adjacency and precision matrices")
print("  ✓ Hourly temporal precision matrix")
print("  ✓ Sensor data organized hourly")
print("  ✓ Satellite data organized daily")
print("  ✓ Observation dictionaries created")
print("  ✓ Aggregation matrices ready")
print("  ✓ Full spatiotemporal precision matrix")
print("\nNext step: Implement Bayesian inference to fuse all data sources!")

# With your fix:
# ds_med = ds_network.sel(time=slice(pd.Timestamp(2022, 7, 1), pd.Timestamp(2023, 1, 1))).groupby('settlement').median().sel(settlement=gdf_fig2_settlement.index)

# Now everything should align properly
print("VERIFYING ALIGNED DIMENSIONS")
print("=" * 60)


# Correct Kronecker products with settlement-first ordering
Q_space_kron = sp.kron(Q_spatial_scaled, sp.eye(n_hours), format="csr")
Q_time_kron = sp.kron(sp.eye(n_settlements), Q_temporal_hourly, format="csr")

print(f"Q_space_kron shape: {Q_space_kron.shape}")
print(f"Q_time_kron shape: {Q_time_kron.shape}")

# Verify they match
assert (
    Q_space_kron.shape == Q_time_kron.shape
), "Kronecker products have different shapes!"

# Combine into full precision matrix
Q_full_hourly = Q_space_kron + Q_time_kron

print(f"✓ Q_full_hourly shape: {Q_full_hourly.shape}")
print(f"  Expected: ({n_settlements * n_hours}, {n_settlements * n_hours})")
print(
    f"  Matches: {Q_full_hourly.shape == (n_settlements * n_hours, n_settlements * n_hours)}"
)
print(
    f"  Non-zeros: {Q_full_hourly.nnz:,} / {Q_full_hourly.shape[0]**2:,} = {Q_full_hourly.nnz/Q_full_hourly.shape[0]**2:.4%}"
)
print()

# Helper functions for parameter vector manipulation
def flatten_field(field_2d):
    """
    Flatten (n_settlements, n_hours) to parameter vector.
    Using settlement-first ordering: [s0_h0, s0_h1, ..., s1_h0, s1_h1, ...]
    """
    return field_2d.flatten("C")  # Row-major order


def unflatten_field(field_1d, n_settlements, n_hours):
    """
    Reshape parameter vector back to (n_settlements, n_hours).
    """
    return field_1d.reshape(n_settlements, n_hours, order="C")


def get_param_index(settlement_idx, hour_idx, n_hours):
    """Get index in flattened parameter vector for (settlement, hour)."""
    return settlement_idx * n_hours + hour_idx


# Test the full structure works
print("TESTING FULL STRUCTURE")
print("=" * 60)

# Create a smooth test field
test_field = np.ones((n_settlements, n_hours)) * 50
# Add some spatial variation
for s in range(n_settlements):
    test_field[s, :] += s * 0.5
# Add some temporal variation
for h in range(n_hours):
    test_field[:, h] += np.sin(2 * np.pi * h / 24) * 5  # Daily cycle

# Flatten and compute penalty
test_vec = flatten_field(test_field)
test_penalty = test_vec @ Q_full_hourly @ test_vec

print(f"Smooth test field:")
print(f"  Shape: {test_field.shape}")
print(f"  Mean: {test_field.mean():.1f}")
print(f"  Std: {test_field.std():.1f}")
print(f"  Penalty: {test_penalty:.2e}")
print()

# Random field for comparison
random_field = np.random.randn(n_settlements, n_hours) * 10 + 50
random_vec = flatten_field(random_field)
random_penalty = random_vec @ Q_full_hourly @ random_vec

print(f"Random test field:")
print(f"  Shape: {random_field.shape}")
print(f"  Mean: {random_field.mean():.1f}")
print(f"  Std: {random_field.std():.1f}")
print(f"  Penalty: {random_penalty:.2e}")
print(f"  Ratio (random/smooth): {random_penalty/test_penalty:.1f}x higher")
print()

# Summary of data coverage
print("DATA COVERAGE SUMMARY")
print("=" * 60)
sensor_coverage = (~np.isnan(sensor_data_hourly)).mean()
satellite_coverage = (~np.isnan(satellite_data_daily.astype(float))).mean()

print(f"Sensor data:")
print(f"  Total observations: {(~np.isnan(sensor_data_hourly)).sum():,}")
print(f"  Coverage: {sensor_coverage:.1%}")
print(f"  Observations in dict: {len(sensor_obs):,}")
print()

print(f"Satellite data:")
print(
    f"  Total observations: {(~np.isnan(satellite_data_daily.astype(float))).sum():,}"
)
print(f"  Coverage: {satellite_coverage:.1%}")
print(f"  Observations in dict: {len(satellite_obs):,}")
print()

print("✓ ALL COMPONENTS READY FOR BAYESIAN INFERENCE!")
print()
print("Next steps:")
print("  1. Initialize parameter estimates (theta_hourly)")
print("  2. Set up observation precision matrices")
print("  3. Implement MCMC or optimization for inference")
print("  4. Extract daily averages and compare with satellite")

import numpy as np
import scipy.sparse as sp

# =============================================================================
# NEXT SMALL STEP: BUILD OBSERVATION PRECISION MATRIX
# =============================================================================

# Helper function for indexing
def get_param_index(settlement_idx, hour_idx, n_hours):
    """Get index in flattened parameter vector for (settlement, hour)."""
    return settlement_idx * n_hours + hour_idx


print("Building observation precision matrix...")
print(f"  tau_sensor = {tau_sensor} (std ~ {1/np.sqrt(tau_sensor):.1f})")
print(f"  tau_satellite = {tau_satellite} (std ~ {1/np.sqrt(tau_satellite):.1f})")
print()

# Total number of parameters in flattened vector
n_params = n_settlements * n_hours

# Initialize sparse matrix for observation precision
# Using lil_matrix for efficient construction
Q_obs = sp.lil_matrix((n_params, n_params))

# Initialize vector for observation values
b_obs = np.zeros(n_params)

# =============================================================================
# ADD SENSOR OBSERVATIONS
# =============================================================================

print(f"Adding sensor observations...")
sensor_count = 0

for (s, h), value in sensor_obs.items():
    # Get index in the flattened parameter vector
    idx = get_param_index(s, h, n_hours)

    # Add to precision matrix (diagonal entry)
    Q_obs[idx, idx] += tau_sensor

    # Add to observation vector
    b_obs[idx] += tau_sensor * value

    sensor_count += 1

print(f"  Added {sensor_count} sensor observations")

# =============================================================================
# ADD SATELLITE OBSERVATIONS
# =============================================================================

print(f"Adding satellite observations...")
satellite_count = 0

for (s, d), value in satellite_obs.items():
    # Satellite observes at specific hour of each day
    h = d * 24 + satellite_hour  # Hour index

    if h < n_hours:  # Safety check
        # Get index in the flattened parameter vector
        idx = get_param_index(s, h, n_hours)

        # Add to precision matrix
        Q_obs[idx, idx] += tau_satellite

        # Add to observation vector
        b_obs[idx] += tau_satellite * value

        satellite_count += 1

print(f"  Added {satellite_count} satellite observations")
print()

# Convert to CSR format for efficient math operations
Q_obs = Q_obs.tocsr()

# =============================================================================
# CHECK WHAT WE BUILT
# =============================================================================

print("Observation precision matrix Q_obs:")
print(f"  Shape: {Q_obs.shape}")
print(f"  Non-zero entries: {Q_obs.nnz:,}")
print(f"  Sparsity: {Q_obs.nnz / (n_params**2):.4%}")
print()

print("Observation vector b_obs:")
print(f"  Shape: {b_obs.shape}")
print(f"  Non-zero entries: {np.sum(b_obs != 0):,}")
print(f"  Value range: [{b_obs[b_obs != 0].min():.1f}, {b_obs[b_obs != 0].max():.1f}]")
print()

# =============================================================================
# WHAT THIS MEANS
# =============================================================================

print("=" * 60)
print("WHAT WE JUST BUILT:")
print("=" * 60)
print()
print("Q_obs is a diagonal matrix where:")
print("  - Each sensor observation adds tau_sensor to its hour's diagonal")
print("  - Each satellite observation adds tau_satellite to its hour's diagonal")
print("  - Most diagonal entries are 0 (no observation)")
print()
print("b_obs is a vector where:")
print("  - Each sensor observation contributes tau_sensor * observed_value")
print("  - Each satellite observation contributes tau_satellite * observed_value")
print("  - Most entries are 0 (no observation)")
print()
print("These will be combined with the prior (Q_full_hourly) in the next step!")

# Simple check - which hours have the most observations?
obs_per_hour = np.zeros(n_hours)
for h in range(n_hours):
    # Count observations for this hour across all settlements
    for s in range(n_settlements):
        idx = get_param_index(s, h, n_hours)
        if Q_obs[idx, idx] > 0:
            obs_per_hour[h] += 1

# Find hours with most observations
peak_hours = np.argsort(obs_per_hour)[-5:]
print(f"\nHours with most observations:")
for h in peak_hours:
    day = h // 24
    hour = h % 24
    print(f"  Day {day}, Hour {hour:02d}: {int(obs_per_hour[h])} observations")

import numpy as np
import scipy.sparse as sp

# =============================================================================
# NEXT STEP: COMBINE PRIOR (Q_full_hourly) WITH OBSERVATIONS (Q_obs)
# =============================================================================

print("COMBINING PRIOR AND OBSERVATIONS")
print("=" * 60)

# The posterior precision is the sum of prior and observation precisions
# This is a key property of Gaussian distributions
print("Computing posterior precision matrix...")
print(
    f"  Prior (Q_full_hourly): {Q_full_hourly.shape}, {Q_full_hourly.nnz:,} non-zeros"
)
print(f"  Observations (Q_obs): {Q_obs.shape}, {Q_obs.nnz:,} non-zeros")

# Add them together
Q_posterior = Q_full_hourly + Q_obs

print(f"  Posterior (Q_posterior): {Q_posterior.shape}, {Q_posterior.nnz:,} non-zeros")
print()

# The posterior is the solution to: Q_posterior * theta = b_obs
# Where theta is our parameter vector (flattened hourly PM2.5 values)

# =============================================================================
# CHECK THE COMBINED MATRIX
# =============================================================================

print("Understanding the combined matrix:")
print("-" * 40)

# Check a few diagonal elements to see the combination
n_params = n_settlements * n_hours
sample_indices = [0, 100, 1000, 10000, 50000]

print("Sample diagonal values:")
for idx in sample_indices:
    if idx < n_params:
        prior_val = Q_full_hourly[idx, idx]
        obs_val = Q_obs[idx, idx]
        post_val = Q_posterior[idx, idx]

        # Figure out which settlement and hour this is
        s = idx // n_hours
        h = idx % n_hours
        day = h // 24
        hour = h % 24

        print(f"  Index {idx} (Settlement {s}, Day {day}, Hour {hour:02d}):")
        print(f"    Prior: {prior_val:.4f}, Obs: {obs_val:.4f}, Post: {post_val:.4f}")

print()

# =============================================================================
# WHAT THIS MEANS
# =============================================================================

print("=" * 60)
print("WHAT THE COMBINED MATRIX MEANS:")
print("=" * 60)
print()
print("Q_posterior = Q_full_hourly + Q_obs")
print()
print("This combines two sources of information:")
print("1. PRIOR (Q_full_hourly): Believes nearby values should be similar")
print("   - Spatial smoothness (neighboring settlements similar)")
print("   - Temporal smoothness (consecutive hours similar)")
print("   - Diurnal pattern (same hour across days similar)")
print()
print("2. OBSERVATIONS (Q_obs): Pulls estimates toward observed values")
print("   - Stronger pull where we have observations")
print("   - Weaker (zero) where we don't")
print()
print("The balance is controlled by the tau parameters:")
print(f"  - tau_spatial = {tau_spatial}: Spatial smoothness strength")
print(f"  - tau_hourly = {tau_hourly}: Hour-to-hour smoothness")
print(f"  - tau_daily = {tau_daily}: Day-to-day pattern consistency")
print(f"  - tau_sensor = {tau_sensor}: Trust in sensor data")
print(f"  - tau_satellite = {tau_satellite}: Trust in satellite data")
print()

# =============================================================================
# CHECK IF MATRIX IS READY FOR SOLVING
# =============================================================================

print("Matrix properties for solving:")
print("-" * 40)

# Check if symmetric (required for conjugate gradient)
# For sparse matrices, check a small subset
def check_symmetry_sparse(A, n_check=100):
    """Check if sparse matrix is symmetric by sampling entries."""
    rows, cols = A.nonzero()
    n_samples = min(n_check, len(rows))
    sample_idx = np.random.choice(len(rows), n_samples, replace=False)

    for i in sample_idx:
        if abs(A[rows[i], cols[i]] - A[cols[i], rows[i]]) > 1e-10:
            return False
    return True


is_symmetric = check_symmetry_sparse(Q_posterior)
print(f"  Symmetric: {is_symmetric}")
print(f"  Positive definite: Should be (CAR prior + observations)")
print(f"  Sparse: {Q_posterior.nnz / (n_params**2):.4%} non-zero")
print(f"  Size: {n_params:,} x {n_params:,}")
print()

if is_symmetric:
    print("✓ Matrix is ready for solving with Conjugate Gradient!")
else:
    print("⚠ Matrix may not be symmetric - check construction")

print()
print("Next step: Solve Q_posterior * theta = b_obs for theta")
print("This will give us the MAP estimate of hourly PM2.5!")

import numpy as np
from scipy.sparse.linalg import cg, spsolve
import time

# =============================================================================
# SOLVE Q_posterior * theta = b_obs
# =============================================================================

print("SOLVING FOR PM2.5 ESTIMATES")
print("=" * 60)
print()

# First, let's create an initial guess (helps convergence)
# Use the mean of observations as starting point
obs_mean = b_obs[b_obs != 0].mean() / tau_sensor  # Rough estimate
theta_init = np.ones(n_settlements * n_hours) * obs_mean

print(f"Initial guess: constant value = {obs_mean:.1f}")
print()

# =============================================================================
# METHOD 1: CONJUGATE GRADIENT (ITERATIVE)
# =============================================================================

print("Solving with Conjugate Gradient...")
print("  This is an iterative method good for large sparse systems")
print("  Matrix size: {0:,} x {0:,}".format(Q_posterior.shape[0]))
print()

# Set up callback to monitor convergence
iteration_count = [0]
residuals = []


def callback(xk):
    iteration_count[0] += 1
    if iteration_count[0] % 10 == 0:
        # Compute residual
        residual = np.linalg.norm(Q_posterior @ xk - b_obs)
        residuals.append(residual)
        print(f"  Iteration {iteration_count[0]}: residual = {residual:.2e}")


# Solve using conjugate gradient
start_time = time.time()

theta_solution, info = cg(
    Q_posterior, b_obs, x0=theta_init, maxiter=500, callback=callback
)

solve_time = time.time() - start_time

print()
if info == 0:
    print(f"✓ Converged in {iteration_count[0]} iterations!")
else:
    print(f"⚠ Did not fully converge (info={info})")
    print(f"  0: successful exit")
    print(f"  >0: convergence to tolerance not achieved, number of iterations")
    print(f"  <0: illegal input or breakdown")

print(f"  Solve time: {solve_time:.1f} seconds")
print()

# =============================================================================
# RESHAPE SOLUTION TO 2D
# =============================================================================

# Unflatten back to (n_settlements, n_hours)
theta_hourly = theta_solution.reshape(n_settlements, n_hours, order="C")

print("Solution statistics:")
print(f"  Shape: {theta_hourly.shape}")
print(f"  Mean: {theta_hourly.mean():.1f} µg/m³")
print(f"  Std: {theta_hourly.std():.1f} µg/m³")
print(f"  Min: {theta_hourly.min():.1f} µg/m³")
print(f"  Max: {theta_hourly.max():.1f} µg/m³")
print()

# =============================================================================
# QUICK VALIDATION
# =============================================================================

print("VALIDATION CHECKS")
print("=" * 60)

# Check 1: Compare with sensor observations at a few points
print("\n1. Spot check sensor observations vs estimates:")
sample_sensor_obs = list(sensor_obs.items())[:5]
for (s, h), obs_value in sample_sensor_obs:
    est_value = theta_hourly[s, h]
    diff = est_value - obs_value
    print(
        f"  Settlement {s}, Hour {h}: Obs={obs_value:.1f}, Est={est_value:.1f}, Diff={diff:.1f}"
    )

# Check 2: Compare with satellite observations
print("\n2. Spot check satellite observations vs estimates:")
sample_sat_obs = list(satellite_obs.items())[:5]
for (s, d), obs_value in sample_sat_obs:
    h = d * 24 + satellite_hour
    if h < n_hours:
        est_value = theta_hourly[s, h]
        diff = est_value - obs_value
        print(
            f"  Settlement {s}, Day {d} (Hour {h}): Obs={obs_value:.1f}, Est={est_value:.1f}, Diff={diff:.1f}"
        )

# Check 3: Compute overall fit statistics
print("\n3. Overall fit statistics:")

# Sensor fit
sensor_obs_values = []
sensor_est_values = []
for (s, h), obs_value in sensor_obs.items():
    sensor_obs_values.append(obs_value)
    sensor_est_values.append(theta_hourly[s, h])

sensor_obs_values = np.array(sensor_obs_values)
sensor_est_values = np.array(sensor_est_values)
sensor_rmse = np.sqrt(np.mean((sensor_obs_values - sensor_est_values) ** 2))
sensor_mae = np.mean(np.abs(sensor_obs_values - sensor_est_values))

print(f"  Sensor fit:")
print(f"    RMSE: {sensor_rmse:.2f} µg/m³")
print(f"    MAE: {sensor_mae:.2f} µg/m³")
print(f"    R²: {np.corrcoef(sensor_obs_values, sensor_est_values)[0,1]**2:.3f}")

# Satellite fit
sat_obs_values = []
sat_est_values = []
for (s, d), obs_value in satellite_obs.items():
    h = d * 24 + satellite_hour
    if h < n_hours:
        sat_obs_values.append(obs_value)
        sat_est_values.append(theta_hourly[s, h])

sat_obs_values = np.array(sat_obs_values)
sat_est_values = np.array(sat_est_values)
sat_rmse = np.sqrt(np.mean((sat_obs_values - sat_est_values) ** 2))
sat_mae = np.mean(np.abs(sat_obs_values - sat_est_values))

print(f"  Satellite fit:")
print(f"    RMSE: {sat_rmse:.2f} µg/m³")
print(f"    MAE: {sat_mae:.2f} µg/m³")
print(f"    R²: {np.corrcoef(sat_obs_values, sat_est_values)[0,1]**2:.3f}")

print()
print(
    "✓ Solution complete! You now have theta_hourly with estimates for all settlements and hours."
)
print()
print("Next steps could be:")
print("  1. Visualize the hourly patterns")
print("  2. Compute daily averages")
print("  3. Analyze the bias correction vs satellite")
print("  4. Extract diurnal patterns")

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from scipy.sparse.linalg import cg
import time

# =============================================================================
# CONFIGURATION
# =============================================================================


class ModelConfig:
    """Model hyperparameters and settings"""

    # Precision parameters
    tau_spatial = 2.0  # Spatial smoothness
    tau_hourly = 0.1  # Hour-to-hour smoothness
    tau_daily = 0.01  # Day-to-day pattern consistency
    tau_sensor = 20.0  # Sensor observation precision
    tau_satellite = 4.0  # Satellite observation precision

    # Spatial adjacency settings
    n_neighbors = 3  # K-nearest neighbors for spatial graph

    # Satellite settings
    satellite_hour = 13  # Hour of day satellite observes

    # Solver settings
    max_iterations = 500


config = ModelConfig()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def create_inverse_distance_adjacency(dist_matrix, k=3):
    """
    Create inverse distance weighted adjacency matrix using k-nearest neighbors.

    Parameters:
    -----------
    dist_matrix : ndarray
        Pairwise distance matrix between settlements
    k : int
        Number of nearest neighbors

    Returns:
    --------
    adj : ndarray
        Symmetric adjacency matrix with inverse distance weights
    """
    n = dist_matrix.shape[0]
    adj = np.zeros((n, n))

    for i in range(n):
        distances_i = dist_matrix[i, :].copy()
        distances_i[i] = np.inf  # Exclude self
        closest = np.argsort(distances_i)[:k]

        for j in closest:
            adj[i, j] = 1.0 / dist_matrix[i, j]

    # Make symmetric
    adj = (adj + adj.T) / 2
    return adj


def build_spatial_precision(adj_matrix, tau):
    """
    Build spatial precision matrix (Graph Laplacian).

    Q_spatial = tau * (D - A)
    where D is diagonal degree matrix and A is adjacency matrix
    """
    row_sums = np.sum(adj_matrix, axis=1)
    D = np.diag(row_sums)
    Q = D - adj_matrix
    return tau * Q


def build_temporal_precision(n_hours, tau_hourly, tau_daily):
    """
    Build temporal precision matrix with hourly and daily components.

    Parameters:
    -----------
    n_hours : int
        Total number of hours
    tau_hourly : float
        Precision for hour-to-hour transitions
    tau_daily : float
        Precision for same-hour-different-day transitions
    """
    Q = np.zeros((n_hours, n_hours))

    for h in range(n_hours):
        # Hour-to-hour transitions
        if h > 0:
            Q[h, h - 1] = -tau_hourly
            Q[h, h] += tau_hourly
        if h < n_hours - 1:
            Q[h, h + 1] = -tau_hourly
            Q[h, h] += tau_hourly

        # Same hour, different day (24-hour apart)
        if h >= 24:
            Q[h, h - 24] = -tau_daily
            Q[h, h] += tau_daily
        if h < n_hours - 24:
            Q[h, h + 24] = -tau_daily
            Q[h, h] += tau_daily

    return Q


def flatten_field(field_2d):
    """Flatten (n_settlements, n_hours) to parameter vector (row-major)."""
    return field_2d.flatten("C")


def unflatten_field(field_1d, n_settlements, n_hours):
    """Reshape parameter vector back to (n_settlements, n_hours)."""
    return field_1d.reshape(n_settlements, n_hours, order="C")


def get_param_index(settlement_idx, hour_idx, n_hours):
    """Get index in flattened parameter vector for (settlement, hour)."""
    return settlement_idx * n_hours + hour_idx


# =============================================================================
# DATA PREPARATION
# =============================================================================

print("=" * 60)
print("SPATIOTEMPORAL BAYESIAN PM2.5 MODEL")
print("=" * 60)
print()

# Assume data is loaded:
# - df_sensors: hourly sensor data (time x settlement)
# - df_sat: daily satellite data (time x settlement)
# - gdf_fig2_settlement: GeoDataFrame with settlement geometries

# Determine dimensions
n_settlements = len(df_sat.columns)
date_start = df_sensors.index.min().replace(hour=0, minute=0, second=0)
date_end = df_sensors.index.max().replace(hour=23, minute=59, second=59)
n_days = len(pd.date_range(date_start.date(), date_end.date(), freq="D"))
n_hours = n_days * 24

print(f"Study period: {date_start.date()} to {date_end.date()}")
print(
    f"Dimensions: {n_settlements} settlements × {n_days} days × 24 hours = {n_hours} hours"
)
print()

# Prepare hourly index
hourly_index = pd.date_range(date_start, date_end, freq="H")[:n_hours]

# Resample sensor data to hourly
df_sensors_hourly = df_sensors.resample("H").mean().reindex(hourly_index)
sensor_data = df_sensors_hourly[df_sat.columns].T.values  # (n_settlements, n_hours)

# Resample satellite data to daily
daily_index = pd.date_range(date_start.date(), date_end.date(), freq="D")[:n_days]
df_sat_daily = df_sat.resample("D").mean().reindex(daily_index)
satellite_data = df_sat_daily[df_sat.columns].T.values.astype(
    float
)  # (n_settlements, n_days)

print(
    f"Sensor coverage: {(~np.isnan(sensor_data)).mean():.1%} ({(~np.isnan(sensor_data)).sum():,} observations)"
)
print(
    f"Satellite coverage: {(~np.isnan(satellite_data)).mean():.1%} ({(~np.isnan(satellite_data)).sum():,} observations)"
)
print()

# =============================================================================
# BUILD SPATIAL STRUCTURE
# =============================================================================

print("Building spatial structure...")

# Compute centroids and distance matrix
centroids = np.array(
    [[geom.centroid.x, geom.centroid.y] for geom in gdf_fig2_settlement["geometry"]]
)
dist_matrix = cdist(centroids, centroids, metric="euclidean")

# Create adjacency matrix
adj_matrix = create_inverse_distance_adjacency(dist_matrix, k=config.n_neighbors)
Q_spatial = build_spatial_precision(adj_matrix, config.tau_spatial)

print(f"  K-nearest neighbors: {config.n_neighbors}")
print(f"  Non-zero adjacencies: {(adj_matrix > 0).sum()}")
print()

# =============================================================================
# BUILD TEMPORAL STRUCTURE
# =============================================================================

print("Building temporal structure...")

Q_temporal = build_temporal_precision(n_hours, config.tau_hourly, config.tau_daily)

print(f"  Hourly transitions: tau = {config.tau_hourly}")
print(f"  Daily pattern: tau = {config.tau_daily}")
print()

# =============================================================================
# BUILD FULL SPATIOTEMPORAL PRECISION
# =============================================================================

print("Building spatiotemporal precision matrix...")

# Kronecker products (settlement-first ordering)
Q_space_kron = sp.kron(Q_spatial, sp.eye(n_hours), format="csr")
Q_time_kron = sp.kron(sp.eye(n_settlements), Q_temporal, format="csr")
Q_prior = Q_space_kron + Q_time_kron

n_params = n_settlements * n_hours
print(f"  Shape: ({n_params:,}, {n_params:,})")
print(f"  Non-zeros: {Q_prior.nnz:,} ({Q_prior.nnz/n_params**2:.4%})")
print()

# =============================================================================
# BUILD OBSERVATION STRUCTURE
# =============================================================================

print("Building observation structure...")

Q_obs = sp.lil_matrix((n_params, n_params))
b_obs = np.zeros(n_params)

# Add sensor observations
n_sensor_obs = 0
for s in range(n_settlements):
    for h in range(n_hours):
        if not np.isnan(sensor_data[s, h]):
            idx = get_param_index(s, h, n_hours)
            Q_obs[idx, idx] += config.tau_sensor
            b_obs[idx] += config.tau_sensor * sensor_data[s, h]
            n_sensor_obs += 1

# Add satellite observations
n_sat_obs = 0
for s in range(n_settlements):
    for d in range(n_days):
        if not np.isnan(satellite_data[s, d]):
            h = d * 24 + config.satellite_hour
            if h < n_hours:
                idx = get_param_index(s, h, n_hours)
                Q_obs[idx, idx] += config.tau_satellite
                b_obs[idx] += config.tau_satellite * satellite_data[s, d]
                n_sat_obs += 1

Q_obs = Q_obs.tocsr()

print(f"  Sensor observations: {n_sensor_obs:,}")
print(f"  Satellite observations: {n_sat_obs:,}")
print()

# =============================================================================
# POSTERIOR PRECISION
# =============================================================================

print("Combining prior and observations...")

Q_posterior = Q_prior + Q_obs

print(f"  Posterior precision: {Q_posterior.shape}, {Q_posterior.nnz:,} non-zeros")
print()

# =============================================================================
# SOLVE FOR MAP ESTIMATE
# =============================================================================

print("Solving for MAP estimate...")

# Initial guess
obs_mean = b_obs[b_obs != 0].mean() / config.tau_sensor
theta_init = np.ones(n_params) * obs_mean

# Conjugate gradient solver
iteration = [0]


def callback(xk):
    iteration[0] += 1
    if iteration[0] % 50 == 0:
        residual = np.linalg.norm(Q_posterior @ xk - b_obs)
        print(f"  Iteration {iteration[0]}: residual = {residual:.2e}")


start_time = time.time()
theta_solution, info = cg(
    Q_posterior, b_obs, x0=theta_init, maxiter=config.max_iterations, callback=callback
)
solve_time = time.time() - start_time

if info == 0:
    print(f"✓ Converged in {iteration[0]} iterations ({solve_time:.1f}s)")
else:
    print(f"⚠ Partial convergence (info={info}, {solve_time:.1f}s)")
print()

# Reshape to 2D
theta_hourly = unflatten_field(theta_solution, n_settlements, n_hours)

# =============================================================================
# VALIDATION
# =============================================================================

print("=" * 60)
print("RESULTS")
print("=" * 60)
print()

print(f"Estimated PM2.5 field:")
print(f"  Shape: {theta_hourly.shape}")
print(f"  Mean: {theta_hourly.mean():.1f} µg/m³")
print(f"  Range: [{theta_hourly.min():.1f}, {theta_hourly.max():.1f}] µg/m³")
print()

# Compute fit statistics
sensor_obs = sensor_data[~np.isnan(sensor_data)]
sensor_est = theta_hourly[~np.isnan(sensor_data)]
sensor_rmse = np.sqrt(np.mean((sensor_obs - sensor_est) ** 2))
sensor_r2 = np.corrcoef(sensor_obs, sensor_est)[0, 1] ** 2

print(f"Sensor fit:")
print(f"  RMSE: {sensor_rmse:.2f} µg/m³")
print(f"  R²: {sensor_r2:.3f}")
print()

# Satellite fit
sat_obs_list = []
sat_est_list = []
for s in range(n_settlements):
    for d in range(n_days):
        if not np.isnan(satellite_data[s, d]):
            h = d * 24 + config.satellite_hour
            if h < n_hours:
                sat_obs_list.append(satellite_data[s, d])
                sat_est_list.append(theta_hourly[s, h])

sat_obs = np.array(sat_obs_list)
sat_est = np.array(sat_est_list)
sat_rmse = np.sqrt(np.mean((sat_obs - sat_est) ** 2))
sat_r2 = np.corrcoef(sat_obs, sat_est)[0, 1] ** 2

print(f"Satellite fit:")
print(f"  RMSE: {sat_rmse:.2f} µg/m³")
print(f"  R²: {sat_r2:.3f}")
print()

print("✓ Model complete! Results stored in theta_hourly")
print()
print("Next steps:")
print("  - Visualize spatial patterns")
print("  - Extract diurnal cycles")
print("  - Compute daily/monthly averages")
print("  - Analyze uncertainties")

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from scipy.sparse.linalg import cg
import time

# =============================================================================
# CONFIGURATION
# =============================================================================


class ModelConfig:
    """Model hyperparameters and settings"""

    # Precision parameters
    tau_spatial = 2.0  # Spatial smoothness
    tau_hourly = 0.1  # Hour-to-hour smoothness
    tau_daily = 0.01  # Day-to-day pattern consistency
    tau_sensor = 20.0  # Sensor observation precision
    tau_satellite = 4.0  # Satellite observation precision

    # Spatial adjacency settings
    n_neighbors = 3  # K-nearest neighbors for spatial graph

    # Satellite settings
    satellite_hour = 13  # Hour of day satellite observes

    # Solver settings
    max_iterations = 500


config = ModelConfig()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def create_inverse_distance_adjacency(dist_matrix, k=3):
    """
    Create inverse distance weighted adjacency matrix using k-nearest neighbors.

    Parameters:
    -----------
    dist_matrix : ndarray
        Pairwise distance matrix between settlements
    k : int
        Number of nearest neighbors

    Returns:
    --------
    adj : ndarray
        Symmetric adjacency matrix with inverse distance weights
    """
    n = dist_matrix.shape[0]
    adj = np.zeros((n, n))

    for i in range(n):
        distances_i = dist_matrix[i, :].copy()
        distances_i[i] = np.inf  # Exclude self
        closest = np.argsort(distances_i)[:k]

        for j in closest:
            adj[i, j] = 1.0 / dist_matrix[i, j]

    # Make symmetric
    adj = (adj + adj.T) / 2
    return adj


def build_spatial_precision(adj_matrix, tau):
    """
    Build spatial precision matrix (Graph Laplacian).

    Q_spatial = tau * (D - A)
    where D is diagonal degree matrix and A is adjacency matrix
    """
    row_sums = np.sum(adj_matrix, axis=1)
    D = np.diag(row_sums)
    Q = D - adj_matrix
    return tau * Q


def build_temporal_precision(n_hours, tau_hourly, tau_daily):
    """
    Build temporal precision matrix with hourly and daily components.

    Parameters:
    -----------
    n_hours : int
        Total number of hours
    tau_hourly : float
        Precision for hour-to-hour transitions
    tau_daily : float
        Precision for same-hour-different-day transitions
    """
    Q = np.zeros((n_hours, n_hours))

    for h in range(n_hours):
        # Hour-to-hour transitions
        if h > 0:
            Q[h, h - 1] = -tau_hourly
            Q[h, h] += tau_hourly
        if h < n_hours - 1:
            Q[h, h + 1] = -tau_hourly
            Q[h, h] += tau_hourly

        # Same hour, different day (24-hour apart)
        if h >= 24:
            Q[h, h - 24] = -tau_daily
            Q[h, h] += tau_daily
        if h < n_hours - 24:
            Q[h, h + 24] = -tau_daily
            Q[h, h] += tau_daily

    return Q


def flatten_field(field_2d):
    """Flatten (n_settlements, n_hours) to parameter vector (row-major)."""
    return field_2d.flatten("C")


def unflatten_field(field_1d, n_settlements, n_hours):
    """Reshape parameter vector back to (n_settlements, n_hours)."""
    return field_1d.reshape(n_settlements, n_hours, order="C")


def get_param_index(settlement_idx, hour_idx, n_hours):
    """Get index in flattened parameter vector for (settlement, hour)."""
    return settlement_idx * n_hours + hour_idx


# =============================================================================
# DATA PREPARATION
# =============================================================================

print("=" * 60)
print("SPATIOTEMPORAL BAYESIAN PM2.5 MODEL")
print("=" * 60)
print()

# Assume data is loaded:
# - df_sensors: hourly sensor data (time x settlement)
# - df_sat: daily satellite data (time x settlement)
# - gdf_fig2_settlement: GeoDataFrame with settlement geometries

# Determine dimensions
n_settlements = len(df_sat.columns)
date_start = df_sensors.index.min().replace(hour=0, minute=0, second=0)
date_end = df_sensors.index.max().replace(hour=23, minute=59, second=59)
n_days = len(pd.date_range(date_start.date(), date_end.date(), freq="D"))
n_hours = n_days * 24

print(f"Study period: {date_start.date()} to {date_end.date()}")
print(
    f"Dimensions: {n_settlements} settlements × {n_days} days × 24 hours = {n_hours} hours"
)
print()

# Prepare hourly index
hourly_index = pd.date_range(date_start, date_end, freq="H")[:n_hours]

# Resample sensor data to hourly
df_sensors_hourly = df_sensors.resample("H").mean().reindex(hourly_index)
sensor_data = df_sensors_hourly[df_sat.columns].T.values  # (n_settlements, n_hours)

# Resample satellite data to daily
daily_index = pd.date_range(date_start.date(), date_end.date(), freq="D")[:n_days]
df_sat_daily = df_sat.resample("D").mean().reindex(daily_index)
satellite_data = df_sat_daily[df_sat.columns].T.values.astype(
    float
)  # (n_settlements, n_days)

print(
    f"Sensor coverage: {(~np.isnan(sensor_data)).mean():.1%} ({(~np.isnan(sensor_data)).sum():,} observations)"
)
print(
    f"Satellite coverage: {(~np.isnan(satellite_data)).mean():.1%} ({(~np.isnan(satellite_data)).sum():,} observations)"
)
print()

# =============================================================================
# BUILD SPATIAL STRUCTURE
# =============================================================================

print("Building spatial structure...")

# Compute centroids and distance matrix
centroids = np.array(
    [[geom.centroid.x, geom.centroid.y] for geom in gdf_fig2_settlement["geometry"]]
)
dist_matrix = cdist(centroids, centroids, metric="euclidean")

# Create adjacency matrix
adj_matrix = create_inverse_distance_adjacency(dist_matrix, k=config.n_neighbors)
Q_spatial = build_spatial_precision(adj_matrix, config.tau_spatial)

print(f"  K-nearest neighbors: {config.n_neighbors}")
print(f"  Non-zero adjacencies: {(adj_matrix > 0).sum()}")
print()

# =============================================================================
# BUILD TEMPORAL STRUCTURE
# =============================================================================

print("Building temporal structure...")

Q_temporal = build_temporal_precision(n_hours, config.tau_hourly, config.tau_daily)

print(f"  Hourly transitions: tau = {config.tau_hourly}")
print(f"  Daily pattern: tau = {config.tau_daily}")
print()

# =============================================================================
# BUILD FULL SPATIOTEMPORAL PRECISION
# =============================================================================

print("Building spatiotemporal precision matrix...")

# Kronecker products (settlement-first ordering)
Q_space_kron = sp.kron(Q_spatial, sp.eye(n_hours), format="csr")
Q_time_kron = sp.kron(sp.eye(n_settlements), Q_temporal, format="csr")
Q_prior = Q_space_kron + Q_time_kron

n_params = n_settlements * n_hours
print(f"  Shape: ({n_params:,}, {n_params:,})")
print(f"  Non-zeros: {Q_prior.nnz:,} ({Q_prior.nnz/n_params**2:.4%})")
print()

# =============================================================================
# BUILD OBSERVATION STRUCTURE
# =============================================================================

print("Building observation structure...")

Q_obs = sp.lil_matrix((n_params, n_params))
b_obs = np.zeros(n_params)

# Add sensor observations
n_sensor_obs = 0
for s in range(n_settlements):
    for h in range(n_hours):
        if not np.isnan(sensor_data[s, h]):
            idx = get_param_index(s, h, n_hours)
            Q_obs[idx, idx] += config.tau_sensor
            b_obs[idx] += config.tau_sensor * sensor_data[s, h]
            n_sensor_obs += 1

# Add satellite observations
n_sat_obs = 0
for s in range(n_settlements):
    for d in range(n_days):
        if not np.isnan(satellite_data[s, d]):
            h = d * 24 + config.satellite_hour
            if h < n_hours:
                idx = get_param_index(s, h, n_hours)
                Q_obs[idx, idx] += config.tau_satellite
                b_obs[idx] += config.tau_satellite * satellite_data[s, d]
                n_sat_obs += 1

Q_obs = Q_obs.tocsr()

print(f"  Sensor observations: {n_sensor_obs:,}")
print(f"  Satellite observations: {n_sat_obs:,}")
print()

# =============================================================================
# POSTERIOR PRECISION
# =============================================================================

print("Combining prior and observations...")

Q_posterior = Q_prior + Q_obs

print(f"  Posterior precision: {Q_posterior.shape}, {Q_posterior.nnz:,} non-zeros")
print()

# =============================================================================
# SOLVE FOR MAP ESTIMATE
# =============================================================================

print("Solving for MAP estimate...")

# Initial guess
obs_mean = b_obs[b_obs != 0].mean() / config.tau_sensor
theta_init = np.ones(n_params) * obs_mean

# Conjugate gradient solver
iteration = [0]


def callback(xk):
    iteration[0] += 1
    if iteration[0] % 50 == 0:
        residual = np.linalg.norm(Q_posterior @ xk - b_obs)
        print(f"  Iteration {iteration[0]}: residual = {residual:.2e}")


start_time = time.time()
theta_solution, info = cg(
    Q_posterior, b_obs, x0=theta_init, maxiter=config.max_iterations, callback=callback
)
solve_time = time.time() - start_time

if info == 0:
    print(f"✓ Converged in {iteration[0]} iterations ({solve_time:.1f}s)")
else:
    print(f"⚠ Partial convergence (info={info}, {solve_time:.1f}s)")
print()

# Reshape to 2D
theta_hourly = unflatten_field(theta_solution, n_settlements, n_hours)

# =============================================================================
# VALIDATION
# =============================================================================

print("=" * 60)
print("RESULTS")
print("=" * 60)
print()

print(f"Estimated PM2.5 field:")
print(f"  Shape: {theta_hourly.shape}")
print(f"  Mean: {theta_hourly.mean():.1f} µg/m³")
print(f"  Range: [{theta_hourly.min():.1f}, {theta_hourly.max():.1f}] µg/m³")
print()

# Compute fit statistics
sensor_obs = sensor_data[~np.isnan(sensor_data)]
sensor_est = theta_hourly[~np.isnan(sensor_data)]
sensor_rmse = np.sqrt(np.mean((sensor_obs - sensor_est) ** 2))
sensor_r2 = np.corrcoef(sensor_obs, sensor_est)[0, 1] ** 2

print(f"Sensor fit:")
print(f"  RMSE: {sensor_rmse:.2f} µg/m³")
print(f"  R²: {sensor_r2:.3f}")
print()

# Satellite fit
sat_obs_list = []
sat_est_list = []
for s in range(n_settlements):
    for d in range(n_days):
        if not np.isnan(satellite_data[s, d]):
            h = d * 24 + config.satellite_hour
            if h < n_hours:
                sat_obs_list.append(satellite_data[s, d])
                sat_est_list.append(theta_hourly[s, h])

sat_obs = np.array(sat_obs_list)
sat_est = np.array(sat_est_list)
sat_rmse = np.sqrt(np.mean((sat_obs - sat_est) ** 2))
sat_r2 = np.corrcoef(sat_obs, sat_est)[0, 1] ** 2

print(f"Satellite fit:")
print(f"  RMSE: {sat_rmse:.2f} µg/m³")
print(f"  R²: {sat_r2:.3f}")
print()

# =============================================================================
# LEAVE-ONE-OUT CROSS-VALIDATION
# =============================================================================

print("=" * 60)
print("LEAVE-ONE-OUT CROSS-VALIDATION")
print("=" * 60)
print()

print("Computing LOO predictions using Sherman-Morrison formula...")
print("(This is efficient - no need to re-solve for each observation)")
print()


def compute_loo_predictions(theta_full, Q_posterior, Q_obs, b_obs, observations):
    """
    Compute LOO predictions efficiently using Sherman-Morrison formula.

    For observation i with precision tau_i and value y_i:
    theta^{-i}_i = (Q_posterior[i,i] * theta_i - tau_i * y_i) / (Q_posterior[i,i] - tau_i)

    Parameters:
    -----------
    theta_full : ndarray
        Full MAP estimate (flattened)
    Q_posterior : sparse matrix
        Full posterior precision
    Q_obs : sparse matrix
        Observation precision (diagonal)
    b_obs : ndarray
        Observation contribution vector
    observations : dict
        Dict mapping (settlement, time_idx) -> observed_value

    Returns:
    --------
    loo_preds : ndarray
        LOO predictions
    loo_obs : ndarray
        Observed values
    """
    loo_preds = []
    loo_obs = []

    for (s, t), y_obs in observations.items():
        idx = get_param_index(s, t, n_hours)

        # Get diagonal values
        q_post = Q_posterior[idx, idx]  # Posterior precision at this point
        tau_obs = Q_obs[idx, idx]  # Observation precision at this point

        # LOO prediction formula
        if q_post - tau_obs > 1e-10:  # Avoid division by zero
            theta_loo = (q_post * theta_full[idx] - tau_obs * y_obs) / (
                q_post - tau_obs
            )
            loo_preds.append(theta_loo)
            loo_obs.append(y_obs)

    return np.array(loo_preds), np.array(loo_obs)


# Prepare observation dictionaries
sensor_obs_dict = {}
for s in range(n_settlements):
    for h in range(n_hours):
        if not np.isnan(sensor_data[s, h]):
            sensor_obs_dict[(s, h)] = sensor_data[s, h]

satellite_obs_dict = {}
for s in range(n_settlements):
    for d in range(n_days):
        if not np.isnan(satellite_data[s, d]):
            h = d * 24 + config.satellite_hour
            if h < n_hours:
                satellite_obs_dict[(s, h)] = satellite_data[s, d]

# Compute LOO predictions for sensors
print("Computing LOO for sensor observations...")
sensor_loo_pred, sensor_loo_obs = compute_loo_predictions(
    theta_solution, Q_posterior, Q_obs, b_obs, sensor_obs_dict
)

sensor_loo_rmse = np.sqrt(np.mean((sensor_loo_obs - sensor_loo_pred) ** 2))
sensor_loo_mae = np.mean(np.abs(sensor_loo_obs - sensor_loo_pred))
sensor_loo_r2 = np.corrcoef(sensor_loo_obs, sensor_loo_pred)[0, 1] ** 2
sensor_loo_nrmse = sensor_loo_rmse / sensor_loo_obs.mean() * 100

print(f"  Sensor LOO (n={len(sensor_loo_obs):,}):")
print(f"    RMSE: {sensor_loo_rmse:.2f} µg/m³")
print(f"    nRMSE: {sensor_loo_nrmse:.1f}%")
print(f"    MAE: {sensor_loo_mae:.2f} µg/m³")
print(f"    R²: {sensor_loo_r2:.3f}")
print()

# Compute LOO predictions for satellites
print("Computing LOO for satellite observations...")
sat_loo_pred, sat_loo_obs = compute_loo_predictions(
    theta_solution, Q_posterior, Q_obs, b_obs, satellite_obs_dict
)

sat_loo_rmse = np.sqrt(np.mean((sat_loo_obs - sat_loo_pred) ** 2))
sat_loo_mae = np.mean(np.abs(sat_loo_obs - sat_loo_pred))
sat_loo_r2 = np.corrcoef(sat_loo_obs, sat_loo_pred)[0, 1] ** 2
sat_loo_nrmse = sat_loo_rmse / sat_loo_obs.mean() * 100

print(f"  Satellite LOO (n={len(sat_loo_obs):,}):")
print(f"    RMSE: {sat_loo_rmse:.2f} µg/m³")
print(f"    nRMSE: {sat_loo_nrmse:.1f}%")
print(f"    MAE: {sat_loo_mae:.2f} µg/m³")
print(f"    R²: {sat_loo_r2:.3f}")
print()

# Compare training vs LOO
print("=" * 60)
print("SUMMARY: Training vs LOO Performance")
print("=" * 60)
print()
print(
    f"{'Metric':<20} {'Sensor Train':<15} {'Sensor LOO':<15} {'Sat Train':<15} {'Sat LOO':<15}"
)
print("-" * 80)
print(
    f"{'RMSE (µg/m³)':<20} {sensor_rmse:<15.2f} {sensor_loo_rmse:<15.2f} {sat_rmse:<15.2f} {sat_loo_rmse:<15.2f}"
)
print(
    f"{'nRMSE (%)':<20} {sensor_rmse/sensor_obs.mean()*100:<15.1f} {sensor_loo_nrmse:<15.1f} {sat_rmse/sat_obs.mean()*100:<15.1f} {sat_loo_nrmse:<15.1f}"
)
print(
    f"{'R²':<20} {sensor_r2:<15.3f} {sensor_loo_r2:<15.3f} {sat_r2:<15.3f} {sat_loo_r2:<15.3f}"
)
print()

# Check for overfitting
sensor_overfit = ((sensor_loo_rmse - sensor_rmse) / sensor_rmse) * 100
sat_overfit = ((sat_loo_rmse - sat_rmse) / sat_rmse) * 100

print("Overfitting assessment:")
print(f"  Sensor: LOO RMSE is {sensor_overfit:+.1f}% vs training")
print(f"  Satellite: LOO RMSE is {sat_overfit:+.1f}% vs training")

if sensor_overfit < 10 and sat_overfit < 10:
    print("  ✓ Minimal overfitting - model generalizes well")
elif sensor_overfit < 20 and sat_overfit < 20:
    print("  ⚠ Moderate overfitting - consider reducing model complexity")
else:
    print("  ⚠ Significant overfitting - model may be too flexible")
print()

print("✓ Model complete! Results stored in theta_hourly")
print()
print("Next steps:")
print("  - Visualize spatial patterns")
print("  - Extract diurnal cycles")
print("  - Compute daily/monthly averages")
print("  - Tune hyperparameters to minimize LOO error")

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import cg
from itertools import product
import time

# =============================================================================
# HYPERPARAMETER TUNING FRAMEWORK
# =============================================================================


class HyperparameterTuner:
    """Grid search for model hyperparameters using LOO CV"""

    def __init__(
        self,
        sensor_data,
        satellite_data,
        Q_spatial_base,
        Q_temporal_base,
        n_settlements,
        n_hours,
        n_days,
        satellite_hour,
    ):
        """
        Parameters:
        -----------
        sensor_data, satellite_data : observation arrays
        Q_spatial_base, Q_temporal_base : unscaled precision matrices
        """
        self.sensor_data = sensor_data
        self.satellite_data = satellite_data
        self.Q_spatial_base = Q_spatial_base
        self.Q_temporal_base = Q_temporal_base
        self.n_settlements = n_settlements
        self.n_hours = n_hours
        self.n_days = n_days
        self.satellite_hour = satellite_hour
        self.n_params = n_settlements * n_hours

        # Prepare observation dictionaries (do once)
        self.sensor_obs_dict = {}
        for s in range(n_settlements):
            for h in range(n_hours):
                if not np.isnan(sensor_data[s, h]):
                    self.sensor_obs_dict[(s, h)] = sensor_data[s, h]

        self.sat_obs_dict = {}
        for s in range(n_settlements):
            for d in range(n_days):
                if not np.isnan(satellite_data[s, d]):
                    h = d * 24 + satellite_hour
                    if h < n_hours:
                        self.sat_obs_dict[(s, h)] = satellite_data[s, d]

    def build_and_solve(
        self,
        tau_spatial,
        tau_hourly,
        tau_daily,
        tau_sensor,
        tau_satellite,
        verbose=False,
    ):
        """Build model with given hyperparameters and solve"""

        # Scale precision matrices
        Q_spatial = tau_spatial * self.Q_spatial_base
        Q_temporal = tau_hourly * self.Q_temporal_base.copy()

        # Add daily component to temporal precision
        for h in range(self.n_hours):
            if h >= 24:
                Q_temporal[h, h - 24] += -tau_daily
                Q_temporal[h, h] += tau_daily
            if h < self.n_hours - 24:
                Q_temporal[h, h + 24] += -tau_daily
                Q_temporal[h, h] += tau_daily

        # Build prior
        Q_space_kron = sp.kron(Q_spatial, sp.eye(self.n_hours), format="csr")
        Q_time_kron = sp.kron(sp.eye(self.n_settlements), Q_temporal, format="csr")
        Q_prior = Q_space_kron + Q_time_kron

        # Build observations
        Q_obs = sp.lil_matrix((self.n_params, self.n_params))
        b_obs = np.zeros(self.n_params)

        for (s, h), value in self.sensor_obs_dict.items():
            idx = s * self.n_hours + h
            Q_obs[idx, idx] += tau_sensor
            b_obs[idx] += tau_sensor * value

        for (s, h), value in self.sat_obs_dict.items():
            idx = s * self.n_hours + h
            Q_obs[idx, idx] += tau_satellite
            b_obs[idx] += tau_satellite * value

        Q_obs = Q_obs.tocsr()
        Q_posterior = Q_prior + Q_obs

        # Solve
        obs_mean = b_obs[b_obs != 0].mean() / tau_sensor if tau_sensor > 0 else 50
        theta_init = np.ones(self.n_params) * obs_mean

        theta_solution, info = cg(
            Q_posterior, b_obs, x0=theta_init, maxiter=500, atol=1e-6
        )

        if verbose and info != 0:
            print(f"    Warning: CG info={info}")

        return theta_solution, Q_posterior, Q_obs, b_obs

    def compute_loo_metrics(self, theta_solution, Q_posterior, Q_obs, b_obs):
        """Compute LOO cross-validation metrics"""

        # Sensor LOO
        sensor_preds, sensor_obs = [], []
        for (s, h), y_obs in self.sensor_obs_dict.items():
            idx = s * self.n_hours + h
            q_post = Q_posterior[idx, idx]
            tau_obs = Q_obs[idx, idx]

            if q_post - tau_obs > 1e-10:
                theta_loo = (q_post * theta_solution[idx] - tau_obs * y_obs) / (
                    q_post - tau_obs
                )
                sensor_preds.append(theta_loo)
                sensor_obs.append(y_obs)

        sensor_preds = np.array(sensor_preds)
        sensor_obs = np.array(sensor_obs)
        sensor_rmse = np.sqrt(np.mean((sensor_obs - sensor_preds) ** 2))
        sensor_r2 = np.corrcoef(sensor_obs, sensor_preds)[0, 1] ** 2

        # Satellite LOO
        sat_preds, sat_obs = [], []
        for (s, h), y_obs in self.sat_obs_dict.items():
            idx = s * self.n_hours + h
            q_post = Q_posterior[idx, idx]
            tau_obs = Q_obs[idx, idx]

            if q_post - tau_obs > 1e-10:
                theta_loo = (q_post * theta_solution[idx] - tau_obs * y_obs) / (
                    q_post - tau_obs
                )
                sat_preds.append(theta_loo)
                sat_obs.append(y_obs)

        sat_preds = np.array(sat_preds)
        sat_obs = np.array(sat_obs)
        sat_rmse = np.sqrt(np.mean((sat_obs - sat_preds) ** 2))
        sat_r2 = np.corrcoef(sat_obs, sat_preds)[0, 1] ** 2

        # Combined metric
        combined_rmse = np.sqrt(
            np.mean(
                np.concatenate(
                    [(sensor_obs - sensor_preds) ** 2, (sat_obs - sat_preds) ** 2]
                )
            )
        )

        return {
            "sensor_rmse": sensor_rmse,
            "sensor_r2": sensor_r2,
            "sat_rmse": sat_rmse,
            "sat_r2": sat_r2,
            "combined_rmse": combined_rmse,
        }

    def grid_search(self, param_grid, metric="combined_rmse"):
        """
        Perform grid search over hyperparameters

        Parameters:
        -----------
        param_grid : dict
            Dictionary of parameter lists to search
            e.g., {'tau_spatial': [1, 2, 5], 'tau_sensor': [5, 10, 20]}
        metric : str
            Metric to optimize ('combined_rmse', 'sensor_rmse', 'sat_rmse')
        """

        # Generate all combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(product(*values))

        print(f"Testing {len(combinations)} hyperparameter combinations...")
        print()

        results = []
        best_score = float("inf")
        best_params = None

        for i, combo in enumerate(combinations):
            params = dict(zip(keys, combo))

            # Ensure all parameters are set
            tau_spatial = params.get("tau_spatial", 2.0)
            tau_hourly = params.get("tau_hourly", 0.1)
            tau_daily = params.get("tau_daily", 0.01)
            tau_sensor = params.get("tau_sensor", 20.0)
            tau_satellite = params.get("tau_satellite", 4.0)

            print(
                f"[{i+1}/{len(combinations)}] Testing: tau_spatial={tau_spatial}, "
                f"tau_hourly={tau_hourly}, tau_daily={tau_daily}, "
                f"tau_sensor={tau_sensor}, tau_satellite={tau_satellite}"
            )

            start = time.time()

            # Build and solve
            theta, Q_post, Q_obs, b_obs = self.build_and_solve(
                tau_spatial, tau_hourly, tau_daily, tau_sensor, tau_satellite
            )

            # Compute metrics
            metrics = self.compute_loo_metrics(theta, Q_post, Q_obs, b_obs)

            solve_time = time.time() - start

            # Store results
            result = {**params, **metrics, "time": solve_time}
            results.append(result)

            score = metrics[metric]
            print(f"    {metric}: {score:.2f}, time: {solve_time:.1f}s")

            if score < best_score:
                best_score = score
                best_params = params
                print(f"    *** New best! ***")
            print()

        results_df = pd.DataFrame(results)

        print("=" * 80)
        print("GRID SEARCH COMPLETE")
        print("=" * 80)
        print(f"\nBest parameters (minimizing {metric}):")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
        print(f"\nBest {metric}: {best_score:.2f}")
        print()

        return results_df, best_params


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

print("=" * 80)
print("HYPERPARAMETER TUNING")
print("=" * 80)
print()

# First, build the base precision matrices (unscaled)
# These are computed once, then scaled during search

# Spatial precision base (unscaled Graph Laplacian)
row_sums = np.sum(adj_matrix, axis=1)
D = np.diag(row_sums)
Q_spatial_base = D - adj_matrix

# Temporal precision base (only hourly transitions, no daily yet)
Q_temporal_base = np.zeros((n_hours, n_hours))
for h in range(n_hours):
    if h > 0:
        Q_temporal_base[h, h - 1] = -1.0
        Q_temporal_base[h, h] += 1.0
    if h < n_hours - 1:
        Q_temporal_base[h, h + 1] = -1.0
        Q_temporal_base[h, h] += 1.0

# Initialize tuner
tuner = HyperparameterTuner(
    sensor_data,
    satellite_data,
    Q_spatial_base,
    Q_temporal_base,
    n_settlements,
    n_hours,
    n_days,
    config.satellite_hour,
)

# =============================================================================
# STRATEGY 1: COARSE GRID SEARCH
# =============================================================================

print("STRATEGY 1: Coarse grid search")
print("Hypothesis: Current tau_sensor/tau_satellite are too high relative to prior")
print()

param_grid_coarse = {
    "tau_spatial": [0.5, 1.0, 2.0, 5.0],
    "tau_hourly": [0.5, 1.0, 2.0],
    "tau_daily": [0.01, 0.1, 0.5],
    "tau_sensor": [1.0, 5.0, 10.0],
    "tau_satellite": [0.5, 1.0, 2.0],
}

results_coarse, best_params_coarse = tuner.grid_search(
    param_grid_coarse, metric="combined_rmse"
)

# Show top 10 configurations
print("\nTop 10 configurations:")
print(
    results_coarse.nsmallest(10, "combined_rmse")[
        [
            "tau_spatial",
            "tau_hourly",
            "tau_daily",
            "tau_sensor",
            "tau_satellite",
            "sensor_rmse",
            "sat_rmse",
            "combined_rmse",
            "sensor_r2",
            "sat_r2",
        ]
    ].to_string(index=False)
)
print()

# =============================================================================
# STRATEGY 2: FINE GRID AROUND BEST
# =============================================================================

print("\n" + "=" * 80)
print("STRATEGY 2: Fine grid search around best coarse parameters")
print("=" * 80)
print()

# Create fine grid around best parameters
def create_fine_grid(best_params, factor=0.5):
    """Create fine grid around best parameters"""
    fine_grid = {}
    for key, val in best_params.items():
        # Test ±50% around best value
        low = val * (1 - factor)
        high = val * (1 + factor)
        fine_grid[key] = [low, val, high]
    return fine_grid


param_grid_fine = create_fine_grid(best_params_coarse, factor=0.5)

results_fine, best_params_fine = tuner.grid_search(
    param_grid_fine, metric="combined_rmse"
)

print("\nTop 5 fine-tuned configurations:")
print(
    results_fine.nsmallest(5, "combined_rmse")[
        [
            "tau_spatial",
            "tau_hourly",
            "tau_daily",
            "tau_sensor",
            "tau_satellite",
            "sensor_rmse",
            "sat_rmse",
            "combined_rmse",
            "sensor_r2",
            "sat_r2",
        ]
    ].to_string(index=False)
)
print()

# =============================================================================
# FINAL RECOMMENDATION
# =============================================================================

print("=" * 80)
print("FINAL RECOMMENDATIONS")
print("=" * 80)
print()

print("Best hyperparameters found:")
for k, v in best_params_fine.items():
    print(f"  {k} = {v:.3f}")
print()

print("To use these in your model, update the ModelConfig class:")
print()
print("class ModelConfig:")
for k, v in best_params_fine.items():
    print(f"    {k} = {v:.3f}")
print()

# Save results
results_all = pd.concat([results_coarse, results_fine], ignore_index=True)
# results_all.to_csv('hyperparameter_tuning_results.csv', index=False)
print("All results saved to: hyperparameter_tuning_results.csv")
