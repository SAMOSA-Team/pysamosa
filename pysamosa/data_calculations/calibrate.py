"""PurpleAir sensor calibration models (linear, nonlinear, sklearn)."""

import pandas as pd
import xarray as xr

from sklearn.linear_model import Lasso
from sklearn.base import BaseEstimator

from pysamosa.data_calculations import india


class SensorCalibrator:
    """PurpleAir sensor calibration with historical and fitted model support."""

    HISTORICAL_MODELS = {
        "campmier_delhi": {
            "type": "linear",
            "params": {"pa_raw": 0.546, "rh": -0.936, "intercept": 50.3},
        },
        "campmier_hamirpur": {
            "type": "linear",
            "params": {"pa_raw": 0.496, "rh": -0.296, "intercept": 22.0},
        },
        "campmier_bengaluru": {
            "type": "linear",
            "params": {"pa_raw": 0.515, "rh": -0.139, "intercept": 14.1},
        },
        "epa_usa": {
            "type": "linear",
            "params": {"pa_raw": 0.546, "rh": -0.0862, "intercept": 5.75},
        },
    }

    def __init__(self, sites: list, seasons: list):
        self.site_codes = {site: f"s{i:02d}" for i, site in enumerate(sorted(sites), 1)}
        self.season_codes = {
            season: f"t{i:02d}" for i, season in enumerate(sorted(seasons), 1)
        }
        self.site_decode = {v: k for k, v in self.site_codes.items()}
        self.season_decode = {v: k for k, v in self.season_codes.items()}
        self.models = {"historical": self.HISTORICAL_MODELS.copy(), "fitted": {}}

    @staticmethod
    def get_pa_raw_columns(data: pd.DataFrame) -> list[str]:
        """Return all column names starting with ``pa_raw_``.

        Args:
            data: DataFrame to search.

        Returns:
            List of matching column names.
        """
        return [col for col in data.columns if col.startswith("pa_raw_")]

    @staticmethod
    def linear_model(
        x: pd.DataFrame, pa_raw_col: str, b_pa: float, b_rh: float, b_int: float
    ) -> pd.Series:
        """Apply a linear calibration: pm25 = pa_raw * b_pa + rh * b_rh + b_int.

        Args:
            x: Input DataFrame with pa_raw and rh columns.
            pa_raw_col: Column name for the raw PA measurement.
            b_pa: Coefficient for the PA raw channel.
            b_rh: Coefficient for relative humidity.
            b_int: Intercept.

        Returns:
            Calibrated PM2.5 values.
        """
        return x[pa_raw_col] * b_pa + x["rh"] * b_rh + b_int

    @staticmethod
    def nonlinear_model(
        x: pd.DataFrame, pa_raw_col: str, alpha: float, beta: float
    ) -> pd.Series:
        """Apply the Chakrabarti nonlinear calibration model.

        Args:
            x: Input DataFrame with pa_raw and rh columns.
            pa_raw_col: Column name for the raw PA measurement.
            alpha: Scaling coefficient.
            beta: Hygroscopic growth coefficient.

        Returns:
            Calibrated PM2.5 values.
        """
        return (x[pa_raw_col] * alpha) / (1 + beta * (x["rh"] / (1 - x["rh"])))

    @classmethod
    def apply_calibration(
        cls,
        data: pd.DataFrame,
        model_name: str,
        pa_raw_col: str | None = None,
    ) -> pd.Series:
        """Apply a named historical calibration model to data.

        Args:
            data: Input DataFrame.
            model_name: Key in HISTORICAL_MODELS.
            pa_raw_col: PA raw column to use; auto-detected if None.

        Returns:
            Calibrated PM2.5 series.
        """
        if pa_raw_col is None:
            pa_raw_cols = [col for col in data.columns if col.startswith("pa_raw_")]
            if not pa_raw_cols:
                raise ValueError("No pa_raw columns found in data")
            pa_raw_col = pa_raw_cols[0]

        if pa_raw_col not in data.columns:
            raise ValueError(f"Required column {pa_raw_col} not found in data")

        model = cls.HISTORICAL_MODELS.get(model_name)
        if model is None:
            raise ValueError(f"Unknown model: {model_name}")

        params = model["params"]
        if model["type"] == "linear":
            return cls.linear_model(
                data, pa_raw_col, params["pa_raw"], params["rh"], params["intercept"]
            )
        if model["type"] == "nonlinear":
            return cls.nonlinear_model(
                data, pa_raw_col, params["alpha"], params["beta"]
            )
        raise ValueError(f"Unsupported model type: {model['type']}")

    def apply_fitted_calibration(
        self,
        data: pd.DataFrame,
        model_name: str,
        pa_raw_col: str | None = None,
    ) -> pd.Series:
        """Apply a fitted or historical calibration model.

        Args:
            data: Input DataFrame.
            model_name: Model key in the fitted or historical models dict.
            pa_raw_col: PA raw column; auto-detected if None.

        Returns:
            Calibrated PM2.5 series.
        """
        if model_name not in self.models["fitted"]:
            return self.apply_calibration(data, model_name, pa_raw_col)

        model = self.models["fitted"][model_name]

        if pa_raw_col is None:
            pa_raw_cols = [col for col in data.columns if col.startswith("pa_raw_")]
            if not pa_raw_cols:
                raise ValueError("No pa_raw columns found in data")
            pa_raw_col = pa_raw_cols[0]

        params = model["params"]
        if model["type"] == "linear":
            return self.linear_model(
                data, pa_raw_col, params["pa_raw"], params["rh"], params["intercept"]
            )
        if model["type"] == "nonlinear":
            return self.nonlinear_model(
                data, pa_raw_col, params["alpha"], params["beta"]
            )
        if model["type"] == "sklearn":
            return pd.Series(
                self.sklearn_model(  # pylint: disable=no-member
                    data, pa_raw_col, params["model"]
                ),
                index=data.index,
            )
        raise ValueError(f"Unsupported model type: {model['type']}")

    def meta_model(
        self,
        data: pd.DataFrame,
        train_data: pd.DataFrame,
        model_names: list[str],
        method: str = "mean",
    ) -> pd.Series:
        """Ensemble multiple calibration models into a single prediction.

        Args:
            data: Input DataFrame.
            train_data: Training data used when method='best_fit'.
            model_names: List of model keys to ensemble.
            method: Ensemble method: 'mean', 'median', or 'best_fit'.

        Returns:
            Ensemble calibrated PM2.5 series.
        """
        predictions = pd.DataFrame(index=data.index)
        for name in model_names:
            predictions[name] = self.apply_calibration(data, name)

        if method == "mean":
            return predictions.mean(axis=1)
        if method == "median":
            return predictions.median(axis=1)
        if method == "best_fit":
            resampled = predictions.resample(train_data.index.freq).mean()
            meta_mod = Lasso(fit_intercept=False)
            meta_mod.fit(resampled, train_data["pm25"])
            return pd.Series(meta_mod.predict(predictions), index=predictions.index)
        raise ValueError(f"Unknown method: {method}")

    def pred_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions from all historical and fitted models.

        Args:
            data: Input DataFrame.

        Returns:
            DataFrame with one column per model plus ensemble statistics.
        """
        results = pd.DataFrame(index=data.index)

        for model_name in self.models["historical"]:
            results[f"historical_{model_name}"] = self.apply_calibration(
                data, model_name
            )
        for model_name in self.models["fitted"]:
            results[f"fitted_{model_name}"] = self.apply_calibration(data, model_name)

        for col in ["site", "season"]:
            if col in data.columns:
                results[col] = data[col]

        pred_cols = results.filter(regex="^(historical|fitted)_")
        results["mean_prediction"] = pred_cols.mean(axis=1)
        results["median_prediction"] = pred_cols.median(axis=1)
        results["std_prediction"] = pred_cols.std(axis=1)

        for col in self.get_pa_raw_columns(data):
            results[f"raw_{col}"] = data[col]

        return results.sort_index()

    def pred_to_xarray(self, df_calibrated: pd.DataFrame) -> xr.Dataset:
        """Convert calibrated predictions DataFrame to an xarray Dataset.

        Args:
            df_calibrated: DataFrame with 'site' and 'time' index levels.

        Returns:
            Dataset with season coordinate.
        """
        df = (
            df_calibrated.reset_index()
            .drop(["season"], axis=1)
            .set_index(["site", "time"])
        )
        ds = df.to_xarray()
        ds = ds.assign_coords(season=("time", india.get_season(ds.time, False).season))
        return ds

    def models_to_xarray(self) -> xr.Dataset:
        """Export model parameters to an xarray Dataset.

        Returns:
            Dataset with one DataArray per model containing its parameters.
        """
        data_vars = {}
        for category in ["historical", "fitted"]:
            for model_name, model_info in self.models[category].items():
                if model_info["type"] in ["linear", "nonlinear"]:
                    params_dict = model_info["params"].copy()
                    params_dict["pa_raw_var"] = model_info.get("pa_raw_var", "")
                    data_vars[f"{category}_{model_name}"] = xr.DataArray(
                        data=list(params_dict.values()),
                        dims=["parameter"],
                        coords={"parameter": list(params_dict.keys())},
                    )
        return xr.Dataset(data_vars)


def run_calibration_pipeline(
    calibrator: SensorCalibrator,
    data: pd.DataFrame,
    sites: list[str] | None = None,
    seasons: list[str] | None = None,
    model_types: list[str] | None = None,
    sklearn_models: list[BaseEstimator] | None = None,
    pa_raw_vars: list[str] | None = None,
) -> list[str]:
    """Fit calibration models for all site/season/variable combinations.

    Args:
        calibrator: Initialized SensorCalibrator instance.
        data: Input DataFrame with 'site', 'season', and PA raw columns.
        sites: Sites to fit; defaults to all unique sites in data.
        seasons: Seasons to fit; defaults to all unique seasons in data.
        model_types: Model types to fit ('linear', 'nonlinear').
        sklearn_models: Additional sklearn estimator instances to fit.
        pa_raw_vars: PA raw column names; auto-detected if None.

    Returns:
        List of fitted model names.
    """
    if model_types is None:
        model_types = ["linear", "nonlinear"]
    if sites is None:
        sites = data["site"].unique()
    if seasons is None:
        seasons = data["season"].unique()
    if pa_raw_vars is None:
        pa_raw_vars = calibrator.get_pa_raw_columns(data)

    fitted_models = []

    for site in sites:
        for season in seasons:
            for pa_raw_var in pa_raw_vars:
                mask = (data["site"] == site) & (data["season"] == season)
                subset = data[mask].copy()
                if len(subset) == 0:
                    continue

                if "linear" in model_types:
                    calibrator.fit_linear_model(subset, pa_raw_var)
                    fitted_models.append(
                        calibrator.generate_model_name(subset, "linear", pa_raw_var)
                    )

                if "nonlinear" in model_types:
                    calibrator.fit_nonlinear_model(subset, pa_raw_var)
                    fitted_models.append(
                        calibrator.generate_model_name(subset, "nonlinear", pa_raw_var)
                    )

                if sklearn_models:
                    for model in sklearn_models:
                        calibrator.fit_sklearn_model(subset, pa_raw_var, model)
                        fitted_models.append(
                            calibrator.generate_model_name(
                                subset, model.__class__.__name__, pa_raw_var
                            )
                        )

    return fitted_models
