import pandas as pd
import xarray as xr

from sklearn.linear_model import Lasso
from sklearn.base import BaseEstimator

from typing import List, Optional

from pysamosa.data_calculations import india


class SensorCalibrator:
    # Historical models defined as a class variable
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
        self.site_codes = {
            site: f"s{i: 02d}" for i, site in enumerate(sorted(sites), 1)
        }
        self.season_codes = {
            season: f"t{i: 02d}" for i, season in enumerate(sorted(seasons), 1)
        }

        self.site_decode = {v: k for k, v in self.site_codes.items()}
        self.season_decode = {v: k for k, v in self.season_codes.items()}

        self.models = {
            "historical": self.HISTORICAL_MODELS.copy(),
            "fitted": {},
        }

    @staticmethod
    def get_pa_raw_columns(data: pd.DataFrame) -> List[str]:
        """Return all columns that start with 'pa_raw_'"""
        return [col for col in data.columns if col.startswith("pa_raw_")]

    @staticmethod
    def linear_model(
        x: pd.DataFrame, pa_raw_col: str, b_pa: float, b_rh: float, b_int: float
    ):
        return x[pa_raw_col] * b_pa + x["rh"] * b_rh + b_int

    @staticmethod
    def nonlinear_model(x: pd.DataFrame, pa_raw_col: str, alpha: float, beta: float):
        return (x[pa_raw_col] * alpha) / (1 + beta * (x["rh"] / (1 - x["rh"])))

    @classmethod
    def apply_calibration(
        cls, data: pd.DataFrame, model_name: str, pa_raw_col: Optional[str] = None
    ) -> pd.Series:
        """
        Unified calibration method that works both as a static method and instance method.
        """
        # If pa_raw_col is not specified, try to detect it from the data
        if pa_raw_col is None:
            pa_raw_cols = [col for col in data.columns if col.startswith("pa_raw_")]
            if len(pa_raw_cols) == 0:
                raise ValueError("No pa_raw columns found in data")
            pa_raw_col = pa_raw_cols[0]

        # Ensure the required pa_raw column exists
        if pa_raw_col not in data.columns:  # noqa: E713
            raise ValueError(  # noqa: E713
                f"Required column {pa_raw_col} not found in data"  # noqa: E713
            )  # noqa: E713

        # Get model from historical models
        model = cls.HISTORICAL_MODELS.get(model_name)
        if model is None:
            raise ValueError(f"Unknown model: {model_name}")

        if model["type"] == "linear":
            params = model["params"]
            return cls.linear_model(
                data, pa_raw_col, params["pa_raw"], params["rh"], params["intercept"]
            )
        elif model["type"] == "nonlinear":
            params = model["params"]
            return cls.nonlinear_model(
                data, pa_raw_col, params["alpha"], params["beta"]
            )
        else:
            raise ValueError(f"Unsupported model type: {model['type']}")

    def apply_fitted_calibration(
        self, data: pd.DataFrame, model_name: str, pa_raw_col: Optional[str] = None
    ) -> pd.Series:
        """Instance method for applying calibration to fitted models"""
        if model_name not in self.models["fitted"]:
            return self.apply_calibration(data, model_name, pa_raw_col)

        model = self.models["fitted"][model_name]

        if pa_raw_col is None:
            pa_raw_cols = [col for col in data.columns if col.startswith("pa_raw_")]
            if len(pa_raw_cols) == 0:
                raise ValueError("No pa_raw columns found in data")
            pa_raw_col = pa_raw_cols[0]

        if model["type"] == "linear":
            params = model["params"]
            return self.linear_model(
                data, pa_raw_col, params["pa_raw"], params["rh"], params["intercept"]
            )
        elif model["type"] == "nonlinear":
            params = model["params"]
            return self.nonlinear_model(
                data, pa_raw_col, params["alpha"], params["beta"]
            )
        elif model["type"] == "sklearn":
            params = model["params"]
            return pd.Series(
                self.sklearn_model(data, pa_raw_col, params["model"]), index=data.index
            )
        else:
            raise ValueError(f"Unsupported model type: {model['type']}")

    def meta_model(
        self,
        data: pd.DataFrame,
        train_data: pd.DataFrame,
        model_names: List[str],
        method: str = "mean",
    ) -> pd.Series:
        predictions = pd.DataFrame(index=data.index)
        for name in model_names:
            predictions[name] = self.apply_calibration(data, name)

        if method == "mean":
            return predictions.mean(axis=1)
        elif method == "median":
            return predictions.median(axis=1)
        elif method == "best_fit":
            resampled_predictions = predictions.resample(train_data.index.freq).mean()
            meta_mod = Lasso(fit_intercept=False)
            meta_mod.fit(resampled_predictions, train_data["pm25"])
            return pd.Series(meta_mod.predict(predictions), index=predictions.index)

    def pred_all(self, data: pd.DataFrame) -> pd.DataFrame:
        results = pd.DataFrame(index=data.index)

        for model_name in self.models["historical"].keys():
            results[f"historical_{model_name}"] = self.apply_calibration(
                data, model_name
            )

        for model_name in self.models["fitted"].keys():
            results[f"fitted_{model_name}"] = self.apply_calibration(data, model_name)

        metadata_cols = ["site", "season"]
        for col in metadata_cols:
            if col in data.columns:
                results[col] = data[col]

        results["mean_prediction"] = results.filter(regex="^(historical|fitted)_").mean(
            axis=1
        )
        results["median_prediction"] = results.filter(
            regex="^(historical|fitted)_"
        ).median(axis=1)
        results["std_prediction"] = results.filter(regex="^(historical|fitted)_").std(
            axis=1
        )

        # Include all pa_raw variables in results
        pa_raw_cols = self.get_pa_raw_columns(data)
        for col in pa_raw_cols:
            results[f"raw_{col}"] = data[col]

        results = results.sort_index()
        return results

    def pred_to_xarray(self, df_calibrated: pd.DataFrame) -> xr.Dataset:
        df = (
            df_calibrated.reset_index()
            .drop(["season"], axis=1)
            .set_index(["site", "time"])
        )
        ds = df.to_xarray()

        ds = ds.assign_coords(season=("time", india.get_season(ds.time, False).season))
        return ds

    def models_to_xarray(self) -> xr.Dataset:
        data_vars = {}

        for category in ["historical", "fitted"]:
            for model_name, model_info in self.models[category].items():
                if model_info["type"] in ["linear", "nonlinear"]:
                    var_name = f"{category}_{model_name}"
                    params_dict = model_info["params"].copy()
                    params_dict["pa_raw_var"] = model_info["pa_raw_var"]
                    data_vars[var_name] = xr.DataArray(
                        data=list(params_dict.values()),
                        dims=["parameter"],
                        coords={"parameter": list(params_dict.keys())},
                    )

        return xr.Dataset(data_vars)


def run_calibration_pipeline(
    calibrator: SensorCalibrator,
    data: pd.DataFrame,
    sites: Optional[List[str]] = None,
    seasons: Optional[List[str]] = None,
    model_types: Optional[List[str]] = ["linear", "nonlinear"],
    sklearn_models: Optional[List[BaseEstimator]] = None,
    pa_raw_vars: Optional[List[str]] = None,
) -> List[str]:
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
                    model_name = calibrator.generate_model_name(
                        subset, "linear", pa_raw_var
                    )
                    fitted_models.append(model_name)

                if "nonlinear" in model_types:
                    calibrator.fit_nonlinear_model(subset, pa_raw_var)
                    model_name = calibrator.generate_model_name(
                        subset, "nonlinear", pa_raw_var
                    )
                    fitted_models.append(model_name)

                if sklearn_models:
                    for model in sklearn_models:
                        calibrator.fit_sklearn_model(subset, pa_raw_var, model)
                        model_name = calibrator.generate_model_name(
                            subset, model.__class__.__name__, pa_raw_var
                        )
                        fitted_models.append(model_name)

    return fitted_models
