"""India-specific calculations: AQI, seasonal classification, diel aggregation, and wind roses."""

import numpy as np
import pandas as pd

import matplotlib as mpl
import cmcrameri

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update(
    {
        "axes.grid": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "Arial",
        "axes.labelweight": "bold",
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 14,
        "axes.titleweight": "bold",
    }
)


def get_aqi(pm25: np.ndarray) -> np.ndarray:
    """Convert PM2.5 concentrations to Indian AQI category codes.

    Args:
        pm25: PM2.5 concentrations [µg/m³].

    Returns:
        Integer AQI codes (0=Good, 1=Moderate, 2=Unhealthy-Sensitive, 3=Unhealthy, 4=Hazardous, 5=Extremely Hazardous).
    """
    aqi = np.zeros_like(np.asarray(pm25))
    aqi[pm25 < 30] = 0
    aqi[(pm25 > 30) & (pm25 <= 60)] = 1
    aqi[(pm25 > 60) & (pm25 <= 90)] = 2
    aqi[(pm25 > 90) & (pm25 <= 120)] = 3
    aqi[(pm25 > 120) & (pm25 <= 250)] = 4
    aqi[pm25 > 250] = 5
    return aqi.astype(int)


def get_season(df: pd.DataFrame, name: bool = True) -> pd.DataFrame:
    """Assign an Indian season label or code to each row based on the datetime index.

    Args:
        df: DataFrame with a datetime index.
        name: If True, return season names; if False, return integer codes (0–3).

    Returns:
        Input DataFrame with an added 'season' column.
    """
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
        df["season"] = "0"
        df = df.set_index(0)
    else:
        df["season"] = "0"

    if name:
        df.loc[df.index.month.isin([1, 2]), "season"] = "Winter"
        df.loc[df.index.month.isin([3, 4, 5]), "season"] = "Pre-Monsoon"
        df.loc[df.index.month.isin([6, 7, 8, 9]), "season"] = "Monsoon"
        df.loc[df.index.month.isin([10, 11, 12]), "season"] = "Post-Monsoon"
    else:
        df.loc[df.index.month.isin([1, 2]), "season"] = 0
        df.loc[df.index.month.isin([3, 4, 5]), "season"] = 1
        df.loc[df.index.month.isin([6, 7, 8, 9]), "season"] = 2
        df.loc[df.index.month.isin([10, 11, 12]), "season"] = 3
        df.season = df.season.astype(int)

    return df


def get_diel(df: pd.DataFrame, agg_func: str = "mean") -> pd.DataFrame:
    """Aggregate a DataFrame by season, sensor, and hour of day.

    Args:
        df: DataFrame with a datetime index; may contain 'season' and/or 'sensor' columns.
        agg_func: Aggregation function name (e.g., 'mean', 'sum').

    Returns:
        Aggregated diel DataFrame.
    """
    if "sensor" in df.columns:
        df["sensor"] = pd.Categorical(
            df["sensor"], categories=df["sensor"].unique(), ordered=True
        )

    if "season" in df.columns and "sensor" in df.columns:
        df_diel = df.groupby(["season", "sensor", df.index.hour]).agg(agg_func)
        for season in df["season"].unique():
            for sensor in df["sensor"].unique():
                df_diel.loc[(season, sensor, 24), :] = df_diel.loc[(season, sensor, 0)]
                df_diel.sort_index(inplace=True)
    elif "season" in df.columns:
        df_diel = df.groupby(["season", df.index.hour]).agg(agg_func)
        for season in df["season"].unique():
            df_diel.loc[(season, 24), :] = df_diel.loc[(season, 0)]
            df_diel.sort_index(inplace=True)
    elif "sensor" in df.columns:
        df_diel = df.groupby(["sensor", df.index.hour]).agg(agg_func)
        for sensor in df["sensor"].unique():
            df_diel.loc[(sensor, 24), :] = df_diel.loc[(sensor, 0)]
            df_diel.sort_index(inplace=True)
    else:
        df_diel = df.groupby(df.index.hour).agg(agg_func)
        df_diel.loc[24] = df_diel.loc[0]
        df_diel.sort_index(inplace=True)

    return df_diel


def get_normalized_diel(df: pd.DataFrame, agg_func: str = "mean") -> pd.DataFrame:
    """Normalize a DataFrame by its daily mean and aggregate by season, sensor, and hour.

    Args:
        df: DataFrame with a datetime index; may contain 'season' and/or 'sensor' columns.
        agg_func: Aggregation function name.

    Returns:
        Normalized diel DataFrame.
    """
    df_daily_mean = df.resample("D").mean()
    df_normalized = df.divide(df_daily_mean.resample("h").ffill())

    if "sensor" in df_normalized.columns:
        df_normalized["sensor"] = pd.Categorical(
            df_normalized["sensor"],
            categories=df_normalized["sensor"].unique(),
            ordered=True,
        )

    if "season" in df_normalized.columns and "sensor" in df_normalized.columns:
        df_diel = df_normalized.groupby(
            ["season", "sensor", df_normalized.index.hour]
        ).agg(agg_func)
        for season in df_normalized["season"].unique():
            for sensor in df_normalized["sensor"].unique():
                df_diel.loc[(season, sensor, 24), :] = df_diel.loc[(season, sensor, 0)]
                df_diel.sort_index(inplace=True)
    elif "season" in df_normalized.columns:
        df_diel = df_normalized.groupby(["season", df_normalized.index.hour]).agg(
            agg_func
        )
        for season in df_normalized["season"].unique():
            df_diel.loc[(season, 24), :] = df_diel.loc[(season, 0)]
            df_diel.sort_index(inplace=True)
    elif "sensor" in df_normalized.columns:
        df_diel = df_normalized.groupby(["sensor", df_normalized.index.hour]).agg(
            agg_func
        )
        for sensor in df_normalized["sensor"].unique():
            df_diel.loc[(sensor, 24), :] = df_diel.loc[(sensor, 0)]
            df_diel.sort_index(inplace=True)
    else:
        df_diel = df_normalized.groupby(df_normalized.index.hour).agg(agg_func)
        df_diel.loc[24] = df_diel.loc[0]
        df_diel.sort_index(inplace=True)

    return df_diel


def get_wind_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Bin wind direction and speed into categorical categories for wind rose plotting.

    Args:
        df: DataFrame with 'wd10' (degrees) and 'spd10' (m/s) columns.

    Returns:
        DataFrame with binned 'wd10' and 'spd10' categorical columns.
    """
    deg_bins = np.arange(0, 360 + 11.25, 11.25)
    deg_labels = np.roll(np.repeat(range(0, 16), 2), 1)
    spd_bins = np.arange(0, 6, 1)
    spd_labels = np.arange(0, 5, 1)

    return pd.concat(
        [
            pd.cut(
                (df.wd10 - 90) % 360, bins=deg_bins, labels=deg_labels, ordered=False
            ),
            pd.cut(df.spd10.apply(np.ceil), bins=spd_bins, labels=spd_labels),
        ],
        axis=1,
    )


def make_windrose(data: pd.DataFrame, ax) -> object:
    """Plot a wind rose on a polar axes object.

    Args:
        data: DataFrame with binned 'wd10' and 'spd10' categorical columns.
        ax: Polar axes to draw on.

    Returns:
        Axes with the wind rose drawn.
    """
    colors = [
        cmcrameri.cm.hawaii.colors[0],
        cmcrameri.cm.hawaii.colors[85],
        cmcrameri.cm.hawaii.colors[171],
        cmcrameri.cm.hawaii.colors[-1],
    ]
    bottom = 0
    for i in range(1, 5):
        if i != 1:
            bottom += (
                data.loc[data.spd10 == i - 1, "wd10"].value_counts().sort_index().values
                / data.groupby("wd10", observed=False).count().sum().values
            )
        ax.bar(
            np.deg2rad(np.arange(0, 360, 22.5)),
            data.loc[data.spd10 == i, "wd10"].value_counts().sort_index().values
            / data.groupby("wd10", observed=False).count().sum().values,
            bottom=bottom,
            width=np.deg2rad(22.5),
            color=colors[i - 1],
        )
        ax.set_theta_zero_location("S")
        ax.set_rmax(0.27)
        ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
        ax.set_xticklabels(["S", "SE", "E", "NE", "N", "NW", "W", "SW"])
        ax.set_rticks([0.05, 0.15, 0.25], ["5%", "15%", "25%"])
        ax.set_rlabel_position(50)

    return ax
