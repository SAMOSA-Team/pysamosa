"""
India-specific Calculations
Last Updated: Jan 20, 2024
This script supplies the basic functions for performing calculations
and data operations specific to India, such as INAQI.
@author: markjcampmier
"""
# Import Packages
import numpy as np
import pandas as pd

import matplotlib as mpl
import cmcrameri

# Add Matplotlib Formatting for Windrose Plots
mpl.rcParams.update(mpl.rcParamsDefault)

mpl.rcParams['axes.grid'] = True
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams['axes.titleweight'] = 'bold'


def get_aqi(pm25):
    """
    This function converts PM2.5 concentrations to AQI values (Indian).

    :param pm25: The PM2.5 concentrations in μg/m^3.
    :type pm25: numpy.array
    :return: The AQI values (Indian).
    :rtype: numpy.array
    """
    aqi = np.zeros_like(np.asarray(pm25))
    aqi[pm25 < 30] = 0  # good
    aqi[(pm25 > 30) & (pm25 <= 60)] = 1  # moderate
    aqi[(pm25 > 60) & (pm25 <= 90)] = 2  # unhealthy for sensitive groups
    aqi[(pm25 > 90) & (pm25 <= 120)] = 3  # unhealthy
    aqi[(pm25 > 120) & (pm25 <= 250)] = 4  # hazardous
    aqi[pm25 > 250] = 5  # extremely hazardous

    aqi = aqi.astype(int)

    return aqi


def get_season(df, name=True):
    """
    This function assigns a season to each row in a DataFrame based on the month.

    :param df: A Pandas DataFrame.
    :type df: pandas.DataFrame
    :param name: Whether to return season as a string or integer [0-3].
    :type name: bool
    :return df: A new DataFrame with an additional column called `season`.
    :rtype: pandas.DataFrame
    """

    # Create a new column called `season`.
    if type(df) != pd.DataFrame:
        df = pd.DataFrame(df)
        df['season'] = '0'
        df = df.set_index(0)
    else:
        df['season'] = '0'

    # Assign the seasons to each row.
    if name:
        df.loc[df.index.month.isin([1, 2]), 'season'] = 'Winter'  # Winter
        df.loc[df.index.month.isin([3, 4, 5]), 'season'] = 'Pre-Monsoon'  # Spring
        df.loc[df.index.month.isin([6, 7, 8, 9]), 'season'] = 'Monsoon'  # Summer
        df.loc[df.index.month.isin([10, 11, 12]), 'season'] = 'Post-Monsoon'  # Fall
    else:
        df.loc[df.index.month.isin([1, 2]), 'season'] = 0  # Winter
        df.loc[df.index.month.isin([3, 4, 5]), 'season'] = 1  # Spring
        df.loc[df.index.month.isin([6, 7, 8, 9]), 'season'] = 2  # Summer
        df.loc[df.index.month.isin([10, 11, 12]), 'season'] = 3  # Fall
    return df


def get_diel(df, agg_func='mean'):
    """
    Aggregate a DataFrame by season, sensor, and hour.

    :param df: An input DataFrame with a datetime index. It may have a 'season' column
               or 'sensor' column, or both.
    :type df: pandas.DataFrame
    :param agg_func: A string indicating the name of the aggregation function
                     to use i.e. 'mean', 'sum'. Defaults to 'mean'.
    :type agg_func: str
    :returns: A DataFrame with the aggregated results.
    :rtype: pandas.DataFrame
    """
    if 'sensor' in df.columns:
        df['sensor'] = pd.Categorical(df['sensor'], categories=df['sensor'].unique(), ordered=True)

    # handle both 'season' and 'sensor' categorical column
    if 'season' in df.columns and 'sensor' in df.columns:
        df_diel = df.groupby(['season', 'sensor', df.index.hour]).agg(agg_func)
        for season in df['season'].unique():
            for sensor in df['sensor'].unique():
                df_diel.loc[(season, sensor, 24), :] = df_diel.loc[(season, sensor, 0)]
                df_diel.sort_index(inplace=True)
    # handle just 'season' in columns
    elif 'season' in df.columns:
        df_diel = df.groupby(['season', df.index.hour]).agg(agg_func)
        for season in df['season'].unique():
            df_diel.loc[(season, 24), :] = df_diel.loc[(season, 0)]
            df_diel.sort_index(inplace=True)
    # handle just 'sensor' in column
    elif 'sensor' in df.columns:
        df_diel = df.groupby(['sensor', df.index.hour]).agg(agg_func)
        for sensor in df['sensor'].unique():
            df_diel.loc[(sensor, 24), :] = df_diel.loc[(sensor, 0)]
            df_diel.sort_index(inplace=True)
    # default aggregation by hour
    else:
        df_diel = df.groupby(df.index.hour).agg(agg_func)
        df_diel.loc[24] = df_diel.loc[0]
        df_diel.sort_index(inplace=True)

    return df_diel


def get_normalized_diel(df, agg_func='mean'):
    """
    Normalize a DataFrame by its daily mean and then aggregate by season, sensor, and hour.

    :param df: An input DataFrame with a datetime index. It may have a 'season' column
               or 'sensor' column, or both.
    :type df: pandas.DataFrame
    :param agg_func: A string indicating the name of the aggregation function
                     to use i.e. 'mean', 'sum'. Defaults to 'mean'.
    :type agg_func: str
    :returns: A DataFrame with the aggregated results normalized by daily mean.
    :rtype: pandas.DataFrame
    """
    # Calculate daily mean and normalize data by it
    df_daily_mean = df.resample('D').mean()
    df_normalized = df.divide(df_daily_mean.resample('H').ffill())

    if 'sensor' in df_normalized.columns:
        df_normalized['sensor'] = pd.Categorical(df_normalized['sensor'], categories=df_normalized['sensor'].unique(),
                                                 ordered=True)

    # handle both 'season' and 'sensor' categorical column
    if 'season' in df_normalized.columns and 'sensor' in df_normalized.columns:
        df_diel_normalized = df_normalized.groupby(['season', 'sensor', df_normalized.index.hour]).agg(agg_func)
        for season in df_normalized['season'].unique():
            for sensor in df_normalized['sensor'].unique():
                df_diel_normalized.loc[(season, sensor, 24), :] = df_diel_normalized.loc[(season, sensor, 0)]
                df_diel_normalized.sort_index(inplace=True)
    # handle just 'season' in columns
    elif 'season' in df_normalized.columns:
        df_diel_normalized = df_normalized.groupby(['season', df_normalized.index.hour]).agg(agg_func)
        for season in df_normalized['season'].unique():
            df_diel_normalized.loc[(season, 24), :] = df_diel_normalized.loc[(season, 0)]
            df_diel_normalized.sort_index(inplace=True)
    # handle just 'sensor' in column
    elif 'sensor' in df_normalized.columns:
        df_diel_normalized = df_normalized.groupby(['sensor', df_normalized.index.hour]).agg(agg_func)
        for sensor in df_normalized['sensor'].unique():
            df_diel_normalized.loc[(sensor, 24), :] = df_diel_normalized.loc[(sensor, 0)]
            df_diel_normalized.sort_index(inplace=True)
    # default aggregation by hour
    else:
        df_diel_normalized = df_normalized.groupby(df_normalized.index.hour).agg(agg_func)
        df_diel_normalized.loc[24] = df_diel_normalized.loc[0]
        df_diel_normalized.sort_index(inplace=True)

    return df_diel_normalized


def get_wind_bins(df):
    """
    Convert wind direction and wind speed data into wind bins.

    :param df: The input dataframe with wind direction and speed data. The dataframe must contain
               'wd10' and 'spd10' columns. 'wd10' refers to wind direction and 'spd10'
               refers to wind speed.
    :type df: pandas.DataFrame
    :returns: A DataFrame with two new categorical columns 'wd10' and 'spd10' representing
              wind direction bins and speed bins respectively.
    :rtype: pandas.DataFrame
    """
    deg_bins = np.arange(0, 360 + 11.25, 11.25)
    deg_labels = np.roll(np.repeat(range(0, 16), 2), 1)
    spd_bins = np.arange(0, 6, 1)
    spd_labels = np.arange(0, 5, 1)

    df_bins = pd.concat([pd.cut((df.wd10 - 90) % 360, bins=deg_bins, labels=deg_labels, ordered=False),
                         pd.cut(df.spd10.apply(np.ceil), bins=spd_bins, labels=spd_labels)],
                        axis=1)
    return df_bins


def make_windrose(data, ax):
    """
    Generates a windrose plot for a specific season.

    :param data: Dataframe containing wind direction and speed frequency values.
                 Dataframe must contain 'wd10' and 'spd10' columns.
    :type data: pandas.DataFrame
    :param ax: The axes object on which the plot will be drawn.
    :type ax: matplotlib.axes.Axes
    :returns: Returns axes with a plot.
    :rtype: matplotlib.axes.Axes
    """

    colors = [cmcrameri.cm.hawaii.colors[0],
              cmcrameri.cm.hawaii.colors[85],
              cmcrameri.cm.hawaii.colors[171],
              cmcrameri.cm.hawaii.colors[-1]]
    bottom = 0
    for i in range(1, 5):
        if i != 1:
            bottom += data.loc[data.spd10 == i - 1, 'wd10'].value_counts().sort_index().values / data.groupby('wd10',
                                                                                                              observed=False).count().sum().values

        ax.bar(np.deg2rad(np.arange(0, 360, 22.5)),
               data.loc[data.spd10 == i, 'wd10'].value_counts().sort_index().values / data.groupby('wd10',
                                                                                                   observed=False).count().sum().values,
               bottom=bottom,
               width=np.deg2rad(22.5), color=colors[i - 1])

        ax.set_theta_zero_location("S")
        ax.set_rmax(0.27)
        ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
        ax.set_xticklabels(['S', 'SE', 'E', 'NE', 'N', 'NW', 'W', 'SW'])
        ax.set_rticks(np.arange(0, 0.30, 0.05), ['', '5%', '10%', '15%', '20%', '25%'])
        ax.set_rlabel_position(50)

    return ax
