"""
Central Pollution Control Board (CPCB) Regulatory Site Data
Last Updated: June 15, 2024
This script formats data downloaded off the Earthmetry platform
from the CPCB landing page. It checks for anomalies and pushes out a netcdf.
@author: markjcampmier
"""
# Import Packages
import os
import glob
import pandas as pd
import geopandas as gpd
import xarray as xr
import threading


# Define functions
def fetch_reg(in_path, pollutant):
    """
    Reads CPCB data from Feather files.

    :param in_path: The path to the input directory containing the Feather files.
    :type in_path: str
    :param pollutant: The name of the pollutant to read.
    :type pollutant: str
    :returns A Pandas DataFrame containing the CPCB data.
    :rtype: pandas.DataFrame
    """

    lst_path = glob.glob(os.path.join(in_path, f"*_{pollutant}_reg.feather"))
    return lst_path


def fetch_meta(in_path):
    """
    Reads CPCB metadata from a GeoJSON file.

    :param in_path: The path to the GeoJSON file containing the CPCB metadata.
    :type in_path: str
    :returns A GeoPandas GeoDataFrame containing the CPCB metadata.
    :rtype: geopandas.GeoDataFrame
    """
    file = os.path.join(in_path, 'meta_reg.geojson')
    gdf_meta = gpd.read_file(file)
    gdf_meta = gdf_meta.set_index('location_name')

    gdf_meta.loc[:, 'longitude'] = gdf_meta.geometry.x
    gdf_meta.loc[:, 'latitude'] = gdf_meta.geometry.y
    return gdf_meta


def dataset_reg(in_path, pollutant, gdf_meta, ds_list):
    """
    Processes CPCB data for a specified pollutant and stores it in a xarray Dataset.

    :param in_path: Path to the CPCB data directory.
    :type in_path: str
    :param pollutant: Name of the pollutant to process.
    :type pollutant: str
    :param gdf_meta: GeoPandas GeoDataFrame containing the CPCB metadata.
    :type gdf_meta: GeoPandas GeoDataFrame
    :param ds_list: List of dictionaries containing the data sources.
    :type ds_list: list
    :return: None
    """

    # Define a dictionary of pollutant long names
    pollutant_long_names = {
        "CO": "Carbon Monoxide",
        "NO2": "Nitrogen Dioxide",
        "SO2": "Sulfur Dioxide",
        "O3": "Ozone",
        "PM25": "PM2.5",
        "PM10": "PM10",
    }

    # Fetch paths to data files for the pollutant
    lst_path = fetch_reg(in_path, pollutant)

    # Read pollutant data from Feather files into a Pandas DataFrame
    df = pd.concat(
        [
            pd.read_feather(
                file, columns=["observation_hour_start", "location_name", "average_value_in_hour"]
            )
            for file in lst_path
        ]
    )

    # Pivot the data into a time-series format
    df = df.pivot_table(
        index="observation_hour_start",
        columns="location_name",
        values="average_value_in_hour",
    )

    # Filter data for locations present in the metadata
    valid_locations = df.columns.intersection(gdf_meta.index)
    df = df.loc[:, valid_locations]

    # Resample data to hourly intervals
    df = df.resample("1H").mean()
    df.index = df.index - pd.Timedelta('1H')

    # Create a xarray Dataset with the pollutant data
    ds = xr.Dataset(
        {pollutant: (("time", "site"), df.values.astype(float))},
        coords={
            "time": ("time", df.index),
            "site": ("site", df.columns.astype(str)),
            "state": ("site", gdf_meta.loc[df.columns.astype(str), 'state_name'].values),
            "district": ("site", gdf_meta.loc[df.columns.astype(str), 'district_name'].values),
            "settlement_name": ("site", gdf_meta.loc[df.columns.astype(str), 'city_name'].values),
        })

    # Append the processed dataset to the list
    ds_list.append(ds)

    return None


def format_reg(in_path):
    """
    Formats and combines CPCB data for various pollutants into a single dataset.

    :param in_path: Path to the directory containing CPCB data files.
    :type in_path: str
    :return: A merged xarray Dataset with processed pollutant data.
    :rtype: xarray.dataset
    """

    # Retrieve location metadata from the input path
    gdf_meta = fetch_meta(in_path)

    # Initialize an empty list to store processed datasets
    ds_list = []
    threads = []

    # Create threads for processing data for each pollutant
    for pollutant in ["CO", "NO2", "SO2", "O3", "PM25", "PM10"]:
        thread = threading.Thread(
            target=dataset_reg, args=(in_path, pollutant, gdf_meta, ds_list)
        )
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish processing
    for thread in threads:
        thread.join()

    # Merge the processed datasets into a single xarray Dataset
    ds_cpcb = xr.merge(ds_list)

    ds_cpcb = xr.merge([gdf_meta.loc[ds_cpcb.site.values, 'latitude'].to_xarray().rename(
        {'location_name': 'site'}),
        gdf_meta.loc[ds_cpcb.site.values, 'longitude'].to_xarray().rename(
            {'location_name': 'site'}),
        ds_cpcb])

    ds_cpcb = ds_cpcb.rename_vars({'PM10': 'pm10',
                                   'CO': 'co',
                                   'SO2': 'so2',
                                   'O3': 'o3',
                                   'PM25': 'pm25',
                                   'NO2': 'no2'})

    ds_cpcb = ds_cpcb.sortby('time')
    ds_cpcb = ds_cpcb.transpose('site', 'time')
    ds_cpcb = ds_cpcb.set_coords(['latitude', 'longitude'])

    ds_cpcb['pm25'] = ds_cpcb['pm25'].assign_attrs(long_name='PM2.5',
                                                   units='ug/m^3',
                                                   description='particulate matter less than 2.5 microns')
    ds_cpcb['pm10'] = ds_cpcb['pm10'].assign_attrs(long_name='PM10',
                                                   units='ug/m^3',
                                                   description='particulate matter less than 10 microns')
    ds_cpcb['no2'] = ds_cpcb['no2'].assign_attrs(long_name='nitrogen dioxide',
                                                 units='ug/m^3',
                                                 description='nitrogen dioxide concentration')
    ds_cpcb['so2'] = ds_cpcb['so2'].assign_attrs(long_name='sulfur dioxide',
                                                 units='ug/m^3',
                                                 description='nitrogen dioxide concentration')
    ds_cpcb['co'] = ds_cpcb['co'].assign_attrs(long_name='carbon monoxide',
                                               units='mg/m^3',
                                               description='carbon monoxide concentration - units are mg/m^3')
    ds_cpcb['o3'] = ds_cpcb['o3'].assign_attrs(long_name='ozone',
                                               units='ug/m^3',
                                               description='ozone concentration')

    ds_cpcb['settlement_name'] = ds_cpcb['settlement_name'].assign_attrs(long_name='settlement name',
                                                                         description='settlement name from project metadata')
    ds_cpcb['district'] = ds_cpcb['district'].assign_attrs(long_name='district name',
                                                           description='district name from shapefile')
    ds_cpcb['state'] = ds_cpcb['state'].assign_attrs(long_name='state name',
                                                     description='state name from shapefile')
    ds_cpcb['latitude'] = ds_cpcb['latitude'].assign_attrs(long_name='latitude',
                                                           units='degrees_north')
    ds_cpcb['longitude'] = ds_cpcb['longitude'].assign_attrs(long_name='longitude',
                                                             units='degrees_east')

    ds_cpcb.attrs['history'] = 'munged'

    return ds_cpcb
