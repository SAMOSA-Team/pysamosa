"""
QuantAQ
Last Updated: June 19, 2024
This script formats the QuantAQ data downloaded via the QuantAQ API.
@author: markjcampmier
"""
# Import Packages
import os
import glob
import pandas as pd
import xarray as xr


# Define Functions
def index_meta(in_path):
    """
    Formats the metadata file into a xarray compatible dataset to merge
    with the QuantAQ dataset.

    :param in_path: The path to the directory containing the QuantAQ data files.
    :type in_path: str
    :returns ds_meta: Dataset of QuantAQ metadata.
    :rtype: xarray.Dataset
    """

    df_meta = pd.read_feather(os.path.join(in_path, 'meta.feather')).set_index('sensor')
    ds_meta = df_meta.to_xarray().set_coords(['latitude', 'longitude', 'site'])
    return ds_meta


def read_quantaq(file_processed, file_flag):
    """
    Formats QuantAQ data into a xarray compatible dataset to merge.

    :param file_processed:
    :type file_processed: str
    :param file_flag:
    :type file_flag: str
    :return: quantaq dataset
    :rtype: xarray.Dataset
    """

    df_qaq = pd.read_feather(file_processed,
                             columns=['time', 'pm1', 'pm25', 'pm10', 'sn', 'met.rh', 'met.temp']
                             ).set_index('time').sort_index()

    df_flag = pd.read_feather(file_flag,
                              columns=['timestamp_local', 'flag', 'sn']
                              ).rename(columns={'timestamp_local': 'time'}
                                       ).set_index('time').sort_index()

    df_qaq = pd.concat([df_qaq, df_flag])

    df_qaq = df_qaq.resample('1H').agg({'pm1': 'mean',
                                        'pm25': 'mean',
                                        'pm10': 'mean',
                                        'met.rh': 'mean',
                                        'met.temp': 'mean',
                                        'flag': 'sum',
                                        'sn': 'first'})

    df_qaq = df_qaq.dropna(subset=['pm1', 'pm25', 'pm10']).reset_index().set_index(['sn', 'time'])

    return df_qaq.to_xarray()


def index_quantaq(in_path):
    """
    Combines QuantAQ datasets into a single xarray.

    :param in_path:
    :type in_path: str
    :return ds_qaq: Dataset of QuantAQ data.
    :rtype ds_qaq: xarray.Dataset
    """

    file_list = glob.glob(os.path.join(in_path, '*_processed.feather'))

    ds_list = [read_quantaq(file_processed, file_processed.replace('_processed', '')) for file_processed in file_list]
    ds_qaq = xr.merge(ds_list)

    ds_qaq = ds_qaq.rename({'sn': 'sensor',
                            'pm1': 'pm1_qaq',
                            'pm25': 'pm25_qaq',
                            'pm10': 'pm10_qaq',
                            'met.rh': 'rh_qaq',
                            'met.temp': 'temp_qaq',
                            'flag': 'flag_qaq'})

    return ds_qaq


def format_quantaq(in_path):
    """
    Reads QuantAQ data from a directory of files.

    :param in_path:
    :type in_path: str

    :return:
    """

    ds_meta = index_meta(in_path)
    ds_qaq = index_quantaq(in_path)

    ds_qaq = xr.merge([ds_qaq, ds_meta])

    ds_qaq = ds_qaq.swap_dims({'sensor': 'site'}).drop('sensor')

    return ds_qaq
