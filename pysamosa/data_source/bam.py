"""
MetOne BAM-1022 Format and Export
Last Updated: June 19, 2024
This script formats MetOne BAM-1022 files (.csv files).
@author: markjcampmier
"""
# Import Packages
import os
import threading
import numpy as np
import pandas as pd
import xarray as xr


# Define Functions

def check_bam(bam_path):
    """
    Checks BAM files for 'Data Report' header and returns a list of valid files.

    :param bam_path: Path to the BAM data directory.
    :type bam_path: str
    :returns: A list of valid BAM file paths.
    :rtype: list
    """
    csv_files, bam_files = [], []
    for root, _, files in os.walk(bam_path):
        for filename in files:
            if (filename.endswith(".csv")) | (filename.endswith(".CSV")):
                filepath = os.path.join(root, filename)
                csv_files.append(filepath)
    for csv_file in csv_files:
        with open(csv_file, 'rb') as csv:
            header = csv.read(4096)  # Read 4096 bytes (approximately 5 lines)
            header_lines = header.decode("utf-8").splitlines()[:5]
            if 'Data Report' in header_lines:
                bam_files.append(csv_file)
    return bam_files


def read_bam(files):
    """
    Reads BAM files into a Pandas DataFrame.

    :param files: A list of BAM file paths or a single BAM file path.
    :type files: list
    :return: A Pandas DataFrame containing all BAM file paths.
    """
    # skip rows are set based on header content,
    # parse_dates to read as pd.Timestamp, set column 0 as datetimeindex

    bam_name = os.path.basename(files)

    files = check_bam(files)

    df_bam = pd.concat([pd.read_csv(file, skiprows=4, parse_dates=True,
                                    index_col=0, usecols=['Time', 'ConcHR(ug/m3)', 'Status']) for file in files])
    df_bam = df_bam.resample('1H').aggregate({'ConcHR(ug/m3)': 'mean', 'Status': 'sum'})
    df_bam.index = df_bam.index - pd.Timedelta('1H')
    df_bam.loc[:, 'bam_name'] = bam_name
    df_bam = df_bam.reset_index()
    df_bam = df_bam.set_index(['Time', 'bam_name'])
    return df_bam, bam_name


def read_meta(in_path):
    """
    Takes the data path, reads the metadata, and converts to xarray.DataArray.

    :param in_path: Path to the BAM data directory.
    :type in_path: str
    :returns: A xarray.DataArray containing the metadata.
    :rtype: xarray.DataArray
    """
    df_meta = pd.read_csv(os.path.join(in_path, 'bam_meta.csv')).set_index('sensor')
    ds_meta = df_meta.to_xarray()
    return ds_meta


def index_bam(bam_path, ds_list):
    """
    Exports a QA'd BAM DataFrame to a Feather file.

    :param bam_path: Path to bam files
    :type bam_path: str
    :param ds_list: Container for parallel processing
    :type ds_list: list
    :return: None
    """

    df_bam, bam_name = read_bam(bam_path)

    ds_bam = df_bam.to_xarray()
    ds_list.append(ds_bam)
    return None


def format_bam(in_path):
    """
    Formats BAM files into a nc file containing munged PM2.5 data.

    :param in_path: A list of BAM file paths or a single BAM file path.
    :type in_path: list or str
    :return: None
    """
    threads, ds_list = [], []
    bam_paths = [os.path.join(in_path, f) for f in os.listdir(in_path) if os.path.isdir(os.path.join(in_path, f))]

    ds_meta = read_meta(in_path)

    for bam_path in bam_paths:
        thread = threading.Thread(
            target=index_bam, args=(bam_path, ds_list)
        )
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish processing
    for thread in threads:
        thread.join()

    ds_bams = xr.merge(ds_list)

    ds_bams = ds_bams.rename({'bam_name': 'sensor',
                              'Time': 'time'})
    ds_bams = xr.merge([ds_bams, ds_meta])
    ds_bams = ds_bams.to_dataframe().reset_index().groupby(['site', 'time']).agg({'ConcHR(ug/m3)': 'mean',
                                                                                  'Status': 'sum'}).to_xarray()
    ds_bams = xr.merge([ds_bams, ds_meta.to_dataframe().reset_index(drop=True).set_index('site').drop_duplicates(
    ).to_xarray().set_coords(['latitude', 'longitude'])])

    ds_bams = ds_bams.rename_vars({'ConcHR(ug/m3)': 'pm25',
                                   'Status': 'status'})
    ds_bams = ds_bams.sortby('time')
    for var in ds_bams.data_vars:
        ds_bams[var].encoding = {"_FillValue": np.nan}

    ds_bams['pm25'] = ds_bams['pm25'].assign_attrs(long_name='PM2.5',
                                                   units="ug/m^3",
                                                   description="BAM-1022 PM2.5 data")
    ds_bams['status'] = ds_bams['status'].assign_attrs(long_name='Status',
                                                       description="BAM reported bit-masked status flags.")

    ds_bams['latitude'].attrs['units'] = 'degrees_north'
    ds_bams['latitude'].attrs['long_name'] = 'Latitude'

    ds_bams['longitude'].attrs['units'] = 'degrees_east'
    ds_bams['longitude'].attrs['long_name'] = 'Longitude'

    ds_bams.attrs['history'] = 'munged'

    return ds_bams
