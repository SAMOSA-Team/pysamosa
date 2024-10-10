import numpy as np
import xarray as xr


def qa_bam(ds):
    """
    Perform quality assurance check on MetOne BAM data.

    The 'PM25_flag' variable in the returned DataSet is set as follows:
        - It's 1 where PM2.5 data are not available (NaN in original data).
        - It's 2 where PM2.5 levels are below 3 or above 1000, interpreting as outside the limit of detection.
        - It's 3 where 'Status' is greater than 0, indicating that the original measurement has some issues.
        - It's 0 otherwise, which means the data is valid.


    :param ds : The input DataSet containing BAM data - should contain PM2.5 and Status data with metadata.
    :type ds: xarray.Dataset
    :returns ds: The output DataSet after performing the quality check, with computed 'PM25_flag' and
                             dropped 'Status' variable.
    :rtype: xarray.Dataset
    """
    ds['pm25_flag'] = xr.where(np.isnan(ds.pm25), 1,  # no data
                               xr.where(((ds.pm25 < 3) | (ds.pm25 > 1000)), 2,  # LOD
                                        xr.where(ds.status > 0, 3, 0)))
    ds.pm25_flag.encoding['_FillValue'] = 1
    ds = ds.drop_vars('status')
    ds.attrs['history'] = 'flagged'
    return ds


def qa_reg(ds):
    """
    Perform quality assurance checks on regulatory air quality data, including PM2.5, PM10, CO, NO2, SO2, O3.

    Each flag is set according to the following rules (applied respectively):
        - The flag is 1 where the data for the respective measure are not available (NaN in original data).
        - The flag is 2 where the levels are below or above the recognised limit of detection. For 'PM25' and 'PM10',
            the limits are 3 and 2000 respectively. For 'CO', the limits are 10 and 4000. For 'NO2', 'SO2' and 'O3',
            the limits are 2 and 100, 2 and 200 respectively.
        - For 'PM25' and 'PM10', the flag is 3 where the 'PM25' value is greater than 'PM10'
            and both their flags are zero.
        - The flag is 0 otherwise.

    :param ds : The input DataSet containing regulatory air quality data.
    :type ds: xarray.Dataset
    :returns ds: Returns the input DataSet with added computed quality check flags for each measure.
    :rtype: xarray.Dataset
    """
    ds['pm25_flag'] = xr.where(np.isnan(ds.pm25), 1,  # no data
                               xr.where(((ds.pm25 < 3) | (ds.pm25 > 1000)), 2,
                                        xr.where((ds.pm25 > ds.pm10), 3,
                                                 0)))

    ds['pm10_flag'] = xr.where(np.isnan(ds.pm10), 1,  # no data
                               xr.where(((ds.pm10 < 3) | (ds.pm10 > 2000)), 2,
                                        xr.where((ds.pm25 > ds.pm10), 3,
                                                 0)))

    ds['co_flag'] = xr.where(np.isnan(ds.co), 1,  # no data
                             xr.where(((ds.co < 10) | (ds.co > 4000)), 2, 0))

    ds['no2_flag'] = xr.where(np.isnan(ds.no2), 1,  # no data
                              xr.where(((ds.no2 < 2) | (ds.no2 > 100)), 2, 0))

    ds['so2_flag'] = xr.where(np.isnan(ds.so2), 1,  # no data
                              xr.where(((ds.so2 < 2) | (ds.so2 > 200)), 2, 0))

    ds['o3_flag'] = xr.where(np.isnan(ds.o3), 1,  # no data
                             xr.where(((ds.o3 < 2) | (ds.o3 > 200)), 2, 0))

    ds.pm25_flag.encoding['_FillValue'] = 1
    ds.pm10_flag.encoding['_FillValue'] = 1
    ds.co_flag.encoding['_FillValue'] = 1
    ds.no2_flag.encoding['_FillValue'] = 1
    ds.so2_flag.encoding['_FillValue'] = 1
    ds.o3_flag.encoding['_FillValue'] = 1

    ds.attrs['history'] = 'flagged'
    return ds


def qa_pa(ds, dt=40):
    """
    Performs quality assurance checks on data derived from PA monitors.

    The set flags in the returned DataSet follow these rules:
        - 'A_flag' and 'B_flag' are set to 1 where the respective (A or B) measure data is not available.
        - 'A_flag' and 'B_flag' are set to 2 where measure A or B respectively, is less than 5 or greater than 500
          (interpreting as outside the limit of detection).
        - 'A_flag' and 'B_flag' are set to 3 where the absolute quota of ('A' - 'B')/mean('A', 'B') is greater than dt.
        - 'A_flag' and 'B_flag' are set to 0 otherwise (valid data).
        - 'RH_flag' is set to 1 where RH data isn't available.
        - 'RH_flag' is set to 2 where RH is less than 0 or greater than 100 (outside the limit of detection).
        - 'RH_flag' is set to 0 otherwise (valid data).

    :param ds: The input DataSet containing PA monitors.
    :type ds: xarray.Dataset
    :param dt: The maximum allowable difference between A and B - should be [0 - 1].
    :type dt: float
    :returns ds: The output DataSet after performing the quality check, with computed flags.
    :rtype: xarray.Dataset
    """

    ds['a_flag'] = xr.where(np.isnan(ds.a), 1,
                            xr.where(((ds.a < 5) | (ds.a > 500)), 2,
                                     xr.where(ds.disagreement > dt, 3, 0)))
    ds['b_flag'] = xr.where(np.isnan(ds.b), 1,
                            xr.where(((ds.b < 5) | (ds.b > 500)), 2,
                                     xr.where(ds.disagreement > dt, 3, 0)))
    ds['rh_flag'] = xr.where(np.isnan(ds.rh), 1,
                             xr.where(((ds.rh < 0) | (ds.rh > 100)), 2, 0))

    ds.a_flag.encoding['_FillValue'] = 1
    ds.b_flag.encoding['_FillValue'] = 1
    ds.rh_flag.encoding['_FillValue'] = 1

    ds.attrs['history'] = 'flagged'
    return ds


def qa_ghsl(ds):
    """
    Performs quality assurance check on population density data.

    The 'population_flag' variable in the output DataSet is set as follows:
        - It's 1 where population density data (`pop_density`) is not available (NaN in original data).
        - It's 2 where population density is zero.
        - It's 0 otherwise, which means the data is valid and stored in a population density variable.

    :param ds: The input DataSet containing GHSL population data.
    :type ds: xarray.Dataset
    :returns ds: The output DataSet after performing the quality check, with computed flags.
    :rtype: xarray.Dataset.
    """
    ds['population_flag'] = xr.where(np.isnan(ds.population), 1,
                                      xr.where(ds.population == 0, 2, 0))
    ds.population.encoding['_FillValue'] = 1
    ds.attrs['history'] = 'flagged'
    return ds


def qa_tropomi(ds):
    """
    Perform quality assurance checks on TROPOMI NO2 column data.

    The 'no2_column_flag' variable in the returned dataset follows these rules:
        - It's 1 where NO2 column data (`no2_column`) is not available (NaN in original data).
        - It's 2 where NO2 column value is less than or equal to zero.
        - It's 0 otherwise, signifying valid data.

   :param ds: The input DataSet containing TROPOMI NO2 column data.
   :type ds: xarray.Dataset
   :returns ds: The output DataSet after performing the quality check, with computed flags.
   :rtype: xarray.Dataset.
    """
    ds['no2_column_flag'] = xr.where(np.isnan(ds.no2_column), 1,
                                     xr.where(ds.no2_column <= 0, 2, 0))
    ds['no2_column_flag'].encoding['_FillValue'] = 1
    ds.attrs['history'] = 'flagged'
    return ds


def qa_quantaq(ds):
    """
    Perform quality assurance checks on QuantAQ PM data.

    The 'no2_column_flag' variable in the returned dataset follows these rules:
        - It's 1 where QuantAQ is not available (NaN in original data).
        - It's 2 where QuantAQ has flagged the data with a bit-wise encoded flag > 0.
        - It's 0 otherwise, signifying valid data.

    :param ds:
    :type ds: xarray.Dataset
    :return ds:
    :rtype ds: xarray.Dataset
    """

    ds['quantaq_flag'] = xr.where(ds.flag_qaq > 0, 2, 0)
    ds = ds.drop('flag_qaq')
    ds['quantaq_flag'].encoding['_FillValue'] = 1
    ds.attrs['history'] = 'flagged'
    return ds