import numpy as np
import xarray as xr

# TODO: should be able to adjust this to be able to work on arbitrary time series chunks
# which then lets use use dask and eventually have a full dask workflow, maybe even from
# the raw data. 



def heatwaves_counts_multi_threshold(datasets_year, thresholds, days_threshold=2):
    """
    Counts the number of heatwave occurrences above multiple thresholds.

    Parameters:
    - datasets_year (list of xarray.DataArray): A list of 3D arrays (time, lat, lon) 
      containing the data for a single year. Each array corresponds to a different 
      variable or condition that needs to be compared against a threshold.
    - thresholds (list of float or xarray.DataArray): A list of threshold values or arrays 
      against which the data is compared to identify heatwaves.
    - days_threshold (int, optional): The minimum number of consecutive days required 
      for an event to be considered a heatwave. Default is 2.

    Returns:
    - counter (xarray.DataArray): A 2D array of shape (lat, lon) where each value 
      represents the count of heatwave occurrences that meet the days_threshold.
    """
    datasets_year = [d.fillna(-9999) for d in datasets_year]
    threshold_exceeded = datasets_year[0] > thresholds[0]
    
    for dataset_year, thresh in zip(datasets_year[1:], thresholds[1:]):
        threshold_exceeded = np.logical_and(threshold_exceeded,
                                            dataset_year > thresh)
    threshold_exceeded = threshold_exceeded.values

    out_shape: tuple = threshold_exceeded.shape[1:]

    last_slice: bool[:, :] = threshold_exceeded[0, :, :]
    curr_slice: bool[:, :] = threshold_exceeded[0, :, :]
    hw_ends: bool[:, :] = np.zeros(out_shape, dtype=bool)
    mask: bool[:, :] = np.zeros(out_shape, dtype=bool)

    accumulator = np.zeros(threshold_exceeded.shape[1:], dtype=np.int32)
    counter = np.zeros(threshold_exceeded.shape[1:], dtype=np.int32)

    for i in range(1, threshold_exceeded.shape[0]):
        last_slice = threshold_exceeded[i - 1, :, :]
        curr_slice = threshold_exceeded[i, :, :]

        accumulator[last_slice] += 1
        np.logical_and(last_slice, np.logical_not(curr_slice), out=hw_ends)
        np.logical_and(hw_ends, (accumulator > days_threshold), out=mask)

        counter[mask] += 1
        accumulator[np.logical_not(curr_slice)] = 0

    np.logical_and(curr_slice, (accumulator > days_threshold), out=mask)
    counter[mask] += 1

    counter = xr.DataArray(counter,
                           coords=[datasets_year[0].latitude.values,
                                   datasets_year[0].longitude.values],
                           dims=['latitude', 'longitude'],
                           name='heatwave_count')

    return counter


def heatwaves_days_multi_threshold(datasets_year, thresholds, days_threshold: int = 2):
    """
    Counts the total number of days under heatwave conditions above multiple thresholds.

    Parameters:
    - datasets_year (list of xarray.DataArray): A list of 3D arrays (time, lat, lon) 
      containing the data for a single year. Each array corresponds to a different 
      variable or condition that needs to be compared against a threshold.
    - thresholds (list of float or xarray.DataArray): A list of threshold values or arrays 
      against which the data is compared to identify heatwaves.
    - days_threshold (int, optional): The minimum number of consecutive days required 
      for an event to be considered a heatwave. Default is 2.

    Returns:
    - days (xarray.DataArray): A 2D array of shape (lat, lon) where each value 
      represents the total number of days under heatwave conditions.
    """
    datasets_year = [d.fillna(-9999) for d in datasets_year]
    threshold_exceeded = datasets_year[0] > thresholds[0]
    
    for _data_year, _thresh in zip(datasets_year[1:], thresholds[1:]):
        threshold_exceeded = np.logical_and(threshold_exceeded,
                                            _data_year > _thresh)

    out_shape: tuple = threshold_exceeded.shape[1:]

    last_slice: bool[:, :] = threshold_exceeded[0, :, :]
    curr_slice: bool[:, :] = threshold_exceeded[0, :, :]
    hw_ends: bool[:, :] = np.zeros(out_shape, dtype=bool)
    mask: bool[:, :] = np.zeros(out_shape, dtype=bool)

    accumulator = np.zeros(out_shape, dtype=np.int32)
    days = np.zeros(out_shape, dtype=np.int32)

    for i in range(1, threshold_exceeded.shape[0]):
        last_slice = threshold_exceeded[i - 1, :, :]
        curr_slice = threshold_exceeded[i, :, :]

        accumulator[last_slice] += 1
        np.logical_and(last_slice, np.logical_not(curr_slice), out=hw_ends)
        np.logical_and(hw_ends, (accumulator > days_threshold), out=mask)

        days[mask] += accumulator[mask]
        accumulator[np.logical_not(curr_slice)] = 0

    np.logical_and(curr_slice, (accumulator > days_threshold), out=mask)
    days[mask] += accumulator[mask]

    days = xr.DataArray(days,
                        coords=[datasets_year[0].latitude.values,
                                datasets_year[0].longitude.values],
                        dims=['latitude', 'longitude'],
                        name='heatwaves_days')

    return days
