import numpy as np
import xarray as xr

# TODO: should be able to adjust this to be able to work on arbitrary time series chunks
# which then lets use use dask and eventually have a full dask workflow, maybe even from
# the raw data. 

def heatwaves_counts_single_threshold(dataset_year, reference, days_threshold=3):
    """
    Accepts data as a (time, lat, lon) shaped boolean array.
    Iterates through the array in the time dimension comparing the current
    time slice to the previous one. For each cell, determines whether the
    cell is True (i.e. is over the heatwave thresholds) and whether this is
    the start, continuation, or end of a sequence of heatwave conditions.
    Accumulates the number of occurances and counts the total occurances.
    """
    dataset_year = dataset_year.fillna(-9999)
    dataset_year_asbool = (dataset_year > reference).values
    
    # Init arrays, pre allocate to (hopefully) improve performance.
    out_shape: tuple = dataset_year_asbool.shape[1:]

    last_slice: bool[:,:] = dataset_year_asbool[0, :, :]
    curr_slice: bool[:,:] = dataset_year_asbool[0, :, :]    
    hw_ends: bool[:,:] = np.zeros(out_shape, dtype=bool)
    mask: bool[:,:] = np.zeros(out_shape, dtype=bool)
    
    # Init as int32 - value will never be > 365
    accumulator = np.zeros(dataset_year_asbool.shape[1:], dtype=np.int32)
    counter = np.zeros(dataset_year_asbool.shape[1:], dtype=np.int32)
    
    for i in range(1, dataset_year_asbool.shape[0]):
        last_slice = dataset_year_asbool[i-1, :, :]
        curr_slice = dataset_year_asbool[i, :, :]

        # Add to the sequence length counter at all positions
        # above threshold at prev time step using boolean indexing
        accumulator[last_slice] += 1
        
        # End of sequence is where prev is true and current is false
        # True where prev and not current
        # Use pre-allicocated arrays for results
        np.logical_and(last_slice, np.logical_not(curr_slice), out=hw_ends)
        np.logical_and(hw_ends, (accumulator > days_threshold), out=mask)

        # Add 1 where the sequences are ending and are > 3
        counter[mask] += 1
        # Reset the accumulator where current slice is empty
        accumulator[np.logical_not(curr_slice)] = 0
    
    # Finally, 'close' the heatwaves that are ongoing at the end of the year
    # End of sequence is where last value of iteration is true and accumulator is over given length
    np.logical_and(curr_slice, (accumulator > days_threshold), out=mask)
    
    # Add the length of the accumulator where the sequences are ending and are > 3
    counter[mask] += 1

    # Convert np array to xr DataArray
    counter = xr.DataArray(counter, 
                            coords=[dataset_year.latitude.values,
                                    dataset_year.longitude.values,
                                   ], 
                           dims=['latitude', 'longitude'],
                           name='heatwave_count'
                          )
    
    return counter


def heatwaves_days_single_threshold(dataset_year, reference, days_threshold: int=3):
    """
    Accepts data as a (time, lat, lon) shaped boolean array.
    Iterates through the array in the time dimension comparing the current
    time slice to the previous one. For each cell, determines whether the
    cell is True (i.e. is over the heatwave thresholds) and whether this is
    the start, continuation, or end of a sequence of heatwave conditions.
    Accumulates the number of days and counts the total lengths.
    """
    dataset_year = dataset_year.fillna(-9999)
    dataset_year_asbool = (dataset_year > reference).values
    
    # pre allocate arrays
    out_shape: tuple = dataset_year_asbool.shape[1:]

    last_slice: bool[:,:] = dataset_year_asbool[0, :, :]
    curr_slice: bool[:,:] = dataset_year_asbool[0, :, :]    
    hw_ends: bool[:,:] = np.zeros(out_shape, dtype=bool)
    mask: bool[:,:] = np.zeros(out_shape, dtype=bool)
    
    # Init as int32 - value will never be > 365
    accumulator = np.zeros(out_shape, dtype=np.int32)
    days = np.zeros(out_shape, dtype=np.int32)
    
    
    for i in range(1, dataset_year_asbool.shape[0]):
        last_slice = dataset_year_asbool[i-1, :, :]
        curr_slice = dataset_year_asbool[i, :, :]

        # Add to the sequence length counter at all positions
        # above threshold at prev time step using boolean indexing
        accumulator[last_slice] += 1
        
        # End of sequence is where prev is true and current is false
        # True where prev and not current
        # Use pre-allicocated arrays for results
        np.logical_and(last_slice, np.logical_not(curr_slice), out=hw_ends)
        np.logical_and(hw_ends, (accumulator > days_threshold), out=mask)

        # Add the length of the accumulator where the sequences are ending and are > 3
        days[mask] += accumulator[mask]
        # Reset the accumulator where current slice is empty
        accumulator[np.logical_not(curr_slice)] = 0
    
    # Finally, 'close' the heatwaves that are ongoing at the end of the year
    # End of sequence is where last value of iteration is true and accumulator is over given length
    np.logical_and(curr_slice, (accumulator > days_threshold), out=mask)
    
    # Add the length of the accumulator where the sequences are ending and are > 3
    days[mask] += accumulator[mask]
    
    # Convert np array to xr DataArray
    days = xr.DataArray(days, 
                            coords=[dataset_year.latitude.values,
                                    dataset_year.longitude.values,
                                   ], 
                           dims=['latitude', 'longitude'],
                           name='heatwave_length'
                          )
    
    return days


def heatwaves_counts_multi_threshold(datasets_year, thresholds, days_threshold=2):
    """
    Accepts data as a (time, lat, lon) shaped boolean array.
    Iterates through the array in the time dimension comparing the current
    time slice to the previous one. For each cell, determines whether the
    cell is True (i.e. is over the heatwave thresholds) and whether this is
    the start, continuation, or end of a sequence of heatwave conditions.
    Accumulates the number of occurances and counts the total occurances.
    """
    datasets_year = [d.fillna(-9999) for d in datasets_year]
    # Init whole array to True
    threshold_exceeded = datasets_year[0] > thresholds[0]
    # For each threshold array, 'and' them together
    for dataset_year, thresh in zip(datasets_year[1:], thresholds[1:]):
        # for each (data, threshold) pair, add constraint the threshold excedance array
        threshold_exceeded = np.logical_and(threshold_exceeded,
                                            dataset_year > thresh)
    # Keep only the numpy array
    threshold_exceeded = threshold_exceeded.values

    # Init arrays, pre allocate to (hopefully) improve performance.
    out_shape: tuple = threshold_exceeded.shape[1:]

    last_slice: bool[:, :] = threshold_exceeded[0, :, :]
    curr_slice: bool[:, :] = threshold_exceeded[0, :, :]
    hw_ends: bool[:, :] = np.zeros(out_shape, dtype=bool)
    mask: bool[:, :] = np.zeros(out_shape, dtype=bool)

    # Init as int32 - value will never be > 365
    accumulator = np.zeros(threshold_exceeded.shape[1:], dtype=np.int32)
    counter = np.zeros(threshold_exceeded.shape[1:], dtype=np.int32)

    # Calculate the run length of the exceedances and count only the ones
    # over the length threshold
    for i in range(1, threshold_exceeded.shape[0]):
        last_slice = threshold_exceeded[i - 1, :, :]
        curr_slice = threshold_exceeded[i, :, :]

        # Add to the sequence length counter at all positions
        # above threshold at prev time step using boolean indexing
        accumulator[last_slice] += 1

        # End of sequence is where prev is true and current is false
        # True where prev and not current
        # Use pre-allicocated arrays for results
        np.logical_and(last_slice, np.logical_not(curr_slice), out=hw_ends)
        np.logical_and(hw_ends, (accumulator > days_threshold), out=mask)

        # Add 1 where the sequences are ending and are > 3
        counter[mask] += 1
        # Reset the accumulator where current slice is empty
        accumulator[np.logical_not(curr_slice)] = 0

    # Finally, 'close' the heatwaves that are ongoing at the end of the year
    # End of sequence is where last value of iteration is true and accumulator is over given length
    np.logical_and(curr_slice, (accumulator > days_threshold), out=mask)

    # Add the length of the accumulator where the sequences are ending and are > 3
    counter[mask] += 1

    # Convert np array to xr DataArray
    counter = xr.DataArray(counter,
                           coords=[datasets_year[0].latitude.values,
                                   datasets_year[0].longitude.values,
                                   ],
                           dims=['latitude', 'longitude'],
                           name='heatwave_count'
                           )

    return counter


def heatwaves_days_multi_threshold(datasets_year, thresholds, days_threshold: int = 2):
    """
    Accepts data as a (time, lat, lon) shaped boolean array.
    Iterates through the array in the time dimension comparing the current
    time slice to the previous one. For each cell, determines whether the
    cell is True (i.e. is over the heatwave thresholds) and whether this is
    the start, continuation, or end of a sequence of heatwave conditions.
    Accumulates the number of days and counts the total lengths.
    """
    datasets_year = [d.fillna(-9999) for d in datasets_year]
    # Init array
    threshold_exceeded = datasets_year[0] > thresholds[0]
    # For each threshold array, 'and' them together
    for _data_year, _thresh in zip(datasets_year[1:], thresholds[1:]):
        # for each (data, threshold) pair, add constraint the threshold excedance array
        threshold_exceeded = np.logical_and(threshold_exceeded,
                                            _data_year > _thresh)

    # Keep only the numpy array
    #threshold_exceeded = threshold_exceeded.values

    # pre allocate arrays
    out_shape: tuple = threshold_exceeded.shape[1:]

    last_slice: bool[:, :] = threshold_exceeded[0, :, :]
    curr_slice: bool[:, :] = threshold_exceeded[0, :, :]
    hw_ends: bool[:, :] = np.zeros(out_shape, dtype=bool)
    mask: bool[:, :] = np.zeros(out_shape, dtype=bool)

    # Init as int32 - value will never be > 365
    accumulator = np.zeros(out_shape, dtype=np.int32)
    days = np.zeros(out_shape, dtype=np.int32)

    for i in range(1, threshold_exceeded.shape[0]):
        last_slice = threshold_exceeded[i - 1, :, :]
        curr_slice = threshold_exceeded[i, :, :]

        # Add to the sequence length counter at all positions
        # above threshold at prev time step using boolean indexing
        accumulator[last_slice] += 1

        # End of sequence is where prev is true and current is false
        # True where prev and not current
        # Use pre-allicocated arrays for results
        np.logical_and(last_slice, np.logical_not(curr_slice), out=hw_ends)
        np.logical_and(hw_ends, (accumulator > days_threshold), out=mask)

        # Add the length of the accumulator where the sequences are ending and are > 3
        days[mask] += accumulator[mask]
        # Reset the accumulator where current slice is empty
        accumulator[np.logical_not(curr_slice)] = 0

    # Finally, 'close' the heatwaves that are ongoing at the end of the year
    # End of sequence is where last value of iteration is true and accumulator is over given length
    np.logical_and(curr_slice, (accumulator > days_threshold), out=mask)

    # Add the length of the accumulator where the sequences are ending and are > 3
    days[mask] += accumulator[mask]

    # Convert np array to xr DataArray
    days = xr.DataArray(days,
                        coords=[datasets_year[0].latitude.values,
                                datasets_year[0].longitude.values,
                                ],
                        dims=['latitude', 'longitude'],
                        name='heatwaves_days'
                        )

    return days


import numpy as np
import xarray as xr

def heatwaves_days_multi_threshold_monthly(datasets_year, thresholds, days_threshold: int = 2):
    """
    Accepts data as a (time, lat, lon) shaped boolean array.
    Iterates through the array in the time dimension comparing the current
    time slice to the previous one. For each cell, determines whether the
    cell is True (i.e., is over the heatwave thresholds) and whether this is
    the start, continuation, or end of a sequence of heatwave conditions.
    Accumulates the number of days and counts the total lengths.
    """
    # Initialize empty arrays to accumulate results
    out_shape: tuple = datasets_year[0].shape[1:]
    days = np.zeros(out_shape, dtype=np.int32)
    
    for data_year in datasets_year:
        # Fill NaN values with a sentinel value (-9999)
        data_year = data_year.fillna(-9999)
        
        # Initialize arrays for this year's calculations
        threshold_exceeded = data_year > thresholds[0]
        accumulator = np.zeros(out_shape, dtype=np.int32)
        last_slice = threshold_exceeded[0, :, :]
        curr_slice = threshold_exceeded[0, :, :]
        hw_ends = np.zeros(out_shape, dtype=bool)
        mask = np.zeros(out_shape, dtype=bool)
        
        for i in range(1, data_year.shape[0]):
            last_slice = threshold_exceeded[i - 1, :, :]
            curr_slice = threshold_exceeded[i, :, :]

            # Add to the sequence length counter at all positions
            # above threshold at the previous time step using boolean indexing
            accumulator[last_slice] += 1

            # End of sequence is where prev is true and current is false
            np.logical_and(last_slice, np.logical_not(curr_slice), out=hw_ends)
            np.logical_and(hw_ends, (accumulator > days_threshold), out=mask)

            # Add the length of the accumulator where the sequences are ending and are > 3
            days[mask] += accumulator[mask]
            # Reset the accumulator where the current slice is empty
            accumulator[np.logical_not(curr_slice)] = 0
        
        # Finally, 'close' the heatwaves that are ongoing at the end of the year
        np.logical_and(curr_slice, (accumulator > days_threshold), out=mask)

        # Add the length of the accumulator where the sequences are ending and are > 3
        days[mask] += accumulator[mask]

    # Convert the numpy array to a DataArray, keeping the 'time' dimension
    time_dim = datasets_year[0].time.values
    days_da = xr.DataArray(
        days,
        coords=[datasets_year[0].latitude.values, datasets_year[0].longitude.values, time_dim],
        dims=['latitude', 'longitude', 'time'],
        name='heatwaves_days'
    )

    return days_da