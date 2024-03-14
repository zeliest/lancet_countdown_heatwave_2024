import numpy as np
from collections import namedtuple
# import xarray as xr

# _Row = namedtuple('_Row', ['ncdf_name', 'common_name'])

# # Manually extracted from the weather files.
# # python compat var long name, netcdf file name, common name
# WEATHER_NAMES = [
#     _Row('t2m', 'temperature_2m'),
#     _Row('tp', 'total_precipitation'), # actually total_precipitation
#     _Row('d2m', 'temperature_dewpoint'),
#     _Row('sp', 'surface_pressure'),
#     _Row('2T_GDS4_SFC', 'temperature_2m'),
#     _Row('g4_lat_1', 'latitude'),
#     _Row('g4_lon_2', 'longitude'),
#     _Row('initial_time0_hours', 'time'),
#     _Row('lat', 'latitude'),
#     _Row('lon', 'longitude'),
# ]

# def standardise_names(dataset):
#     return dataset.rename({r.ncdf_name: r.common_name for r in WEATHER_NAMES if r.ncdf_name in data.variables})


def area_of_pixel(pixel_size, center_lat):
    """Calculate m^2 area of a wgs84 square pixel.

    Adapted from: https://gis.stackexchange.com/a/127327/2397

    Parameters:
        pixel_size (float): length of side of pixel in degrees.
        center_lat (float): latitude of the center of the pixel. Note this
            value +/- half the `pixel-size` must not exceed 90/-90 degrees
            latitude or an invalid area will be calculated.

    Returns:
        Area of square pixel of side length `pixel_size` centered at
        `center_lat` in m^2.

    """
    a = 6378137  # meters
    b = 6356752.3142  # meters
    e = np.sqrt(1 - (b/a)**2)
    area_list = []
    for f in [center_lat + pixel_size/2, center_lat - pixel_size/2]:
        zm = 1 - e*np.sin(np.radians(f))
        zp = 1 + e*np.sin(np.radians(f))
        area = (np.pi * b**2 * (
                np.log(zp/zm) / (2*e) +
                np.sin(np.radians(f)) / (zp*zm)))
        
        area_list.append(area)
    return (pixel_size / 360.) * (area_list[0] - area_list[1])