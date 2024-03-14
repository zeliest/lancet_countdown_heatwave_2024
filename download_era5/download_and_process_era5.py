from collections import namedtuple
from pathlib import Path

import numpy as np
import xarray as xr
import cdsapi

project_path = os.path.abspath(os.path.join('..'))
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from source.config import DATA_SRC


SUBDAILY_TEMPERATURES_FOLDER = DATA_SRC / 'era5' / 'era5_0.25deg' / 'hourly_temperature_2m'
SUBDAILY_TEMPERATURES_FOLDER.mkdir(exist_ok=True)

TEMPERATURE_SUMMARY_FOLDER = DATA_SRC / 'era5' / 'era5_0.25deg' / 'daily_temperature_summary'
TEMPERATURE_SUMMARY_FOLDER.mkdir(exist_ok=True)

MAX_YEAR = 2023


assert SUBDAILY_TEMPERATURES_FOLDER.is_dir()
assert TEMPERATURE_SUMMARY_FOLDER.is_dir()



def retreive_year(out_file, year):
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type':'reanalysis',
            'variable':'2m_temperature',
            'year': year,
#             'grid':'0.5/0.5', now get the default rez which is 0.25deg
            'month':[
                '01','02','03',
                '04','05','06',
                '07','08','09',
                '10','11','12'
            ],
            'day':[
                '01','02','03',
                '04','05','06',
                '07','08','09',
                '10','11','12',
                '13','14','15',
                '16','17','18',
                '19','20','21',
                '22','23','24',
                '25','26','27',
                '28','29','30',
                '31'
            ],
            'time':[
                '00:00','01:00','02:00',
                '03:00','04:00','05:00',
                '06:00','07:00','08:00',
                '09:00','10:00','11:00',
                '12:00','13:00','14:00',
                '15:00','16:00','17:00',
                '18:00','19:00','20:00',
                '21:00','22:00','23:00'
            ],
            'format':'grib'
        },
        str(out_file))

def generate_daily_summary(source_file):
    # resample and save a single year.
    # Force loading at the start - idea is that once loaded we re-use the data anyway
    # for min/max/mean
    daily = xr.open_dataset(source_file, engine='cfgrib').load()
    daily = daily.resample(time='1D')
    tmin = daily.min()
    tmax = daily.max()
    tmean = daily.mean()

    tmin = tmin.rename({'t2m': 't_min'})
    tmax = tmax.rename({'t2m': 't_max'})
    tmean = tmean.rename({'t2m': 't_mean'})
    daily_summary = xr.merge([tmin, tmax, tmean])
    return daily_summary


def download_and_summarise_year(year, overwrite=False):
    '''Download the GRIB file for hourly temperatures for one year
    Generate a summary file with temperature min, max, mean

    TODO: later, probably can delete the hourly data file after summarising it
    '''
    year = str(year)
    out_file = SUBDAILY_TEMPERATURES_FOLDER / f'{year}_temperature.grib'
    summary_file = TEMPERATURE_SUMMARY_FOLDER / f'{year}_temperature_summary.nc'

    if overwrite is True or out_file.exists() is False:
        retreive_year(out_file, year)
    else:
        print(f'Skip {out_file}, already exists.')

    if overwrite is True or not summary_file.exists():
        daily_summary = generate_daily_summary(out_file)

        # Encode using rounding to nearest 0.01K to save space, looses a small amount of precision
        # but should be ok since it not used for detailed simulations...
        daily_summary.to_netcdf(summary_file,
                            encoding={
                                't_min': {'dtype': 'int16', 'scale_factor': 0.01, '_FillValue': -9999},
                                't_max': {'dtype': 'int16', 'scale_factor': 0.01, '_FillValue': -9999},
                                't_mean': {'dtype': 'int16', 'scale_factor': 0.01, '_FillValue': -9999}
                            })

    else:
        print(f'Skip {summary_file}, already exists.')
        

def generate_daily_summary(source_file):
    # resample and save a single year.
    # Force loading at the start - idea is that once loaded we re-use the data anyway
    # for min/max/mean
    daily = xr.open_dataset(source_file, engine='cfgrib').load()
    daily = daily.resample(time='1D')
    tmin = daily.min()
    tmax = daily.max()
    tmean = daily.mean()

    tmin = tmin.rename({'t2m': 't_min'})
    tmax = tmax.rename({'t2m': 't_max'})
    tmean = tmean.rename({'t2m': 't_mean'})
    daily_summary = xr.merge([tmin, tmax, tmean])
    return daily_summary


def download_and_summarise_year(year, overwrite=False):
    '''Download the GRIB file for hourly temperatures for one year
    Generate a summary file with temperature min, max, mean

    TODO: later, probably can delete the hourly data file after summarising it
    '''
    year = str(year)
    out_file = SUBDAILY_TEMPERATURES_FOLDER / f'{year}_temperature.grib'
    summary_file = TEMPERATURE_SUMMARY_FOLDER / f'{year}_temperature_summary.nc'

    if overwrite is True or out_file.exists() is False:
        retreive_year(out_file, year)
    else:
        print(f'Skip {out_file}, already exists.')

    if overwrite is True or not summary_file.exists():
        daily_summary = generate_daily_summary(out_file)

        # Encode using rounding to nearest 0.01K to save space, looses a small amount of precision
        # but should be ok since it not used for detailed simulations...
        daily_summary.to_netcdf(summary_file,
                            encoding={
                                't_min': {'dtype': 'int16', 'scale_factor': 0.01, '_FillValue': -9999},
                                't_max': {'dtype': 'int16', 'scale_factor': 0.01, '_FillValue': -9999},
                                't_mean': {'dtype': 'int16', 'scale_factor': 0.01, '_FillValue': -9999}
                            })

    else:
        print(f'Skip {summary_file}, already exists.')# # Run each year one at a time
        
        
for year in range(1980, MAX_YEAR+1):
    download_and_summarise_year(year)