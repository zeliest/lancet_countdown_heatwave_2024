from pathlib import Path as _Path

DATA_SRC = _Path('/nfs/n2o/wcr/szelie/').expanduser()
WEATHER_SRC =  _Path('/nfs/n2o/wcr/szelie/era5').expanduser()
POP_DATA_SRC = DATA_SRC / 'lancet/population' 
