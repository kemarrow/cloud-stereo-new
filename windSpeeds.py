# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 14:29:42 2022

@author: kathe
"""

import netCDF4
import gzip
import numpy as np
import pandas as pd
import os

from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')

path = "C:/Users/kathe/Documents/WindSpeeds/"

with gzip.open(path+'20211024_1200.gz') as gz:
    with netCDF4.Dataset('temporary', mode='r', memory=gz.read()) as nc:
        profileAirport = nc.variables['profileAirport'][:]
        profileTime = nc.variables['profileTime']
        altitude = nc.variables['altitude'][:]
        windDir = nc.variables['windDir'][:]
        windSpeed = nc.variables['windSpeed'][:]
        # print(nc.variables.keys())

        times = netCDF4.num2date(profileTime[:], profileTime.units)
        airport = np.ma.getdata(profileAirport)
        
        filename = os.path.join(path, '20211024_1200.csv')
        print(f'Writing data in tabular form to {filename} (this may take some time)...')
        airports_grid, times_grid, altitude_grid, wind_dir_grid, wind_speed_grid = [
                x.flatten() for x in np.meshgrid(airport[0,:],
                times[0], altitude[0], windDir[0,:], windSpeed[0,:], indexing='ij')]
        df = pd.DataFrame({
            'airport': airports_grid,
            'time': [t.isoformat() for t in times_grid],
            'altitude': altitude_grid,
            'wind direction': wind_dir_grid,
            'wind speed': wind_speed_grid})
            # 't2m': t2m[:].flatten()})
        df.to_csv(filename, index=False)
        print('Done')
        
    
        
