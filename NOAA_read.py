import netCDF4
import gzip
from matplotlib import pyplot as plt
 

with gzip.open('20211024_1200.gz') as gz:
    with netCDF4.Dataset('temporary', mode='r', memory=gz.read()) as nc:
        airport = nc.variables['profileAirport']
        profileTime = nc.variables['profileTime']
        altitude = nc.variables['altitude']
        windDir = nc.variables['windDir']
        windSpeed = nc.variables['windSpeed']
        print(nc.variables.keys())

        print(windSpeed[:])
        print(airport[:])
        plt.plot(windSpeed[:],altitude[:])
        plt.show()
        #plt.show()
