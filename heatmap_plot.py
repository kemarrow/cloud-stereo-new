"""
Plots heat map of speed and updraft

@author: Erin
"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import pickle
#from reading_madis import test_read

import gzip
import typing
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset, chartostring

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'large',
          'figure.figsize': (6, 6),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

date =  '2021-10-24_12A'

with open('cloud_height_'+date+'.pkl','rb') as f:
    height = pickle.load(f)

with open('cloud_speed_'+date+'.pkl','rb') as f:
    speed = pickle.load(f)

with open('cloud_updraft_'+date+'.pkl','rb') as f:
    updraft = pickle.load(f)

print(np.shape(updraft))
print(np.shape(height))
print(np.nanmax(height))


x = height.flatten()
y = speed.flatten()

#y =2*y

#y[y< 5] = None

#x= x[0:640*480*40]
#y= y[0:640*480*40]

updraft_flat = updraft.flatten()
xh =  x

print(np.shape(x))
#print(np.shape(updraft_flat))
#print(np.shape(xh))

idx = (~np.isnan(x+y))

fig6, ax6 = plt.subplots(1, 1)
H, xedges, yedges = np.histogram2d(y[idx], x[idx], bins=(100, 100), range=[[0,20],[500, 4000]])
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
spd = ax6.imshow(H.T, extent=extent, interpolation='nearest', origin='lower', cmap='coolwarm', aspect='auto')
divider = make_axes_locatable(ax6)
cax6 = divider.append_axes('right', size='5%', pad=0.05)
fig6.colorbar(spd, cax=cax6, orientation='vertical')
ax6.set_ylabel('Altitude (m)')
ax6.set_xlabel('Speed (m/s)')

spd = y[idx]
alt = x[idx]
spd[spd>20] = None
idx3 = (~np.isnan(alt+spd))

print(alt[idx3])

numbins = 100
    
df = pd.DataFrame({'altitude': alt[idx3], 'speed':spd[idx3]})
print(df.head)
bins = pd.cut(df['altitude'], numbins)
    
altitude = df.groupby(bins)['speed'].median()
count  = df.groupby(bins)['altitude'].count().rename('Count')
df2 = pd.concat([altitude, count], axis=1)
    
df2.loc[df2.Count < 400*29, 'speed']  = None
print('Im here')
print(df2.head())
df3 = df2
#len(df3.iloc[:,0])
minbin = np.nanmin(alt[idx3])
maxbin= np.nanmax(alt[idx3])
sizeofbin = (maxbin-minbin)/numbins
print(minbin)
print(maxbin)
    #minalt = np.nanmin(df3.iloc[:,0])
    #print(minalt)
minalt = min(alt[idx3])
alt_bins = []
for i in range(0, len(df3.iloc[:,0]), 1):
    p = minalt + ((2*i+1)*sizeofbin/2)
    alt_bins.append(p)



print('length', len(df3.iloc[:,0]))



test_file = Path("20211024_1200.gz")
airports = np.array(["EGLL", "EGSS", "EGKK", "EGMC", "EGGW", "EGMC"])

def test_read(
    filename: Path,
    extract_keys: typing.List[str] = [
        "profileAirport",
        "profileTime",
        "altitude",
        "windDir",
        "windSpeed",
    ],
) -> typing.Dict[str, np.ndarray]:
    output = {}
    with gzip.open(filename) as gz:
        with Dataset("dummy", mode="r", memory=gz.read()) as ncdf:
            print(ncdf.variables.keys())
            for key in extract_keys:
                tmp = ncdf.variables[key][:]
                if tmp.dtype.str.startswith("|S"):
                    tmp = chartostring(tmp)
                output[key] = tmp
    return output


data = test_read(test_file)

airport_idx = np.array([code in airports for code in data["profileAirport"]])
print(airport_idx.sum())
data_airports = {key: value[airport_idx] for key, value in data.items()}

for airport, alt, speed in zip(
    data_airports["profileAirport"],
    data_airports["altitude"],
    data_airports["windSpeed"],
):
    print(airport)
    #color = 'y'
    if airport == 'EGKK':
        label = 'Gatwick'
        color = 'y'
    if airport == 'EGLL':
        label = 'Heathrow'
        color = 'm'
    ax6.plot(speed[3:17], alt[3:17], label=label, color = color, linewidth= 2)
ax6.plot(df3.iloc[:,0][0: 80], alt_bins[0: 80], color='k', label='Median data', linewidth = 2.5)
ax6.legend()
ax6.set_title("Cloud speed")
plt.show()

plt.show()

#_____________________________________________________________-
# fig6, (ax6, ax7) = plt.subplots(1, 2)
# H, xedges, yedges = np.histogram2d(x[idx], y[idx], bins=(100, 100), range=[[500, 6000], [0,20]])
# extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
# spd = ax6.imshow(H.T, extent=extent, interpolation='nearest', origin='lower', cmap='coolwarm', aspect='auto')
# divider = make_axes_locatable(ax6)
# cax6 = divider.append_axes('right', size='5%', pad=0.05)
# fig6.colorbar(spd, cax=cax6, orientation='vertical')
# ax6.set_xlabel('Altitude (m)')
# ax6.set_ylabel('Speed (m/s)')


# idxh = (~np.isnan(xh+updraft_flat))
# #fig7, ax7 = plt.subplots(1,1)
# Hh, xedgesh, yedgesh = np.histogram2d(xh[idxh], updraft_flat[idxh], bins=(100, 100), range=[[500, 6000], [-30,30]])
# extenth = [xedgesh[0], xedgesh[-1], yedgesh[0], yedgesh[-1]]
# upd = ax7.imshow(Hh.T, extent=extenth, interpolation='nearest', origin='lower', cmap='coolwarm', aspect='auto')
# dividerh = make_axes_locatable(ax7)
# cax7 = dividerh.append_axes('right', size='5%', pad=0.05)
# fig6.colorbar(upd, cax=cax7, orientation='vertical')

# ax7.set_xlabel('Height (m)')
# ax7.set_ylabel('Updraft (m/s)')

# spd = y[idx]
# alt = x[idx]
# spd[spd>20] = None
# idx3 = (~np.isnan(alt+spd))

# print(alt[idx3])

# numbins = 100
    
# df = pd.DataFrame({'altitude': alt[idx3], 'speed':spd[idx3]})
# print(df.head)
# bins = pd.cut(df['altitude'], numbins)
    
# altitude = df.groupby(bins)['speed'].median()
# count  = df.groupby(bins)['altitude'].count().rename('Count')
# df2 = pd.concat([altitude, count], axis=1)
    
# df2.loc[df2.Count < 400*10, 'speed']  = None
# print('Im here')
# print(df2.head())
# df3 = df2
# #len(df3.iloc[:,0])
# minbin = np.nanmin(alt[idx3])
# maxbin= np.nanmax(alt[idx3])
# sizeofbin = (maxbin-minbin)/numbins
# print(minbin)
# print(maxbin)
#     #minalt = np.nanmin(df3.iloc[:,0])
#     #print(minalt)
# minalt = min(alt[idx3])
# alt_bins = []
# for i in range(0, len(df3.iloc[:,0]), 1):
#     p = minalt + (2*i+1)*sizeofbin/2
#     alt_bins.append(p)

# ax6.plot(alt_bins,df3.iloc[:,0], color='k', label='Weighted average')


