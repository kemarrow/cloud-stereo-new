"""
Plots heat map of speed and updraft

@author: Erin
"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import pickle
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
ax6.set_ylabel('Height (m)')
ax6.set_xlabel('Speed (m/s)')

spd = y[idx]
alt = x[idx]
spd[spd>20] = None
idx3 = (~np.isnan(alt+spd))

print(alt[idx3])

numbins = 200
    
df = pd.DataFrame({'altitude': alt[idx3], 'speed':spd[idx3]})
print(df.head)
bins = pd.cut(df['altitude'], numbins)
    
altitude = df.groupby(bins)['speed'].median()
count  = df.groupby(bins)['altitude'].count().rename('Count')
df2 = pd.concat([altitude, count], axis=1)
    
df2.loc[df2.Count < 400*27, 'speed']  = None
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

ax6.plot(df3.iloc[:,0], alt_bins, color='k', label='Median')
ax6.legend()
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

# plt.show()

fig5, ax5 = plt.subplots(1,1)
n, bins, patches = ax5.hist(updraft*5, 1000, facecolor='g', alpha=0.75)


from scipy.stats import norm
from scipy.stats import normpdf 
import matplotlib.mlab as mlab

idx = (~np.isnan(updraft))

# best fit of data
(mu, sigma) = norm.fit(updraft[idx]*5)

# the histogram of the data
#n, bins, patches = plt.hist(datos, 60, normed=1, facecolor='green', alpha=0.75)

# add a 'best fit' line
y = mlab.normpdf( bins, mu, sigma)
sx5.plot(bins, y, 'r--', linewidth=2)


ax5.set_xlabel('Change in height (m)')
ax5.set_ylabel('Number of pixels')
#plt.xlim(-20, 20)
plt.show()

#ax6.set_title('histogram2d')
#ax6.grid()


