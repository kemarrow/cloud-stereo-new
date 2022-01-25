import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

height = np.loadtxt('cloud_height2.csv', delimiter =  ',')
#speed = np.loadtxt('cloud_speed2.csv', delimiter =  ',')
updraft  = np.loadtxt('cloud_updraft3.csv', delimiter =  ',')

x = height.flatten()
#y = speed.flatten()
updraft_flat = updraft.flatten()
xh =  x
print(np.shape(updraft_flat))
print(np.shape(xh))

#idx = (~np.isnan(x+y))
#fig6, ax6 = plt.subplots(1,1)
#H, xedges, yedges = np.histogram2d(x[idx], y[idx], bins=(100, 100))# range=[[1000, 9000], [0,200]] )
#extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
#spd = ax6.imshow(H.T, extent=extent, interpolation='nearest', origin='lower', cmap='coolwarm', aspect='auto')
#divider = make_axes_locatable(ax6)
#cax6 = divider.append_axes('right', size='5%', pad=0.05)
#fig6.colorbar(spd, cax=cax6, orientation='vertical')
#ax6.set_xlabel('Height (m)')
#ax6.set_ylabel('Speed (m/s)')


idxh = (~np.isnan(xh+updraft_flat))
fig7, ax7 = plt.subplots(1,1)
Hh, xedgesh, yedgesh = np.histogram2d(xh[idxh], updraft_flat[idxh], bins=(100, 100))
extenth = [xedgesh[0], xedgesh[-1], yedgesh[0], yedgesh[-1]]
upd = ax7.imshow(Hh.T, extent=extenth, interpolation='nearest', origin='lower', cmap='coolwarm', aspect='auto')
dividerh = make_axes_locatable(ax7)
cax7 = dividerh.append_axes('right', size='5%', pad=0.05)
fig7.colorbar(upd, cax=cax7, orientation='vertical')

ax7.set_xlabel('Height (m)')
ax7.set_ylabel('Updraft (m/s)')


#ax6.set_title('histogram2d')
#ax6.grid()
plt.show()

