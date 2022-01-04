# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 14:45:37 2021

plots all the lidar scan for the day

@author: kathe
"""

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


class LidarData():
    def __init__(self,
                 fname=None,
                 system_id=None,
                 num_gates=0,
                 gate_length=0,
                 gate_pulse_length=0,
                 pulses_per_ray=0,
                 start_time=None,
                 data=None,
                 data_locs=None):
        self.fname = fname
        self.system_id = system_id
        self.num_gates = num_gates
        self.gate_length = gate_length
        self.gate_pulse_length = gate_pulse_length
        self.pulses_per_ray = pulses_per_ray
        self.start_time = start_time
        self.data = data
        self.data_locs = data_locs

    @classmethod
    def fromfile(cls, filename):
        with open(filename) as f:
            header = [f.readline().split(':', maxsplit=1) for i in range(17)]

            fname = header[0][1].strip()
            system_id = header[1][1].strip()
            num_gates = int(header[2][1].strip())
            gate_length = header[3][1].strip()
            gate_pulse_length = header[4][1].strip()
            pulses_per_ray = header[5][1].strip()
            start_time = header[9][1].strip()
            data_locs_format = header[13][0].split(' ')[0].strip()
            data_format = header[15][0].split(' ')[0].strip()

            data = []
            data_locs = []
            while True:
                try:
                    data_locs_in = np.array(f.readline().split()).astype('float')
                    if len(data_locs_in) == 0:
                        break
                    data_locs.append(data_locs_in)
                    data.append(np.array(
                        [f.readline().split() for i in range(num_gates)]).astype('float'))
                except:
                    break
            data = np.array(data)
            data_locs = np.array(data_locs)

            return cls(
                 fname=fname,
                 system_id=system_id,
                 num_gates=num_gates,
                 gate_length=gate_length,
                 gate_pulse_length=gate_pulse_length,
                 pulses_per_ray=pulses_per_ray,
                 start_time=start_time,
                 data=data,
                 data_locs=data_locs)

    # starting all these at 20 means we avoid the really large peak at zero distance
    def getDistance(self):
        return self.data[:,20:,0]*3 #multiply by 3 to get distance in m

    def getDoppler(self):
         return self.data[:,20:,1]

    def getBackscatter(self):
         return self.data[:,20:,2]

    def getBeta(self):
         return self.data[:,20:,3]





filepath = "lidar/"
filename = filepath + 'User5_18_20211009_121000.hpl.txt'
# this file is the full calibration scan
# filename = filepath+"User1_18_20210916_095204.hpl"


ld = LidarData.fromfile(filename)

data_locs = ld.data_locs
elev = (ld.data_locs[:, 2]*10).astype('int')/10
elev_steps = len(np.unique(elev))

azi = (ld.data_locs[:, 1]*10).astype('int')/10

backscatter_max = np.max(ld.data[:, 1:600, 2], axis=1).reshape((elev_steps, -1))

dist_array = np.argmax(ld.data[:, 3:600, 2], axis=1).reshape((elev_steps, -1))+3
#dist_array[backscatter_max<2] = 500
#%%
def findCloud(backscatter, dist_thresh=300):
    # np.argmax returns the index of where the backscatter is highest
    # index in this case = range gate i.e. distance
    if np.max(backscatter) > 10:
        print('building reflection')
        return (None, None)
    cloud = np.argmax(backscatter[dist_thresh:])
    if backscatter[cloud+dist_thresh] - np.median(backscatter[dist_thresh:]) > 0.03:
    # if backscatter[cloud+dist_thresh] - backscatter[cloud+dist_thresh+10] > 0.05:
        print((cloud+dist_thresh+20)*3)
        backscatter = backscatter.tolist()
        del backscatter[cloud+dist_thresh-10:cloud+dist_thresh+10]
        cloud2 = np.argmax(backscatter[dist_thresh:])
        if backscatter[cloud2+dist_thresh] - np.median(backscatter[dist_thresh:]) > 0.03:
            print((cloud2+dist_thresh+20)*3)
            del backscatter[cloud2+dist_thresh-10:cloud2+dist_thresh+10]
            cloud3 = np.argmax(backscatter[dist_thresh:])
            if backscatter[cloud3+dist_thresh] - np.median(backscatter[dist_thresh:]) > 0.03:
                return (cloud+dist_thresh, cloud2+dist_thresh, cloud3+dist_thresh), ((cloud+dist_thresh+20)*3, (cloud2+dist_thresh+20)*3, (cloud3+dist_thresh+20)*3) #cloud index, cloud distance
            else:
                return (cloud+dist_thresh, cloud2+dist_thresh), ((cloud+dist_thresh+20)*3, (cloud2+dist_thresh+20)*3) #cloud index, cloud distance
        else:
            return cloud+dist_thresh, (cloud+dist_thresh+20)*3 #cloud index, cloud distance
    else:
        print('no clouds')
        return (None, None)
#%%

### UNCOMMENT TO PLOT THINGS ###

# fig1, ax1 = plt.subplots(1,1)
# backscatter = ax1.imshow(backscatter_max, origin='lower', extent=[azi.min(), azi.max(), elev.min(), elev.max()])
# divider1 = make_axes_locatable(ax1)
# cax1 = divider1.append_axes('right', size='5%', pad=0.05)
# fig1.colorbar(backscatter, cax=cax1, orientation='vertical')
# ax1.set_title('Max intensity')
# ax1.set_ylabel('Elevation')

# fig2, ax2 = plt.subplots(1,1)
# dist = ax2.imshow(dist_array, origin='lower', extent=[azi.min(), azi.max(), elev.min(), elev.max()])
# divider2 = make_axes_locatable(ax2)
# cax2 = divider2.append_axes('right', size='5%', pad=0.05)
# fig2.colorbar(dist, cax=cax2, orientation='vertical')
# ax2.set_title('20211004_11 distance')
# plt.savefig('20211004_11_lidarPlot.png', bbox_inches='tight')
# ax2.set_ylabel('Elevation')
# ax2.set_xlabel('Azimuth')
# plt.show()

# fig3, ax3 = plt.subplots(1,1)
# clouds_dist = dist_array.flatten()
# # clouds_dist = clouds_dist[clouds_dist>100]
# ax3.hist(clouds_dist*3, bins=40)
# ax3.set_xlabel('cloud heights (m)')
# ax3.set_title('20211004_11 lidar cloud heights')
# #plt.savefig('20211004_11_lidarHist.png', bbox_inches='tight')
# plt.show()

# plt.subplots_adjust(hspace=0.5)
# #plt.savefig('hpt_geometric_calibration_scan.pdf', bbox_inches='tight')

# fig, (ax1,ax2) = plt.subplots(1,2)
# ax1.plot(azi, clouds_dist,'.')
# ax2.plot(elev, clouds_dist,'.')

# plt.plot(ld.getDistance()[0], ld.getBackscatter()[0])
# plt.xlabel('distance (m)')
# plt.ylabel('backscatter')
# plt.show()

# for i in range(50,60):
#     index, cloud = findCloud(ld.getBackscatter()[i])
#     print(cloud)
#     fig, ax = plt.subplots(1,1)
#     ax.plot(ld.getDistance()[i], ld.getBackscatter()[i])
#     if cloud is not None:
#         ax.plot(cloud, ld.getBackscatter()[i][index], 'x')
#     plt.show()
#%%
a = 132
cloudindex, cloud = findCloud(ld.getBackscatter()[a])
print(cloud)
fig, ax = plt.subplots(1,1)
ax.plot(ld.getDistance()[a], ld.getBackscatter()[a])
for i in range(0, len(cloud)):
    ax.plot(cloud[i], ld.getBackscatter()[a][cloudindex[i]], 'x')
ax.set_xlabel('distance (m)')
ax.set_ylabel('backscatter')
plt.show()

# peaks, _ = sig.find_peaks(ld.getBackscatter()[51], height=1, threshold=None, distance=None, prominence=0.01, width=None, wlen=None, rel_height=None, plateau_size=None)

# fig, ax = plt.subplots(1,1)
# ax.plot(ld.getDistance()[51], ld.getBackscatter()[51])
# ax.plot(peaks, ld.getBackscatter()[51][peaks], "x")
# plt.show()
