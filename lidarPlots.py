# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 12:51:17 2022

@author: kathe
"""
import numpy as np
import matplotlib.pyplot as plt


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
     
        
# should just add this to class? 
    
def findCloud(backscatter, dist_thresh=500):
    # np.argmax returns the index of where the backscatter is highest
    # index in this case = range gate i.e. distance
    if np.max(backscatter) > 10:
        # print('building reflection')
        return (None, None)
    cloud = np.argmax(backscatter[dist_thresh:])
    if backscatter[cloud+dist_thresh] - np.median(backscatter[dist_thresh:]) > 0.03: 
        # print((cloud+dist_thresh+20)*3)
        return cloud+dist_thresh, (cloud+dist_thresh+20)*3 #cloud index, cloud distance
    else:
        # print('no clouds')
        return (None, None)
    
# ### Plot things! ###    
    
# filepath = "C:/Users/kathe/OneDrive - Imperial College London/MSci Project/lidar/"
# filename = filepath + 'User5_18_20211022_111000.hpl'
# # this file is the full calibration scan
# # filename = filepath+"User1_18_20210916_095204.hpl"


# ld = LidarData.fromfile(filename)

# data_locs = ld.data_locs
# elev = (ld.data_locs[:, 2]*10).astype('int')/10
# elev_steps = len(np.unique(elev))

# azi = (ld.data_locs[:, 1]*10).astype('int')/10

# backscatter_max = np.max(ld.data[:, 1:600, 2], axis=1).reshape((elev_steps, -1))

# dist_array = np.argmax(ld.data[:, 3:600, 2], axis=1).reshape((elev_steps, -1))+3
# #dist_array[backscatter_max<2] = 500


# for a in range(0,100):
#     cloudindex, cloud = findCloud(ld.getBackscatter()[a])

#     if cloud is not None:
#         print(a)
#         fig, (ax1,ax2,ax3) = plt.subplots(3,1)
#         ax1.plot(ld.getDistance()[a], ld.getBackscatter()[a])
#         ax1.plot(cloud, ld.getBackscatter()[a][cloudindex], 'x')
#         ax1.set_xlabel('distance (m)')
#         ax1.set_ylabel('backscatter')
        
#         ax2.plot(ld.getDistance()[a], ld.getBeta()[a])
#         ax2.set_ylabel('beta')
#         # ax2.set_yscale('log')
        
#         ax3.plot(ld.getDistance()[a], ld.getDoppler()[a])
#         ax3.plot(cloud, ld.getDoppler()[a][cloudindex], 'x')
#         ax3.set_ylabel('Doppler')
#         ax3.set_yscale('log')

#         print(ld.getDoppler()[a][cloudindex])
# plt.show()