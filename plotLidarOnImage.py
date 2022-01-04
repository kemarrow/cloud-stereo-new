# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 16:10:43 2021

@author: kathe
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

from lidarCameraMatching import lidarH, transform
from lidarPlots import LidarData, findCloud

def singleTransform(coord, mat):
    coord = list(coord)
    coord = np.append(coord, [1])
    val = np.matmul(mat, coord)
    val[0] = val[0]/val[2]
    val[1] = val[1]/val[2]
    val = np.array(val)
    return val   

fp = 'frames/lowres_output_C1_20211004_11/'
# img = cv.imread(fp + 'C1_041021_frame_{}.jpg') # camera 1 image

filename = 'lidar/User5_18_20211009_120008.hpl.txt'

# want to take a camera image and plot the lidar onto it then use the findcloud thing to spot the clouds

ld = LidarData.fromfile(filename)
data_locs = ld.data_locs
decimal_time = data_locs[:,0]
azi = data_locs[:,1]
elev = data_locs[:,2]
distance = ld.getDistance()
backscatter = ld.getBackscatter()

success_coords = []
camera_time = 11    
for frame in range(0, 10):
    camera_time += 5/3600
    img = cv.imread(fp+'C1_041021_frame_{:0}.jpg'.format(frame))
    # fig, ax = plt.subplots(1,1)
    # fig2, ax2 = plt.subplots(1,1)
    # ax.imshow(img)
    # ax.set_title('Frame {:0}'.format(frame))
    for time in decimal_time:
            if np.abs(time-camera_time) < 0.001:
                i = list(decimal_time).index(time)
                cloudindex, cloud = findCloud(backscatter[i], 500)
                # ax2.plot(ld.getDistance()[i], ld.getBackscatter()[i])
                # if cloud is not None:
                #     ax2.plot(cloud, ld.getBackscatter()[i][cloudindex], 'x')
                # ax2.set_xlabel('distance (m)')
                # ax2.set_ylabel('backscatter')
                # ax2.set_title('Lidar in frame {:0}'.format(frame))
                # plt.show()
                point = np.array((azi[i], elev[i]))
                new = singleTransform(point, lidarH)
                # print(new)
                # if findCloud(backscatter[i], 500) == (None, None):
                #     ax.plot(new[0], new[1], 'rx')
                # else:
                #     ax.plot(new[0], new[1], 'gx')
#                 if cloud is not None:    
#                     success_coords.append(np.array([point, cloud, time]))

# success_coords = np.array(success_coords)

def decimalToNormal(decimal):
    new_time = []
    for t in decimal:
        hours = int(t)
        minutes = (t*60) % 60
        seconds = (t*3600) % 60
        new_time.append("%d:%02d.%02d" % (hours, minutes, seconds))
    return new_time

# times = decimalToNormal(success_coords[:,2])

