# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 16:10:43 2021

@author: kathe
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

from lidarPlots import LidarData, findCloud

lidarH = np.array([[ 1.36071800e+01, -9.17182893e-01, -2.00941892e+03],
       [ 7.76186504e-01, -1.45996552e+01,  5.07045411e+02],
       [ 1.67058286e-03,  7.20348355e-04,  1.00000000e+00]])

def singleTransform(coord, mat):
    coord = list(coord)
    coord = np.append(coord, [1])
    val = np.matmul(mat, coord)
    val[0] = val[0]/val[2]
    val[1] = val[1]/val[2]
    val = np.array(val)
    return val   

filepath = "C:/Users/kathe/OneDrive - Imperial College London/MSci Project/lidar/"
filename = filepath + 'User5_18_20211022_111000.hpl'

# want to take a camera image and plot the lidar onto it then use the findcloud thing to spot the clouds

ld = LidarData.fromfile(filename)
data_locs = ld.data_locs
decimal_time = data_locs[:,0]
azi = data_locs[:,1]
elev = data_locs[:,2]
distance = ld.getDistance()
backscatter = ld.getBackscatter()

prefix_right = 'tl'
prefix_left = 'tl4'
vidfolder = "C:/Users/kathe/OneDrive - Imperial College London/MSci Project/Videos/"
dtime = '2021-10-22'
hour = 11
vidcapR = cv.VideoCapture(f'{vidfolder}/{prefix_right}_{dtime}_{hour:0>2}A.mp4')
# Check if camera opened successfully
if vidcapR.isOpened()== False:
    print("Error opening right camera")

success_coords = []
camera_time = hour + 10/3600    # video does not start precisely on the hour
for frame_no in range(0, 150):
    camera_time += 5/3600
    vidcapR.set(cv.CAP_PROP_POS_FRAMES,frame_no) # Where frame_no is the frame you want
    successR, imgRLarge = vidcapR.read() # Read the frame
    if successR == True:
        img = cv.resize(imgRLarge,(640,480),fx=0,fy=0, interpolation = cv.INTER_CUBIC)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        for time in decimal_time:
                if np.abs(time-camera_time) < 2.5/3600:
                    i = list(decimal_time).index(time)
                    cloudindex, cloud = findCloud(backscatter[i], 500)
                    if cloud is not None:
                        print(decimal_time)
                        fig, (ax, ax2) = plt.subplots(2,1)
                        ax.imshow(img)
                        ax2.set_title('Frame {:0}'.format(frame_no))
                        ax2.plot(ld.getDistance()[i], ld.getBackscatter()[i])
                        ax2.plot(cloud, ld.getBackscatter()[i][cloudindex], 'x')
                        ax2.set_xlabel('distance (m)')
                        ax2.set_ylabel('backscatter')
                        ax2.set_title('Lidar in frame {:0}'.format(frame_no))
                        point = np.array((azi[i], elev[i]))
                        new = singleTransform(point, lidarH)
                        ax.plot(new[0], new[1], 'rx')
                        plt.show()
                    # print(new)
                    # if findCloud(backscatter[i], 500) == (None, None):
                    #     ax.plot(new[0], new[1], 'rx')
                    # else:
                    #     ax.plot(new[0], new[1], 'gx')
    #                 if cloud is not None:    
    #                     success_coords.append(np.array([point, cloud, time]))
vidcapR.release()

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

