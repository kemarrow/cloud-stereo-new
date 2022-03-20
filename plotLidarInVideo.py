# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 12:10:57 2021

@author: kathe
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

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

def flattenList(t):
    flat_list = []
    for sublist in t:
        for item in sublist:
            flat_list.append(item)
    return flat_list

#import camera video
vidfolder = "C:/Users/kathe/OneDrive - Imperial College London/MSci Project/Videos/"
dtime = '2021-10-22'
hour = 11
prefix_right = 'tl'
camera1 = cv2.VideoCapture(f'{vidfolder}/{prefix_right}_{dtime}_{hour:0>2}A.mp4')

filepath = "C:/Users/kathe/OneDrive - Imperial College London/MSci Project/lidar/"

decimal_time = []
azi = []
elev = []
dist = []

for time in [1,2,3,4,5]:
        filename = filepath + 'User5_18_20211022_11{0}000.hpl'.format(time)
        ld = LidarData.fromfile(filename)
        data_locs = ld.data_locs
        decimal_time.append(list(data_locs[:,0]))
        azi.append(list(data_locs[:,1]))
        elev.append(list(data_locs[:,2]))
        dist.append(list(np.argmax(ld.data[:, 3:600, 2], axis=1)))
        
decimal_time = np.array(flattenList(decimal_time), ndmin=1)
azi = np.array(flattenList(azi), ndmin=1)
elev = np.array(flattenList(elev), ndmin=1)
dist = np.array(flattenList(dist), ndmin=1)

# Check if camera opened successfully
if (camera1.isOpened()== False):
    print("Error opening video stream or file")

#sets the codec for the new video we wil create 
#fourcc = cv2.VideoWriter_fourcc(*'MPEG-4')
#out = cv2.VideoWriter('test.mp4',fourcc, 1, (640,480))
out = cv2.VideoWriter('test4.mp4', cv2.VideoWriter_fourcc('A','V','C','1'), 10, (640,480))

times = []
heights = []
camera_time = 11
while True:
    success, frame = camera1.read()
    if success == True:
        camera_time += 5/3600
        for time in decimal_time:
            if np.abs(time-camera_time) < 0.001:
                i = list(decimal_time).index(time)
                fig, (ax1, ax2) = plt.subplots(1,2)
                fig.tight_layout()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                ax1.imshow(frame)
                point = np.array((azi[i], elev[i]))
                new = singleTransform(point, lidarH)
                if dist[i] > 110:
                    times.append(decimal_time[i])
                    heights.append(dist[i])
                    ax1.plot(new[0], new[1], 'gx')
                else:
                    ax1.plot(new[0], new[1], 'rx')
                ax2.plot(np.array(times), np.array(heights)*3, '.')
                ax2.set_xlabel('decimal time')
                ax2.set_ylabel('lidar height (m)')
                canvas = FigureCanvas(fig)
                canvas.draw()
                mat = np.array(canvas.renderer._renderer)
                mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
                out.write(mat)
                plt.close(fig)
    else:
        break


# camera_time = 11
# success, frame = camera1.read()
# while success == True:
#     success,frame = camera1.read()
#     camera_time += 5/3600
#     
#             cv2.imshow('frame', frame)
    


lidar_time = []
for t in decimal_time:
    hours = int(t)
    minutes = (t*60) % 60
    seconds = (t*3600) % 60

    lidar_time.append("%d:%02d.%02d" % (hours, minutes, seconds))
    
# When everything done, release the video capture object
camera1.release()

# Closes all the frames
cv2.destroyAllWindows()