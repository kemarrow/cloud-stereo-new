# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 14:10:39 2021

@author: kathe
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from undistortRectify import rimgR

fp = 'frames/lowres_output_C1_20211004_11/'
img = cv.imread(fp+'C1_041021_frame_1.jpg') # camera 1 image


fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(img)

fig2 = plt.figure()
ax2 = fig2.add_subplot(121)
ax2.imshow(rimgR)


# Simple mouse click function to store coordinates
def onclick_lidar(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata

    coords_lidar.append((ix, iy))

    # Disconnect after 30 clicks
    if len(coords_lidar) == 30:
        fig2.canvas.mpl_disconnect(cid_lidar)
        plt.close()
    return

def onclick_im(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata

    coords_im.append((ix, iy))
    
      # Disconnect after 50 clicks
    if len(coords_im) == 30:
        fig.canvas.mpl_disconnect(cid_im)
        plt.close()
    return

coords_im = []
coords_lidar = []
cid_im = fig.canvas.mpl_connect('button_press_event', onclick_im)
cid_lidar = fig2.canvas.mpl_connect('button_press_event', onclick_lidar)