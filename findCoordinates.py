# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 14:10:39 2021

@author: kathe
"""
import matplotlib.pyplot as plt
import cv2 as cv

fpR = r"C:\Users\kathe\OneDrive - Imperial College London\MSci Project\output_C1_20211003_12\C1_031021_frame_146.jpg"
imgRLarge = cv.imread(fpR) # camera 1 image
imgR = cv.resize(imgRLarge,(640,480),fx=0,fy=0, interpolation = cv.INTER_CUBIC)

fpL = r"C:\Users\kathe\OneDrive - Imperial College London\MSci Project\output_C3_20211003_12\C3_031021_frame_146.jpg"
# imgLLarge = cv.imread(fpL) # camera 3 image
# imgL = cv.resize(imgLLarge,(640,480),fx=0,fy=0, interpolation = cv.INTER_CUBIC)
imgL = cv.imread(fpL)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(imgL)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.imshow(imgR)


# Simple mouse click function to store coordinates
def onclick_lidar(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata

    coords_lidar.append((ix, iy))

    # Disconnect after 30 clicks
    if len(coords_lidar) == 10:
        fig2.canvas.mpl_disconnect(cid_lidar)
        plt.close()
    return

def onclick_im(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata

    coords_im.append((ix, iy))
    
      # Disconnect after 50 clicks
    if len(coords_im) == 10:
        fig.canvas.mpl_disconnect(cid_im)
        plt.close()
    return

coords_im = []
coords_lidar = []
cid_im = fig.canvas.mpl_connect('button_press_event', onclick_im)
cid_lidar = fig2.canvas.mpl_connect('button_press_event', onclick_lidar)