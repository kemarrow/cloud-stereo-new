# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 14:10:39 2021

@author: kathe
"""
import matplotlib.pyplot as plt
import cv2 as cv

# fpR = r"C:\Users\kathe\OneDrive - Imperial College London\MSci Project\output_C1_20211003_12\C1_031021_frame_209.jpg"
fpR = r"C:\Users\kathe\OneDrive - Imperial College London\MSci Project\longExposures\tl_2021-09-23_230002_CAL1_sharpened.jpg"
imgRLarge = cv.imread(fpR) # camera 1 image
imgR = cv.resize(imgRLarge,(640,480),fx=0,fy=0, interpolation = cv.INTER_CUBIC)
imgR = cv.cvtColor(imgR, cv.COLOR_BGR2RGB)

# fpL = r"C:\Users\kathe\OneDrive - Imperial College London\MSci Project\output_C3_20211003_12\C3_031021_frame_209.jpg"
fpL = r"C:\Users\kathe\OneDrive - Imperial College London\MSci Project\longExposures\tl4_2021-09-23_230002_CAL1.jpg"
imgLLarge = cv.imread(fpL) # camera 3 image
imgL = cv.resize(imgLLarge,(640,480),fx=0,fy=0, interpolation = cv.INTER_CUBIC)
# imgL = cv.imread(fpL)
imgL = cv.cvtColor(imgL, cv.COLOR_BGR2RGB)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(imgL)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.imshow(imgR)

def onclick_L(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata

    coords_L.append((ix, iy))
    
      # Disconnect after 50 clicks
    if len(coords_L) == 10:
        fig.canvas.mpl_disconnect(cid_L)
        plt.close()
    return

# Simple mouse click function to store coordinates
def onclick_R(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata

    coords_R.append((ix, iy))

    # Disconnect after 30 clicks
    if len(coords_R) == 10:
        fig2.canvas.mpl_disconnect(cid_R)
        plt.close()
    return

coords_L = []
coords_R = []
cid_L = fig.canvas.mpl_connect('button_press_event', onclick_L)
cid_R = fig2.canvas.mpl_connect('button_press_event', onclick_R)