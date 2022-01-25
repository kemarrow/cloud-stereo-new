"""
makes video by combining all jpg files within a folder

@author: Erin
"""


import cv2
import numpy as np
import glob
 
img_array = []
for filename in glob.glob('Depth_frames/*.jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)


out = cv2.VideoWriter('depth_10_24_12A.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 20, (640,480))
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()