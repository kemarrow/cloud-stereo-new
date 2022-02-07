"""
makes video by combining all jpg files within a folder

@author: Erin
"""


import cv2
import numpy as np 

#test

#this code changes the resolution of a video to 640x480 (which is resolution of camera 3)
#load original video
cap = cv2.VideoCapture(r'Videos/C1_2021-10-20_11A.mp4')

#sets the codec for the new video we wil create 
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(r'Videos/lowres_C1_2021-10-20_11A.mp4',fourcc, 20, (640,480))

#resize each frame 
while True:
    ret, frame = cap.read()
    if ret == True:
        b = cv2.resize(frame,(640,480),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        out.write(b)
    else:
        break
    
cap.release()
out.release()
cv2.destroyAllWindows()