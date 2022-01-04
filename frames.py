#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 15:48:52 2021

@author: apple
"""
#%%
import cv2
import numpy as np 

#test

#this code changes the resolution of a video to 640x480 (which is resolution of camera 3)
#load original video
cap = cv2.VideoCapture(r'Videos/C1_2021-10-17_12A.mp4')

#sets the codec for the new video we wil create 
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(r'Videos/lowres_C1_2021-10-17_12A.mp4',fourcc, 20, (640,480))

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


#%%
camera1 = cv2.VideoCapture(r'Videos/lowres_C1_2021-09-27_15A.mp4')
#camera1 = out
# Check if camera opened successfully
if (camera1.isOpened()== False):
    print("Error opening video stream or file")

# Read until video is completed
while(camera1.isOpened()):
    # Capture frame-by-frame
    success, frame = camera1.read()
    if success == True:
        # Display the resulting frame
        cv2.imshow('Frame',frame)
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
            cv2.destroyAllWindows()  
        # Break the loop
    else:
        print('nope')
        break
    
camera1 = cv2.VideoCapture('lowres_C1_2021-10-03_12A.mp4')
success,frame = camera1.read()
count = 0
while success:
    #cv2.imwrite(r"Videos/lowres_C1_frame_%d.jpg" % count, frame)     # save frame as JPEG file      
    success,frame = camera1.read()
    count += 1


# When everything done, release the video capture object
camera1.release()

# Closes all the frames
cv2.destroyAllWindows()
print('done')