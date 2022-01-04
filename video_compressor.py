import cv2
import numpy as np 

#test

#this code changes the resolution of a video to 640x480 (which is resolution of camera 3)
#load original video

def compress_vid(video):
    cap = cv2.VideoCapture(video)

#sets the codec for the new video we wil create 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(r'Videos/lowres_C1_2021-10-09_12A.mp4',fourcc, 20, (640,480))

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