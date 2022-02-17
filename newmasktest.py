#from __future__ import print_function
import cv2 as cv
from mask2 import mask_C1, mask_C3
from matplotlib import pyplot as plt
#import argparse
#parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              #OpenCV. You can process both videos and images.')
#parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
#parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')

#args = parser.parse_args()
#if args.algo == 'MOG2':
#backSub = cv.createBackgroundSubtractorMOG2(detectShadows= True)
#else:
# 
backSub = cv.createBackgroundSubtractorKNN()

date = '2021-10-24_12A'
capture = cv.VideoCapture('Videos/lowres_C1_'+ date+'.mp4')
if not capture.isOpened():
    print('Unable to open: ' )#+ args.input)
    exit(0)
while True:
    ret, frame = capture.read()
    if frame is None:
        break
    
    #frame  = cv.bitwise_and(frame, frame, mask=mask_C1[:, :, 0])
    fgMask = backSub.apply(frame)
    newm = mask_C1[:, :, 0] & fgMask
    fgMask = cv.bitwise_and(frame, frame, mask=newm)
    
    #cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    #cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               #cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    
    
    cv.imshow('Frame', frame)
    cv.imshow('mask',fgMask)
   

    
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break