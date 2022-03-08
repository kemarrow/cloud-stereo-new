"""
makes video by combining all jpg files within a folder

@author: Erin
"""
import cv2
import numpy as np
import glob
import os
import imageio
 
'''
img_array = []
for filename in glob.glob('Depth_frames/*.jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

out = cv2.VideoWriter('heights.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 20, (640,480))
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

'''

png_dir = 'gif_speeds'
images = []
for file_name in sorted(os.listdir(png_dir)):
    if file_name.endswith('.png'):
        print(file_name)
        file_path = os.path.join(png_dir, file_name)
        images.append(imageio.imread(file_path))

#test

imageio.mimsave('movie3.gif', images)