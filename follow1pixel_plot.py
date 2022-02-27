import numpy as np
import cv2 as cv
from mask2 import mask_C1, mask_C3
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.image import NonUniformImage
import pickle

height = np.loadtxt('data/height1pix_2021-10-24_12A.csv', delimiter=',')
height1 = np.loadtxt('data/height1pix_2021-10-24_12A-204-268.csv', delimiter=',')
height1[height1>10000]=None

x= np.linspace(1, len(height), len(height))
x1 = np.linspace(1, len(height1), len(height1))
time1 = x1*5
time = x*5



plt.plot(time1, height1)
plt.show()

