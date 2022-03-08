import numpy as np
import cv2 as cv
from mask2 import mask_C1, mask_C3
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.image import NonUniformImage
import pickle

date = '2021-10-24_12A'
with open('data/heightpix_'+date+'2.pkl','rb') as f:
    height = pickle.load(f)

with open('data/framecount_'+date+'2.pkl','rb') as f:
    frame = pickle.load(f)



print(height)
print(frame)

print(np.shape(height))
for i in range(0, 40):
    x  = np.linspace(0, len(height[i]),len(height[i]))
    plt.plot(x, height[i])

plt.xlabel('Frame Number')
plt.ylabel('Height (m)')

plt.show()


