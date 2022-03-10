import numpy as np
import cv2 as cv
from mask2 import mask_C1, mask_C3
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.image import NonUniformImage
import pickle

date = '2021-10-24_12A'
#with open('data/heightpix_'+date+'new.pkl','rb') as f:
    #height = pickle.load(f)

height = np.load('data/heightpix_'+date+'10new.npy')
print(height.shape)
print(np.isnan(height).sum())

#height = height[(height == np.nan).any(axis  = -1)]
#print((height == np.nan).any(axis  = -1).shape)

print(height.shape)
height = height.reshape(-1, 10)
height = height[::500]
print(np.shape(height.T))
#h = height.T[:][::10]
#f = height.T[:, 1][::10]
plt.plot(height.T, alpha = 0.4)

plt.xlabel('Frame Number')
plt.ylabel('Height (m)')
plt.show()







#with open('data/framecount_'+date+'new.pkl','rb') as f:
    #frame = pickle.load(f)




#print(np.shape(height))
#for i in range(0, 40):
    #x  = np.linspace(0, len(height[i]),len(height[i]))
    #plt.plot(x, height[i])



#plt.show()


