# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 23:00:13 2021

@author: kathe
"""

import numpy as np
import matplotlib.pyplot as plt

lidar = np.loadtxt('lidar_depths6.csv', delimiter=',')
stereo = np.loadtxt('stereo_depths6.csv', delimiter=',')
disparity= np.loadtxt('disparity.csv', delimiter=',')

plt.plot(lidar, stereo, '.')
fit, cov = np.polyfit(lidar, stereo, deg=1, cov=True)
# plt.plot(lidar, fit[0]*lidar + fit[1])
plt.ylabel('stereo depths (m)')
plt.xlabel('lidar depths (m)')
plt.savefig('lidarStereoDepths_withContrast.png', bbox_inches='tight')
plt.show()
