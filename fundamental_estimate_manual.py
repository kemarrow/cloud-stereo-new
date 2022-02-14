# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 20:31:29 2022

@author: kathe
"""

import numpy as np
import cv2 
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
import tqdm
import pyproj
from haversine import haversine
from scipy.spatial.transform import Rotation as R
import pickle


ptsR = np.array([(637.4916317991632, 277.8765690376569),
 (637.4916317991632, 292.60460251046027),
 (634.8138075313807, 306.663179916318),
 (593.3075313807531, 218.29497907949792),
 (527.7008368200836, 265.826359832636),
 (593.9769874476988, 50.26150627615061),
 (413.8933054393305, 85.07322175732219),
 (218.4121338912134, 287.2489539748954),
 (214.39539748953973, 247.75104602510464),
 (251.8849372384937, 35.53347280334731),
 (309.4581589958159, 87.75104602510464),
 (444.0188284518828, 401.72594142259413),
 (243.18200836820083, 394.3619246861925),
 (381.7594142259414, 421.14016736401675),
 (267.95188284518827, 434.52928870292885),
 (176.90585774058576, 440.55439330543936),
 (165.52510460251045, 436.5376569037657),
 (26.278242677824267, 405.0732217573222),
 (613.3912133891213, 36.20292887029291),
 (375.73430962343093, 214.94769874476987),
 (398.49581589958154, 363.56694560669456),
 (561.1736401673639, 433.85983263598325),
 (470.7970711297071, 433.19037656903765),
 (385.77615062761504, 300.6380753138075),
 (531.7175732217573, 214.94769874476987),
 (484.1861924686192, 16.788702928870293),
 (628.7887029288703, 310.6799163179916),
 (186.27824267782427, 441.22384937238496),
 (105.94351464435147, 441.22384937238496),
 (131.3828451882845, 450.59623430962347)])
                
                
ptsL = np.array([(632.8054393305439, 267.8347280334728),
 (631.4665271966527, 281.89330543933056),
 (628.1192468619247, 295.95188284518827),
 (589.9602510460251, 208.2531380753138),
 (525.0230125523012, 254.4456066945607),
 (597.9937238493724, 39.550209205020906),
 (417.91004184100416, 72.35355648535568),
 (220.42050209205019, 272.52092050209205),
 (215.73430962343096, 231.68410041841003),
 (254.56276150627613, 18.127615062761492),
 (311.4665271966527, 73.02301255230128),
 (624.7719665271966, 439.2154811715481),
 (415.23221757322176, 427.1652719665272),
 (565.1903765690377, 457.29079497907946),
 (454.06066945606693, 470.6799163179916),
 (213.05648535564853, 429.17364016736406),
 (202.34518828451883, 426.4958158995816),
 (184.26987447698744, 429.17364016736406),
 (617.407949790795, 25.4916317991632),
 (377.07322175732213, 202.22803347280336),
 (401.173640167364, 350.1778242677824),
 (555.8179916317991, 418.46234309623435),
 (467.44979079497904, 417.1234309623431),
 (393.14016736401675, 289.2573221757322),
 (541.7594142259414, 206.91422594142261),
 (509.62552301255226, 8.755230125522985),
 (631.4665271966527, 300.6380753138075),
 (190.96443514644352, 421.80962343096235),
 (111.96861924686192, 419.80125523012555),
 (136.73849372384936, 429.84309623430966)])

CamM_left = np.array([[5.520688775958645920e+02,0.000000000000000000e+00,3.225866125962970159e+02],
          [0.000000000000000000e+00,5.502640890663026312e+02,2.362389385357402034e+02],
          [0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00]])

distort = np.array([2.808374038768443048e-01,-9.909134707088265159e-01,6.299531255281858727e-04,-1.301770463801651002e-03,1.093982545460403522e+00])

translation_vector = np.array([ 61.09920255, -16.,   0.])

w,h = 640,480
    
starsL = np.array([(632.8054393305439, 267.8347280334728),
 (631.4665271966527, 281.89330543933056),
 (628.7887029288703, 295.95188284518827),
 (589.9602510460251, 208.2531380753138),
 (524.3535564853556, 254.4456066945607),
 (597.9937238493724, 40.219665271966505),
 (618.7468619246862, 25.4916317991632),
 (454.73012552301253, 137.96025104602512),
 (417.91004184100416, 71.68410041841008),
 (391.8012552301255, 262.47907949790795),
 (389.1234309623431, 189.50836820083686),
 (377.07322175732213, 201.55857740585776),
 (330.2112970711297, 214.94769874476987),
 (312.1359832635983, 73.02301255230128),
 (254.56276150627613, 18.127615062761492),
 (219.75104602510459, 272.52092050209205),
 (215.73430962343096, 232.35355648535568),
 (330.2112970711297, 214.94769874476987)])

starsR = np.array([(638.1610878661088, 278.5460251046025),
 (637.4916317991632, 292.60460251046027),
 (634.8138075313807, 306.663179916318),
 (592.6380753138075, 217.62552301255232),
 (527.7008368200836, 265.826359832636),
 (593.9769874476988, 50.26150627615061),
 (614.0606694560669, 36.87238493723851),
 (452.72175732217573, 149.34100418410043),
 (414.56276150627616, 85.07322175732219),
 (393.14016736401675, 275.1987447698745),
 (388.4539748953975, 202.89748953974896),
 (375.73430962343093, 214.27824267782427),
 (330.2112970711297, 227.66736401673643),
 (308.7887029288703, 87.75104602510464),
 (251.8849372384937, 35.53347280334731),
 (218.4121338912134, 287.2489539748954),
 (214.39539748953973, 247.081589958159),
 (329.5418410041841, 226.99790794979083)])

#rotation,___ = cv2.findHomography(starsR, starsL)
starsL_3d = np.append(starsL, np.ones((len(starsL), 1)), -1)
starsR_3d = np.append(starsR, np.ones((len(starsR), 1)), -1)
rotation, ___ = R.align_vectors(starsR_3d, starsL_3d)
rotation = rotation.as_matrix()

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
    CamM_left, distort, (w, h), 1, (w, h))

fundamental_matrix, inlier_mask = cv2.findFundamentalMat(ptsL, ptsR, cv2.FM_RANSAC, 3)
essential_matrix, inlier_mask = cv2.findEssentialMat(ptsL, ptsR, cameraMatrix=newcameramtx)

relDirs = np.rad2deg(cv2.Rodrigues(cv2.decomposeEssentialMat(essential_matrix)[0])[0])

pose = cv2.recoverPose(essential_matrix, ptsL, ptsR, newcameramtx)
relDirs2 = np.rad2deg(cv2.Rodrigues(pose[1])[0])

inter_cam_dist = np.sqrt(np.sum(translation_vector**2))
T = pose[2]*inter_cam_dist

Rleft, Rright, Pleft, Pright, Q, roi_left, roi_right = cv2.stereoRectify(
    newcameramtx, distort,
    newcameramtx, distort,
    imageSize=(w,h),
    R=pose[1],
    T=pose[2]*inter_cam_dist,
    flags= cv2.CALIB_ZERO_DISPARITY)

success, Hleft, Hright = cv2.stereoRectifyUncalibrated(
    np.float32(ptsL), np.float32(ptsR),
    fundamental_matrix, imgSize=(w, h)
    )


stereo_cal = {'F': fundamental_matrix,
              'E': essential_matrix,
              'R': pose[1],
              'T': T,
              'Rleft': Rleft,
              'Rright': Rright,
              'Pleft': Pleft,
              'Pright': Pright,
              'Q': Q,
              'Hleft': Hleft,
              'Hright': Hright,
              'roi_left': roi_left,
              'roi_right': roi_right}

with open('stereo_cal_mat.pkl', 'wb') as f:
    pickle.dump(stereo_cal, f)
    
    
    
#%%

folder = "C:/Users/kathe/OneDrive - Imperial College London/MSci Project/longExposures/"
imgRLarge = cv2.imread(folder+'tl_2021-10-30_230002_CAL1.jpg')
imgLLarge = cv2.imread(folder+'tl4_2021-10-30_230002_CAL1.jpg')
imgR = cv2.resize(imgRLarge,(640,480),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
imgL = cv2.resize(imgLLarge,(640,480),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, sharex=True, sharey=True)
ax1.imshow(imgL)
ax2.imshow(imgR)
ax1.plot(starsL[:,0], starsL[:,1], 'r.', label='C3 star positions')
ax2.plot(starsR[:,0], starsR[:,1], 'g.', label='C1 star positions')


starsL_rect = cv2.undistortPoints(starsL, newcameramtx, distort, 
                                  Rleft, np.delete(Pleft, 3, axis=1))
starsR_rect = cv2.undistortPoints(starsR, newcameramtx, distort, 
                                  Rright, np.delete(Pright, 3, axis=1))

starsL_rect = starsL_rect.reshape((18,2))
starsR_rect = starsR_rect.reshape((18,2))

ax3.plot(starsL[:,0], starsL[:,1], 'r.', label='C3 star positions')
ax3.plot(starsR[:,0], starsR[:,1], 'g.', label='C1 star positions')
ax3.legend()


ax4.plot(starsL[:,0], starsL[:,1], 'rx', label='C3 star positions')
ax4.plot(starsR[:,0], starsR[:,1], 'gx', label='C1 star positions')
ax4.plot(starsL_rect[:,0], starsL_rect[:,1], 'r.', label='C3 rectified star positions')
ax4.plot(starsR_rect[:,0], starsR_rect[:,1], 'g.', label='C1 rectified star positions')
ax4.legend()
