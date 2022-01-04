# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 12:23:49 2021

@author: kathe
"""
import numpy as np
import optuna
import cv2

from lidarPlots import LidarData
from mask2 import mask_C1, mask_C3
#%% camera matrices
CamM_left = np.array([[5.520688775958645920e+02,0.000000000000000000e+00,3.225866125962970159e+02],
          [0.000000000000000000e+00,5.502640890663026312e+02,2.362389385357402034e+02],
          [0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00]])

CamM_right = np.array([[5.520688775958645920e+02,0.000000000000000000e+00,3.225866125962970159e+02],
          [0.000000000000000000e+00,5.502640890663026312e+02,2.362389385357402034e+02],
          [0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00]])


Distort_left = np.array([2.808374038768443048e-01,-9.909134707088265159e-01,6.299531255281858727e-04,-1.301770463801651002e-03,1.093982545460403522e+00])
Distort_right = np.array([2.808374038768443048e-01,-9.909134707088265159e-01,6.299531255281858727e-04,-1.301770463801651002e-03,1.093982545460403522e+00])

w, h = 640, 480

fundamental_matrix = np.loadtxt(r'matrices/fundamental_matrix.csv', delimiter = ',')
essential_matrix = np.loadtxt(r'matrices/essential_matrix.csv', delimiter = ',')
pose = np.loadtxt(r'matrices/pose[1].csv', delimiter = ',')
T = np.loadtxt(r'matrices/T.csv', delimiter = ',')
Rleft = np.loadtxt(r'matrices/Rleft.csv', delimiter = ',')
Rright = np.loadtxt(r'matrices/Rright.csv', delimiter = ',')
Pleft = np.loadtxt(r'matrices/Pleft.csv', delimiter = ',')
Pright = np.loadtxt(r'matrices/Pright.csv', delimiter = ',')
Q = np.loadtxt(r'matrices/Q.csv', delimiter = ',')
Hleft = np.loadtxt(r'matrices/Hleft.csv', delimiter = ',')
Hright = np.loadtxt(r'matrices/Hright.csv', delimiter = ',')
roi_left = np.loadtxt(r'matrices/roi_left.csv', delimiter = ',')
roi_right = np.loadtxt(r'matrices/roi_right.csv', delimiter = ',')

cameraH = np.array([[ 9.42118957e-01,  4.18822851e-01, -7.25736750e+01],
       [-3.63139244e-01,  9.57252119e-01,  1.40435108e+02],
       [ 8.82977443e-05,  1.30033207e-04,  1.00000000e+00]])

lidarH = np.array([[ 1.36071800e+01, -9.17182893e-01, -2.00941892e+03],
       [ 7.76186504e-01, -1.45996552e+01,  5.07045411e+02],
       [ 1.67058286e-03,  7.20348355e-04,  1.00000000e+00]])
#%% functions
def flattenList(t):
    flat_list = []
    for sublist in t:
        for item in sublist:
            flat_list.append(item)
    return flat_list

def findCloud(backscatter, dist_thresh=300):
    # np.argmax returns the index of where the backscatter is highest
    # index in this case = range gate i.e. distance
    if np.max(backscatter) > 10:
        # print('building reflection')
        return (None, None)
    cloud = np.argmax(backscatter[dist_thresh:])
    if backscatter[cloud+dist_thresh] - np.median(backscatter[dist_thresh:]) > 0.03:
    # if backscatter[cloud+dist_thresh] - backscatter[cloud+dist_thresh+10] > 0.05: 
        # print((cloud+dist_thresh+20)*3)
        return cloud+dist_thresh, (cloud+dist_thresh+20)*3 #cloud index, cloud distance
    else:
        # print('no clouds')
        return (None, None)
    
def lidarToRectifiedSingle(lidar, H_L, H_C):
    z = np.append(lidar, [1])
    val = np.matmul(H_L, z)
    val[0] = val[0]/val[2]
    val[1] = val[1]/val[2]
    newval = np.array([val[0], val[1], 1])
    rectval = np.matmul(H_C, newval)
    rectval[0] = rectval[0]/rectval[2]
    rectval[1] = rectval[1]/rectval[2]
    return rectval
#%%

# read lidar data
decimal_time = []
azi = []
elev = []
distance = []
backscatter = []
for dtime in [1,2,3,4,5]:
        filename = 'lidar/20211004/User5_18_20211004_11{0}000.hpl'.format(dtime)
        ld = LidarData.fromfile(filename)
        data_locs = ld.data_locs
        decimal_time.append(list(data_locs[:,0]))
        azi.append(list(data_locs[:,1]))
        elev.append(list(data_locs[:,2]))
        distance.append(list(ld.getDistance()))
        backscatter.append(list(ld.getBackscatter()))

decimal_time = np.array(flattenList(decimal_time), ndmin=1)
azi = np.array(flattenList(azi), ndmin=1)
elev = np.array(flattenList(elev), ndmin=1)
distance = np.array(flattenList(distance), ndmin=1)
backscatter = np.array(flattenList(backscatter), ndmin=1)
#%%

vidcapR = cv2.VideoCapture('Videos/lowres_C1_2021-10-04_11A.mp4')
vidcapL = cv2.VideoCapture('Videos/C3_2021-10-04_11A.mp4')
frame = 0
while(vidcapR.isOpened() and vidcapL.isOpened()):
    success, imgR = vidcapR.read()
    success2, imgL = vidcapL.read()
    if success==True and success2==True:
        frame += 1
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            CamM_left, Distort_left, (w, h), 1, (w, h))
        mapx, mapy = cv2.initUndistortRectifyMap(
            CamM_left, Distort_left, None, newcameramtx, (w, h),  cv2.CV_32FC1)

        imgR = cv2.bitwise_and(imgR, imgR, mask=mask_C1)
        imgL = cv2.bitwise_and(imgL, imgL, mask=mask_C3)

        #assume both images have same height and width

        new_camera_matrixleft, roi = cv2.getOptimalNewCameraMatrix(CamM_left,Distort_left,(w,h),1,(w,h))
        new_camera_matrixright, roi = cv2.getOptimalNewCameraMatrix(CamM_right,Distort_right,(w,h),1,(w,h))

        #Undistort images
        imgR_undistorted = cv2.undistort(imgR, CamM_right, Distort_right, None, new_camera_matrixright)
        imgL_undistorted = cv2.undistort(imgL, CamM_left, Distort_left, None, new_camera_matrixleft)

        #creates new map for each camera with the rotation and pose (R and P) values
        mapLx, mapLy = cv2.initUndistortRectifyMap(new_camera_matrixleft, Distort_left, Rleft, Pleft, (w,h), cv2.CV_32FC1)
        mapRx, mapRy = cv2.initUndistortRectifyMap(new_camera_matrixright, Distort_right, Rright, Pright, (w,h), cv2.CV_32FC1)

        # remaps each image to the new map
        rimgR = cv2.remap(imgR, mapRx, mapRy,
                              interpolation=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0, 0))
        rimgL = cv2.remap(imgL, mapLx, mapLy,
                              interpolation=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0, 0))
        cv2.imwrite(f'rectframes/rimgR_{frame}.jpg', rimgR)
        cv2.imwrite(f'rectframes/rimgL_{frame}.jpg', rimgL)
    else: 
        break
vidcapL.release()
vidcapR.release()
# Closes all the frames
cv2.destroyAllWindows()
#%%

def objective(trial):
    #min_disp = trial.suggest_int('min_disp', 0, 2)
    max_disp = trial.suggest_int('max_disp', 4, 8)
    #win_size = trial.suggest_int('win_size', 5, 11)
    block_size = trial.suggest_int('block_size', 3, 11)
    uniqueness_ratio = trial.suggest_int('uniqueness_ratio', 5, 30)
    speckle_window_size = trial.suggest_int('speckle_window_size', 50, 200)
    speckle_range = trial.suggest_int('speckle_range', 1, 3)
    disp12_max_diff = trial.suggest_int('disp12_max_diff', 5, 10)
    num_disp = 16*max_disp
    
    stereo = cv2.StereoSGBM_create(minDisparity= 0,
        numDisparities = num_disp,
        blockSize = block_size,
        uniquenessRatio = uniqueness_ratio,
        speckleWindowSize =  speckle_window_size,
        speckleRange = speckle_range,
        disp12MaxDiff = disp12_max_diff,
        P1= 8*3*block_size**2,
        P2=32*3*block_size**2)
    
    stereo_depths = []
    lidar_depths = []
    camera_time = 11
    for j in range(1, 717):
        rimgL = cv2.imread(f'rectframes/rimgL_{j}.jpg')
        rimgR = cv2.imread(f'rectframes/rimgR_{j}.jpg')
        disparity_map = stereo.compute(rimgL, rimgR).astype(np.float32)
        im3d = cv2.reprojectImageTo3D(disparity_map/32, Q, handleMissingValues = True)
        im3d[im3d == np.inf] = 0
        im3d[im3d > 9_000] = 0
        im3d[im3d < -9_000] = 0
        im3d[im3d == -np.inf] = 0
        depths = np.sqrt(im3d[:,:,0]**2 + im3d[:,:,1]**2 + im3d[:,:,2]**2)
        
        camera_time += 5/3600
        for time in decimal_time:
            if np.abs(time-camera_time) < 0.001:
                i = list(decimal_time).index(time)
                cloudindex, cloud = findCloud(backscatter[i], 500)  
                point = np.array((azi[i], elev[i]))
                new = lidarToRectifiedSingle(point, lidarH, cameraH)
                if cloud is not None:
                    if int(new[0]) < 640 and int(new[1]) < 480:
                        if depths[int(new[1]), int(new[0])] != 0:
                            stereo_depths.append(depths[int(new[1]), int(new[0])])
                            lidar_depths.append(cloud)
                            
    # Handle pruning based on the intermediate value.
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return np.corrcoef(lidar_depths, stereo_depths)[0][1]

study = optuna.create_study(pruner=optuna.pruners.MedianPruner(),
                                        direction='maximize')
study.optimize(objective, n_trials=50)