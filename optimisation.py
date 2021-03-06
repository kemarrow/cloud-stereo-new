# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 12:23:49 2021

@author: kathe
"""
import numpy as np
import optuna
import pickle
import cv2
import glob

from lidarPlots import LidarData

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

stereo_cal = pickle.load( open( 'stereo_cal_mat.pkl', "rb" ) )

cameraH = np.array([[ 1.03570118e+00,  2.20209220e-01, -6.56401026e+01],
 [-2.47759214e-01,  9.72997542e-01,  7.11706249e+01],
 [ 1.01353307e-04, -5.38746250e-05,  1.00000000e+00]])

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

year = 2021
month = 10
day = 24
hour = 12
folder = "C:/Users/kathe/OneDrive - Imperial College London/MSci Project/lidar"
for time in [0,1,2,3,4,5]:
        pattern = f'{folder}/User5_18_{year}{month:0>2}{day:0>2}_{hour:0>2}{time}*.hpl'
        filenames = glob.glob(pattern)
        if len(filenames) != 1:
            continue
        else:
            filename = filenames[0]
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
vidcapR = cv2.VideoCapture(r"C:/Users/kathe/OneDrive - Imperial College London/MSci Project/Videos/tl_2021-10-24_12A.mp4")
vidcapL = cv2.VideoCapture(r"C:/Users/kathe/OneDrive - Imperial College London/MSci Project/Videos/tl4_2021-10-24_12A.mp4")
frame = 0
while(vidcapR.isOpened() and vidcapL.isOpened()):
    success, imgRLarge = vidcapR.read()
    success2, imgL = vidcapL.read()
    if success==True and success2==True:
        imgR = cv2.resize(imgRLarge,(640,480),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)

        #assume both images have same height and width

        new_camera_matrixleft, roi = cv2.getOptimalNewCameraMatrix(CamM_left,Distort_left,(w,h),0,(w,h))
        new_camera_matrixright, roi = cv2.getOptimalNewCameraMatrix(CamM_right,Distort_right,(w,h),0,(w,h))

        #Undistort images
        imgR_undistorted = cv2.undistort(imgR, CamM_right, 
                                         Distort_right, new_camera_matrixright)
        imgL_undistorted = cv2.undistort(imgL, CamM_left, 
                                         Distort_left, new_camera_matrixleft)

        #creates new map for each camera with the rotation and pose (R and P) values
        mapLx, mapLy = cv2.initUndistortRectifyMap(new_camera_matrixleft, 
                                                   Distort_left, stereo_cal.get('Rleft'), stereo_cal.get('Pleft'), (w,h), cv2.CV_32FC1)
        mapRx, mapRy = cv2.initUndistortRectifyMap(new_camera_matrixright, 
                                                   Distort_right, stereo_cal.get('Rright'), stereo_cal.get('Pright'), (w,h), cv2.CV_32FC1)
        
        # remaps each image to the new map
        rimgR = cv2.remap(imgR_undistorted, mapRx, mapRy,
                              interpolation=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0, 0))
        rimgL = cv2.remap(imgL_undistorted, mapLx, mapLy,
                              interpolation=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0, 0))
        cv2.imwrite(f'rectframes/rimgR_{frame}.jpg', rimgR)
        cv2.imwrite(f'rectframes/rimgL_{frame}.jpg', rimgL)
    else: 
        break
    frame += 1
vidcapL.release()
vidcapR.release()
# Closes all the frames
cv2.destroyAllWindows()


#%%

def objective(trial):
    #min_disp = trial.suggest_int('min_disp', 0, 2)
    max_disp = trial.suggest_int('max_disp', 4, 8)
    win_size = trial.suggest_int('win_size', 5, 11)
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
        P1= 8*3*win_size**2,
        P2=32*3*win_size**2)
    
    stereo_depths = []
    lidar_depths = []
    camera_time = hour + 10/3600
    for j in range(1, 717):
        rimgL = cv2.imread(f'rectframes/rimgL_{j}.jpg')
        rimgR = cv2.imread(f'rectframes/rimgR_{j}.jpg')
        disparity_map = stereo.compute(rimgL, rimgR).astype(np.float32) - 32
        im3d = cv2.reprojectImageTo3D(disparity_map/16, stereo_cal.get('Q'), handleMissingValues = True)
        im3d[im3d == np.inf] = 0
        im3d[im3d > 9_000] = 0
        im3d[im3d < -9_000] = 0
        im3d[im3d == -np.inf] = 0
        depths = np.sqrt(im3d[:,:,0]**2 + im3d[:,:,1]**2 + im3d[:,:,2]**2)
        
        camera_time += 5/3600
        for time in decimal_time:
            if np.abs(time-camera_time) < 2.5/3600:
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
study.enqueue_trial({'max_disp': 4,
                     'win_size': 11,
                     'block_size': 11,
                     'uniqueness_ratio': 13, 
                     'speckle_window_size': 176,
                     'speckle_range': 3,
                     'disp12_max_diff': 6})
study.optimize(objective, n_trials=100)