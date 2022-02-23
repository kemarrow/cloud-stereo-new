import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from mask2 import img1_masked, img3_masked
from bearings import rotation_matrix, translation_vector, baseline_dist
from rectifiedUnrectifiedMapping import C1_H, lidarToRectifiedSingle
from lidarCameraMatching import lidar_coords, camera_coords, lidarH
from plotLidarOnImage import success_coords, decimalToNormal
from lidarPlots import LidarData, findCloud
from scipy.optimize import minimize, LinearConstraint

#from fundamental_estimate import fundamental_matrix, inlier_mask, pts1, pts2,essential_matrix,T, R1, R2, P1, P2, Q, H1, H2, roi_left, roi_right, mapx, mapy
from mask2 import mask_C1, mask_C3

def flattenList(t):
    flat_list = []
    for sublist in t:
        for item in sublist:
            flat_list.append(item)
    return flat_list

def singleTransform(coord, mat):
    coord = list(coord)
    coord = np.append(coord, [1])
    val = np.matmul(mat, coord)
    val[0] = val[0]/val[2]
    val[1] = val[1]/val[2]
    val = np.array(val)
    return val   

def gaussian_contrast(img):
    img = cv.GaussianBlur(img, (7,7),0)
    #img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    l, a, b = cv.split(img) #Splitting the LAB image to different channels
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(1,1)) #Applying CLAHE to L-channel
    cl = clahe.apply(l)
    limg = cv.merge((cl,a,b)) #Merge the CLAHE enhanced L-channel with the a and b channel
    #final = cv.cvtColor(limg, cv.COLOR_BGR2GRAY)
    return limg

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

#camera matrices and distortion for the 640x480 resolution
#i am using the 640x480v2 values for both cameras because camera3 doesn't have its own
CamM_left = np.array([[5.520688775958645920e+02,0.000000000000000000e+00,3.225866125962970159e+02],
          [0.000000000000000000e+00,5.502640890663026312e+02,2.362389385357402034e+02],
          [0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00]])

CamM_right = np.array([[5.520688775958645920e+02,0.000000000000000000e+00,3.225866125962970159e+02],
          [0.000000000000000000e+00,5.502640890663026312e+02,2.362389385357402034e+02],
          [0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00]])


Distort_left = np.array([2.808374038768443048e-01,-9.909134707088265159e-01,6.299531255281858727e-04,-1.301770463801651002e-03,1.093982545460403522e+00])
Distort_right = np.array([2.808374038768443048e-01,-9.909134707088265159e-01,6.299531255281858727e-04,-1.301770463801651002e-03,1.093982545460403522e+00])

w1, h1 = 640, 480

#Create stereo matching object and set disparity parameters
#Note: disparity range is tuned according to specific parameters obtained through trial and error.
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

#use this code if you want to mask sky and buildings
vidcapR = cv.VideoCapture('Videos/lowres_C1_2021-10-04_11A.mp4')
vidcapL = cv.VideoCapture('Videos/C3_2021-10-04_11A.mp4')

#win_size = 5
#min_disp = 0
#max_disp = 64

#function to minimise
def function(trial):#, win_size, min_disp, max_disp,block_size,uniqueness_ratio,speckle_windowSize, speckle_range, disp12_maxDiff):
    #probably a better way to make everything integers...

    win_size = trial.suggest_float('win_size ',0, 10)
    win_size = int(win_size)
    min_disp = trial.suggest_float('min_disp',0, 10)
    max_disp = trial.suggest_float('max_disp ',0, 10)
    block_size = trial.suggest_float('block_size ',0, 10)
    uniqueness_ratio =  trial.suggest_float('uniqueness_ratio',0, 10)
    speckle_windowSize = trial.suggest_float('speckle_windowSize',50, 150)
    speckle_range = trial.suggest_float('speckle_range',0, 10)
    disp12_maxDiff = trial.suggest_float('disp12_maxDiff',8, 26)
    num_disp = 16*max_disp - 16*min_disp #need to change this to multiple of 16, not sure how yet
                                #p1 = 8*3*a**2,
                                #p2 =32*3*a**2 
    win_size = int(win_size)
    min_disp = int(min_disp)
    max_disp = int(max_disp)
    block_size = int(block_size)
    uniqueness_ratio= int(uniqueness_ratio)
    speckle_windowSize  = int(speckle_windowSize)
    speckle_range  = int(speckle_range)
    disp12_maxDiff  = int(disp12_maxDiff)
    num_disp  = int(num_disp)

                                # Needs to be divisible by 16
                                #Create Block matching object.
    stereo = cv.StereoSGBM_create(minDisparity= min_disp,
        numDisparities = num_disp,
        blockSize = block_size,
        uniquenessRatio = uniqueness_ratio,
        speckleWindowSize =  speckle_windowSize,
        speckleRange = speckle_range,
        disp12MaxDiff = disp12_maxDiff,
        P1= 8*3*win_size**2,
        P2=32*3*win_size**2 ) 

    camera_time = 11
    stereo_depths = []
    lidar_depths = []
    disparity = []
    while(vidcapR.isOpened() and vidcapL.isOpened()):
            success, imgR = vidcapR.read()
            success2, imgL = vidcapL.read()
            if success==True and success==True:
                #imgR = gaussian_contrast(imgR)
                    #imgL = gaussian_contrast(imgL)
                newcameramtx, roi = cv.getOptimalNewCameraMatrix(
                CamM_left, Distort_left, (w1, h1), 1, (w1, h1))
                mapx, mapy = cv.initUndistortRectifyMap(
                    CamM_left, Distort_left, None, newcameramtx, (w1, h1), 5)
        
                imgR = cv.bitwise_and(imgR, imgR, mask=mask_C1)
                imgL = cv.bitwise_and(imgL, imgL, mask=mask_C3)
        
                                        #assume both images have same height and width
                new_camera_matrixleft, roi = cv.getOptimalNewCameraMatrix(CamM_left,Distort_left,(w1,h1),1,(w1,h1))
                new_camera_matrixright, roi = cv.getOptimalNewCameraMatrix(CamM_right,Distort_right,(w1,h1),1,(w1,h1))
        
                                        #Undistort images
                imgR_undistorted = cv.undistort(imgR, CamM_right, Distort_right, None, new_camera_matrixright)
                imgL_undistorted = cv.undistort(imgL, CamM_left, Distort_left, None, new_camera_matrixleft)
        
                                        #creates new map for each camera with the rotation and pose (R and P) values
                mapLx, mapLy = cv.initUndistortRectifyMap(new_camera_matrixleft, Distort_left, Rleft, Pleft, (w1,h1), cv.CV_32FC1)
                mapRx, mapRy = cv.initUndistortRectifyMap(new_camera_matrixright, Distort_right, Rright, Pright, (w1,h1), cv.CV_32FC1)
        
                                        # remaps each image to the new map
                rimgR = cv.remap(imgR, mapRx, mapRy,
                    interpolation=cv.INTER_NEAREST,
                    borderMode=cv.BORDER_CONSTANT,
                    borderValue=(0, 0, 0, 0))
                rimgL = cv.remap(imgL, mapLx, mapLy,
                    interpolation=cv.INTER_NEAREST,
                    borderMode=cv.BORDER_CONSTANT,
                    borderValue=(0, 0, 0, 0))
        
                # compute disparity map for the rectified images
                disparity_map2 = stereo.compute(rimgL, rimgR).astype(np.float32)
                                        #disp_masked = cv.bitwise_and(disparity_map2, disparity_map2, mask=mask_C1)
        
                im3d = cv.reprojectImageTo3D(disparity_map2/32, Q, handleMissingValues = True)
        
                im3d[im3d == np.inf] = 0
                im3d[im3d > 9_000] = 0
                im3d[im3d == -np.inf] = 0
        
                depths = im3d[:,:,2]
        
                camera_time += 5/3600
                for time in decimal_time:
                    if np.abs(time-camera_time) < 0.001:
                        i = list(decimal_time).index(time)
                        cloudindex, cloud = findCloud(backscatter[i], 500)
                        point = np.array((azi[i], elev[i]))
                        new = lidarToRectifiedSingle(point, lidarH, C1_H)
                        if cloud is not None:
                             if int(new[0]) < 640 and int(new[1]) < 480:
                                if depths[int(new[1]), int(new[0])] != 0: #error here, idk why: IndexError: index 452 is out of bounds for axis 0 with size 5
                                    stereo_depths.append(depths[int(new[1]), int(new[0])])
                                    lidar_depths.append(cloud)
                                else:
                                    pass        
                        if cv.waitKey(1) & 0xFF == ord('q'):
                                break 
                p = np.polynomial.polynomial.polyfit(lidar_depths, stereo_depths, 1)
                ffit = np.poly1d(p[::-1])

    vidcapL.release()
    vidcapR.release()
    cv.destroyAllWindows()

    return abs(p[1]-1)

import optuna



study = optuna.create_study()
p = study.optimize(function, n_trials=100)
print('p', p)

min_disp = np.linspace(0, 4, 1)
max_disp = np.linspace(3, 7, 1)
block_size = np.linspace(3, 17, 1)
uniqueness_ratio = np.linspace(4, 18, 1)
speckle_windowSize = np.linspace(50, 151, 5)
speckle_range = np.linspace(2, 11, 1)
disp12_maxDiff = np.linspace(8, 26,1)
#max_disp,block_size,uniqueness_ratio,speckle_windowSize, speckle_range, disp12_maxDiff

#bnds = ((0, 10), (0,3), (3,6), (3,16), (4,17), (50,150), (2,10), (8,25))

#p0 = [5, 2, 4, 3, 10, 100, 3, 25]
#haven't figured it out yet, issue with inputs needing to be integers and not sure how to use the bounds and constraints arguments
#res = minimize(function, p0, args = (min_disp, max_disp,block_size,uniqueness_ratio,speckle_windowSize, speckle_range, disp12_maxDiff), bounds = bnds)

#print(res)




