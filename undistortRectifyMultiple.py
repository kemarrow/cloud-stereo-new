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
from mask2 import mask_C1, mask_C3
#from rectifiedUnrectifiedMapping import C1_H

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

#images with masked buildings
# imgR = img1_masked
# imgL = img3_masked
#%%
# #Camera matrices and distortion for the 1280x960 resolution (camera 1 and camera 2)
# CamM_1 = np.array([[5.115201885073643666e+02,0.0,3.069700637546008579e+02],
#                   [0.0,5.110005291739051358e+02,2.441830592759120862e+02],
#                   [0.0,0.0,1.0]])
# CamM_2 = np.array([[5.063166013857816665e+02,0.0,3.166661931952413056e+02],
#                    [1.0,5.067357512519903935e+02,2.526423174030390157e+02],
#                    [0.0,0.0,1.0]])

# Distort_1 = np.array([2.791793909906162274e-01,-7.229131626061370275e-01,3.998341915440934737e-03,2.866146329555672653e-03,6.224929102783540724e-01])
# Distort_2 = np.array([2.326755157584974587e-01,-6.011054678561147391e-01,3.963575587693899294e-04,-2.566491984608918874e-04,4.822591716560123420e-01])
#%%
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

w, h = 640, 480

#Create stereo matching object and set disparity parameters
#Note: disparity range is tuned according to specific parameters obtained through trial and error.
win_size = 8
min_disp = 1
max_disp = 7
num_disp = 16*max_disp - 16*min_disp # Needs to be divisible by 16
#Create Block matching object.
stereo = cv.StereoSGBM_create(minDisparity= min_disp,
 numDisparities = num_disp,
 blockSize = 9,
 uniquenessRatio = 13,
 speckleWindowSize = 5,
 speckleRange = 14,
 disp12MaxDiff = 7,
 P1 = 8*3*win_size**2,
 P2 =32*3*win_size**2)


decimal_time = []
azi = []
elev = []
distance = []
backscatter = []
for dtime in [1,2,3,4,5]:
        filename = 'lidar/User5_18_20211009_12{0}000.hpl.txt'.format(dtime)
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
vidcapR = cv.VideoCapture('Videos/lowres_C1_2021-10-09_12A.mp4')
vidcapL = cv.VideoCapture('Videos/C3_2021-10-09_12A.mp4')

camera_time = 11
stereo_depths = []
stereo_coords = []
time_points = []
lidar_depths = []
disparity = []
while(vidcapR.isOpened() and vidcapL.isOpened()):
    success, imgR = vidcapR.read()
    success2, imgL = vidcapL.read()
    if success==True and success2==True:

        imgR = gaussian_contrast(imgR)
        imgL = gaussian_contrast(imgL)
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(
            CamM_left, Distort_left, (w, h), 1, (w, h))
        mapx, mapy = cv.initUndistortRectifyMap(
            CamM_left, Distort_left, None, newcameramtx, (w, h),  cv.CV_32FC1)

        rg_ratio1 = imgR[:, :, 1]/imgR[:, :, 2]
        skymask_C1 =  mask_C1[:,:,0] & (rg_ratio1<1.1) & (rg_ratio1>0.93)  #
        imgR_skymask = (cv.cvtColor(imgR, cv.COLOR_BGR2GRAY))
        imgR_skymask = cv.remap(imgR_skymask, mapx, mapy, cv.INTER_LINEAR)
        imgR_skymask = cv.bitwise_and(imgR_skymask, imgR_skymask, mask=skymask_C1)

        #rg_ratio2 = imgL[:, :, 1]/imgL[:, :, 2]
        #mask_C3 =  mask_C3[:,:,0] & (rg_ratio1<1.1) & (rg_ratio1>0.93)  #
        #imgR_skymask = (cv.cvtColor(imgL, cv.COLOR_BGR2GRAY))
        # imgR_skymask = cv.remap(imgR_skymask, mapx, mapy, cv.INTER_LINEAR)
        # imgR_skymask = cv.bitwise_and(imgR_skymask, imgR_skymask, mask=mask_C3)

        imgR = cv.bitwise_and(imgR, imgR, mask=mask_C1)
        imgL = cv.bitwise_and(imgL, imgL, mask=mask_C3)

        #assume both images have same height and width

        new_camera_matrixleft, roi = cv.getOptimalNewCameraMatrix(CamM_left,Distort_left,(w,h),1,(w,h))
        new_camera_matrixright, roi = cv.getOptimalNewCameraMatrix(CamM_right,Distort_right,(w,h),1,(w,h))

        #Undistort images
        imgR_undistorted = cv.undistort(imgR, CamM_right, Distort_right, None, new_camera_matrixright)
        imgL_undistorted = cv.undistort(imgL, CamM_left, Distort_left, None, new_camera_matrixleft)

        #creates new map for each camera with the rotation and pose (R and P) values
        mapLx, mapLy = cv.initUndistortRectifyMap(new_camera_matrixleft, Distort_left, Rleft, Pleft, (w,h), cv.CV_32FC1)
        mapRx, mapRy = cv.initUndistortRectifyMap(new_camera_matrixright, Distort_right, Rright, Pright, (w,h), cv.CV_32FC1)

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

        mask_new = mask_C1 & mask_C3 #new to rotate and translate the mask
        disp_masked = cv.bitwise_and(disparity_map2, disparity_map2, mask=mask_new)

        im3d = cv.reprojectImageTo3D(disp_masked/32, Q, handleMissingValues = True)

        im3d = cv.reprojectImageTo3D(disparity_map2/32, Q, handleMissingValues = True)
        im3d[im3d == np.inf] = 0
        im3d[im3d > 9_000] = 0
        im3d[im3d == -np.inf] = 0

        depths = im3d[:,:,2]
        stereo_x = im3d[:,:,0]
        stereo_y = im3d[:,:,1]
        stereo_z = im3d[:,:,2]
        depths = np.sqrt( im3d[:,:,0]**2 + im3d[:,:,1]**2 + im3d[:,:,2]**2)

        rg_ratio1 = rimgR[:, :, 1]/rimgR[:, :, 2]
        skymask_C1 =  mask_C1[:,:,0] & (rg_ratio1<1.1) & (rg_ratio1>0.93)  #
        depths_skymask = cv.bitwise_and(depths, depths, mask=skymask_C1)

        camera_time += 5/3600
        for time in decimal_time:
            if np.abs(time-camera_time) < 0.001:
                i = list(decimal_time).index(time)
                cloudindex, cloud = findCloud(backscatter[i], 500)
                point = np.array((azi[i], elev[i]))
                new = lidarToRectifiedSingle(point, lidarH, C1_H)
                if cloud is not None:
                    if int(new[0]) < 640 and int(new[1]) < 480:
                        if depths[int(new[1]), int(new[0])] != 0:
                            stereo_depths.append(depths[int(new[1]), int(new[0])])
                            stereo_coords.append(np.array([new[0], new[1]]))  #check x and y are the right way around
                            time_points.append(camera_time) #currently in decimal time
                            lidar_depths.append(cloud)
                else:
                    pass

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break



    # When everything done, release the video capture object
vidcapL.release()
vidcapR.release()

    #coefs = np.polynomial.polynomial.polyfit(lidar_depths, stereo_depths, 1)
    #ffit = np.poly1d(coefs[::-1])

# Closes all the frames
cv.destroyAllWindows()

print(stereo_depths)

plt.plot(lidar_depths, stereo_depths,'.')
#%%
fig, ax = plt.subplots(1,1)
ax.plot(lidar_depths, stereo_depths,'.')
ax.set_xlabel('lidar depths (m)')
ax.set_ylabel('stereo depths (m)')
ax.set_ylim(0, 9000)
#plt.savefig('xyzSquaredDepthsDiv16.png', bbox_inches='tight')
coefs = np.polynomial.polynomial.polyfit(lidar_depths, stereo_depths, 1)
ffit = np.poly1d(coefs[::-1])

x_new = np.linspace(min(lidar_depths), max(lidar_depths))
ax.plot(x_new, x_new, label='y=x')
ax.legend()
#ax.plot(x_new, ffit(x_new), label='fit')
plt.show()

from scipy.stats import pearsonr
print(pearsonr(lidar_depths, stereo_depths))
#%%
#find the points which have stereo distance overestimated
anomalies = []
anomaly_times = []
anomalous_stereo = []
anomalous_lidar = []
for i in range (0, len(lidar_depths)):
    if stereo_depths[i] > lidar_depths[i] + 1600:
        anomalies.append(stereo_coords[i])
        anomaly_times.append(time_points[i])
        anomalous_stereo.append(stereo_depths[i])
        anomalous_lidar.append(lidar_depths[i])
        ax.plot(lidar_depths[i], stereo_depths[i], 'r.')
anomalies = np.array(anomalies)

# fig, ax = plt.subplots(1,1)
# ax.imshow(disparity_map2, cmap='coolwarm')
# ax.plot(anomalies[:,0], anomalies[:,1], 'r.')

frame = (np.array(anomaly_times) - 11) * (3600/5)

f = 0
vidcapR = cv.VideoCapture('Videos/lowres_C1_2021-10-09_12A.mp4')
while vidcapR.read()[0]:
    for j in range(0, 20):
        if f == int(frame[j]):
            fig, ax = plt.subplots(1,1)
            ax.imshow(vidcapR.read()[1])
            ax.plot(anomalies[j,0], anomalies[j,1], 'rx')
            print(int(frame[j]))
            print(anomalous_stereo[j])
            print(anomalous_lidar[j])
    f += 1
vidcapR.release()
#%%

#find the points which have stereo distance underestimated
underestimates = []
underestimate_times = []
underestimate_stereo = []
underestimate_lidar = []
for i in range (0, len(lidar_depths)):
    if stereo_depths[i] > lidar_depths[i] + 1600:
        underestimates.append(stereo_coords[i])
        underestimate_times.append(time_points[i])
        underestimate_stereo.append(stereo_depths[i])
        underestimate_lidar.append(lidar_depths[i])
        ax.plot(lidar_depths[i], stereo_depths[i], 'r.')
underestimates = np.array(underestimates)

# fig, ax = plt.subplots(1,1)
# ax.imshow(disparity_map2, cmap='coolwarm')
# ax.plot(anomalies[:,0], anomalies[:,1], 'r.')

frame = (np.array(underestimate_times) - 11) * (3600/5)

f = 0
vidcapR = cv.VideoCapture('Videos/lowres_C1_2021-10-04_11A.mp4')
while vidcapR.read()[0]:
    for j in range(0, 20):
        if f == int(frame[j]):
            fig, ax = plt.subplots(1,1)
            ax.imshow(vidcapR.read()[1])
            ax.plot(underestimates[j,0], underestimates[j,1], 'rx')
            print(int(frame[j]))
            print(underestimate_stereo[j])
            print(underestimate_lidar[j])
    f += 1
vidcapR.release()


#%% #this code block plots the world coordinates
from haversine import inverse_haversine, Direction
from mpl_toolkits import mplot3d

camera1 = np.array([51.49880908055068, -0.1788492157810761, 30])
camera3 = np.array([51.4993318750954, -0.17901837289811393, 46])

depths = depths.flatten()
stereo_xz = np.sqrt(np.array(stereo_x)**2 + np.array(stereo_z)**2)

angles = np.arctan(np.array(stereo_z)/np.array(stereo_x))
world_coords = []
for j in range(0, len(angles)):
    coord = inverse_haversine([camera3[0], camera3[1]], stereo_xz[j]/1000, angles[j])
    world_coords.append(coord)


world_coords = np.array(world_coords)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(world_coords[:,1], world_coords[:,0], depths)
ax.plot(camera3[1], camera3[0], 0, 'rx', label = 'position of camera 3')
ax.set_xlabel('longitude')
ax.set_ylabel('latitude')
ax.set_zlabel('height')
#ax.set_xlim(51.498,51.53)
#ax.set_ylim(-0.181, -0.1)
ax.legend()
#plt.savefig('worldCoords.png', bbox_inches = 'tight')
#print(world_coords)
