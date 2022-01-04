import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from mask2 import img1_masked, img3_masked
from bearings import rotation_matrix, translation_vector, baseline_dist
#from rectifiedUnrectifiedMapping import C1_H, lidarToRectified, unrectifiedToRectified
#from lidarCameraMatching import lidar_coords, camera_coords, lidarH

#from fundamental_estimate import fundamental_matrix, inlier_mask, pts1, pts2,essential_matrix,T, R1, R2, P1, P2, Q, H1, H2, roi_left, roi_right, mapx, mapy
from mask2 import mask_C1, mask_C3

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

def gaussian_contrast(img):
    img = cv.GaussianBlur(img, (7,7),0)
    #img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    l, a, b = cv.split(img) #Splitting the LAB image to different channels
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(1,1)) #Applying CLAHE to L-channel
    cl = clahe.apply(l)
    limg = cv.merge((cl,a,b)) #Merge the CLAHE enhanced L-channel with the a and b channel
    #final = cv.cvtColor(limg, cv.COLOR_BGR2GRAY)
    return limg

#imgR = img1_masked #gaussian_contrast(img1_masked)
#imgL = img3_masked #gaussian_contrast(img3_masked)

#imgR = #

#use this code if you want to mask sky and buildings
vidcapR = cv.VideoCapture(r'Videos/lowres_C1_2021-10-09_12A.mp4')
vidcapL = cv.VideoCapture(r'Videos/C3_2021-10-09_12A.mp4')
success, imgR = vidcapR.read()
success2, imgL = vidcapL.read()


#rg_ratio1 = imgR[:, :, 1]/imgR[:, :, 2]
#mask_C1 =  mask_C1[:, :, 0] & (rg_ratio1<1.1) & (rg_ratio1>0.93)  #
#imgR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
#imgR = cv.remap(imgR, mapx, mapy, cv.INTER_LINEAR)
imgR = cv.bitwise_and(imgR, imgR, mask=mask_C1[:, :, 0])

#rg_ratio2 = imgL[:, :, 1]/imgL[:, :, 2]
#mask_C3 =  mask_C3[:, :, 0] & (rg_ratio1<1.1) & (rg_ratio1>0.93)  #
#imgL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
##imgL = cv.remap(imgL, mapx, mapy, cv.INTER_LINEAR)
imgL = cv.bitwise_and(imgL, imgL, mask=mask_C3[:, :, 0])

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


#assume both images have same height and width
h,w = imgL.shape[:2]
print(h,w)

new_camera_matrixleft, roi = cv.getOptimalNewCameraMatrix(CamM_left,Distort_left,(w,h),1,(w,h))
new_camera_matrixright, roi = cv.getOptimalNewCameraMatrix(CamM_right,Distort_right,(w,h),1,(w,h))

#Undistort images
imgR_undistorted = cv.undistort(imgR, CamM_right, Distort_right, None, new_camera_matrixright)
imgL_undistorted = cv.undistort(imgL, CamM_left, Distort_left, None, new_camera_matrixleft)

#Set disparity parameters
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
#Compute disparity map
print ("\nComputing the disparity  map...")
#disparity_map = stereo.compute(img1_undistorted, img2_undistorted)

#Show disparity map before generating 3D cloud to verify that point cloud will be usable.
#fig, (ax1,ax2,ax3) = plt.subplots(1,3)
#ax1.imshow(img1_undistorted)
#ax2.imshow(img2_undistorted)
#disp1 = ax3.imshow(disparity_map,'coolwarm')
#divider1 = make_axes_locatable(ax3)
#cax1 = divider1.append_axes('right', size='5%', pad=0.05)
#fig.colorbar(disp1, cax=cax1, orientation='vertical')
#plt.show()

# Find Fundamental matrix by calculating image points with SIFT algorithm,
# then matching with Flann based matcher
# Initiate SIFT detector
##sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
#kp1, des1 = sift.detectAndCompute(img1_undistorted, None)
##kp2, des2 = sift.detectAndCompute(img2_undistorted, None)

#imgSift = cv.drawKeypoints(
    #img2_undistorted, kp2, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#cv.imshow("SIFT Keypoints", imgSift)
#plt.show()

#FLANN_INDEX_KDTREE = 1
#index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
#search_params = dict(checks=50)   # or pass empty dictionary
#flann = cv.FlannBasedMatcher(index_params, search_params)
#matches = flann.knnMatch(des1, des2, k=2)

# Keep good matches: calculate distinctive image features
# Lowe, D.G. Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision 60, 91–110 (2004). https://doi.org/10.1023/B:VISI.0000029664.99615.94
# https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
#matchesMask = [[0, 0] for i in range(len(matches))]


# Stereo rectification (uncalibrated variant)
# Adapted from: https://stackoverflow.com/a/62607343
hL, wL = imgL.shape[:2]
hR, wR = imgR.shape[:2]
#_, H1, H2 = cv.stereoRectifyUncalibrated(
 ##   np.float32(pts1), np.float32(pts2), fundamental_matrix, imgSize=(w1, h1)


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

# # Undistort (rectify) the images and save them
# # Adapted from: https://stackoverflow.com/a/62607343
#rimg1 = cv.warpPerspective(img1, H1, (w1, h1))
#rimg2 = cv.warpPerspective(img2, H2, (w2, h2))
# cv.imwrite("rectified_1.png", img1_rectified)
# cv.imwrite("rectified_2.png", img2_rectified)

# compute disparity map for the rectified images
disparity_map2 = stereo.compute(rimgL, rimgR).astype(np.float32)


im3d = cv.reprojectImageTo3D(disparity_map2/32, Q, handleMissingValues = True)

im3d[im3d == np.inf] = 0
im3d[im3d == 10000] = 0
im3d[im3d == -np.inf] = 0
im3d[im3d > 9000] = 0
depths = np.sqrt(im3d[:,:,0]**2 + im3d[:,:,1]**2 + im3d[:,:,2]**2)

# print(im3d)
#trying to get distance idk if it works yet
fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4, sharex=True, sharey = True)
ax1.imshow(rimgR,'gray')
ax1.set_xlabel('Camera 1')
ax2.imshow(rimgL,'gray')
ax2.set_xlabel('Camera 3')
ax3.set_xlabel('Disparity Map')
ax4.set_xlabel('distance map')
disp = ax3.imshow(disparity_map2,'coolwarm')
divider = make_axes_locatable(ax3)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(disp, cax=cax, orientation='vertical')

#disparity_map2.flatten()
#depth = baseline_dist*CamM_1[0,0]/disparity_map2



#depth = [disparity_map2[0], disparity_map2[1], D]
#print(np.shape(depth))
#print(np.shape(disparity_map2))

#print('Depth', D)
depth = ax4.imshow(depths, 'coolwarm')
#print(np.dtype(depth))
divider2 = make_axes_locatable(ax4)
cax2 = divider2.append_axes('right', size='5%', pad=0.05)
#fig.colorbar(depth, cax=cax2, orientation='vertical')

fig.colorbar(depth, cax=cax2, orientation='vertical')

plt.show()


# fig, ax = plt.subplots(1,1)

# depths = disparity_map2.flatten()
# depths = depths[depths < 800]
# depths = depths[depths > 200]
# ax.hist(depths, bins=40)
# plt.show()






#from bearings import  baseline_dist
print('base line dist', baseline_dist)
print('f', CamM_left[0,0])
print('disparity', disparity_map2)
