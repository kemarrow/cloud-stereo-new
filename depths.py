"""
disparity and depth maps for multiple frames of a video 

@author: Erin
"""
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mask2 import img1_masked, img3_masked
from bearings import rotation_matrix, translation_vector, baseline_dist
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

vidcapR = cv.VideoCapture(r'Videos/lowres_C1_2021-10-09_12A.mp4')
vidcapL = cv.VideoCapture(r'Videos/C3_2021-10-09_12A.mp4')

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


#assume both images have same height and width
h  = 480
w = 640

#find new camera matrix  
new_camera_matrixleft, roi = cv.getOptimalNewCameraMatrix(CamM_left,Distort_left,(w,h),1,(w,h))
new_camera_matrixright, roi = cv.getOptimalNewCameraMatrix(CamM_right,Distort_right,(w,h),1,(w,h))

#creates new map for each camera with the rotation and pose (R and P) values
mapLx, mapLy = cv.initUndistortRectifyMap(new_camera_matrixleft, Distort_left, Rleft, Pleft, (w,h), cv.CV_32FC1)
mapRx, mapRy = cv.initUndistortRectifyMap(new_camera_matrixright, Distort_right, Rright, Pright, (w,h), cv.CV_32FC1)

while(vidcapR.isOpened() and vidcapL.isOpened()):
    success, imgR1 = vidcapR.read()
    success2, imgL1 = vidcapL.read()
    if success==True and success2==True:
        #apply mask to buildings for each image
        imgR = cv.bitwise_and(imgR1, imgR1, mask=mask_C1[:, :, 0])
        imgL = cv.bitwise_and(imgL1, imgL1, mask=mask_C3[:, :, 0])

        #Undistort images
        imgR_undistorted = cv.undistort(imgR, CamM_right, Distort_right, None, new_camera_matrixright)
        imgL_undistorted = cv.undistort(imgL, CamM_left, Distort_left, None, new_camera_matrixleft)

        # remaps each image to the new map
        rimgR = cv.remap(imgR, mapRx, mapRy,
                            interpolation=cv.INTER_NEAREST,
                            borderMode=cv.BORDER_CONSTANT,
                            borderValue=(0, 0, 0, 0))
        rimgL = cv.remap(imgL, mapLx, mapLy,
                            interpolation=cv.INTER_NEAREST,
                            borderMode=cv.BORDER_CONSTANT,
                            borderValue=(0, 0, 0, 0))
                
        #combine mask1 and mask3 and rectify 
        new_mask = mask_C1[:,:,0] & mask_C3[:,:,0]
        disp_mask = cv.remap(new_mask, mapLx, mapLy,
                            interpolation=cv.INTER_NEAREST,
                            borderMode=cv.BORDER_CONSTANT,
                            borderValue=(0, 0, 0, 0))

        #compute disparity map for the rectified images
        disparity_map2 = stereo.compute(rimgL, rimgR).astype(np.float32)

        im3d = cv.reprojectImageTo3D(disparity_map2/32, Q, handleMissingValues = True)

        #set out of range depths to 0
        im3d[im3d == np.inf] = 0
        im3d[im3d == 10000] = 0
        im3d[im3d == -np.inf] = 0
        im3d[im3d > 9000] = 0
        
        depths = np.sqrt(im3d[:,:,0]**2 + im3d[:,:,1]**2 + im3d[:,:,2]**2)
        depths = cv.bitwise_and(depths, depths, mask=disp_mask)
        #plot
        fig, (ax1,ax2,ax3,ax4, ax5) = plt.subplots(1,5, sharex=True, sharey = True)
        ax5.imshow(disp_mask)
        ax1.imshow(rimgR,'gray')
        ax1.set_xlabel('Camera 1')

        ax2.imshow(rimgL,'gray')
        ax2.set_xlabel('Camera 3')
        
        ax3.set_xlabel('Disparity Map')
        disp = ax3.imshow(disparity_map2,'coolwarm')
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(disp, cax=cax, orientation='vertical')

        depth = ax4.imshow(depths, 'coolwarm')
        ax4.set_xlabel('distance map')
        divider2 = make_axes_locatable(ax4)
        cax2 = divider2.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(depth, cax=cax2, orientation='vertical')
        plt.show()

        if cv.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        break

cv.destroyAllWindows()
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
