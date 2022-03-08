
import numpy as np
import cv2 as cv
#from bearings import rotation_matrix, translation_vector, baseline_dist
from mask2 import mask_C1, mask_C3
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.image import NonUniformImage
import pickle
import ptlflow
from ptlflow.utils import flow_utils
from ptlflow.utils.io_adapter import IOAdapter
import pandas  as pd

stereo_cal = pickle.load(open( 'stereo_cal_mat.pkl', "rb" ) )

fundamental_matrix = stereo_cal.get('F')
essential_matrix = stereo_cal.get('E')
R = stereo_cal.get('R')
T = stereo_cal.get('T')
Rleft = stereo_cal.get('Rleft')
Rright = stereo_cal.get('Rright')
Pleft = stereo_cal.get('Pleft')
Pright = stereo_cal.get('Pright')
Q = stereo_cal.get('Q')
Hleft = stereo_cal.get('Hleft')
Hright = stereo_cal.get('Hright')
roi_left = stereo_cal.get('roi_left')
roi_left = stereo_cal.get('roi_right')


CamM_left = np.array([[5.520688775958645920e+02,0.000000000000000000e+00,3.225866125962970159e+02],
          [0.000000000000000000e+00,5.502640890663026312e+02,2.362389385357402034e+02],
          [0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00]])

CamM_right = np.array([[5.520688775958645920e+02,0.000000000000000000e+00,3.225866125962970159e+02],
          [0.000000000000000000e+00,5.502640890663026312e+02,2.362389385357402034e+02],
          [0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00]])

Distort_left = np.array([2.808374038768443048e-01,-9.909134707088265159e-01,6.299531255281858727e-04,-1.301770463801651002e-03,1.093982545460403522e+00])
Distort_right = np.array([2.808374038768443048e-01,-9.909134707088265159e-01,6.299531255281858727e-04,-1.301770463801651002e-03,1.093982545460403522e+00])

h, w  = 480, 640

new_camera_matrixleft, roi = cv.getOptimalNewCameraMatrix(CamM_left,Distort_left,(w,h),1,(w,h))
new_camera_matrixright, roi = cv.getOptimalNewCameraMatrix(CamM_right,Distort_right,(w,h),1,(w,h))

mapLx, mapLy = cv.initUndistortRectifyMap(new_camera_matrixleft, Distort_left, Rleft, Pleft, (w,h), cv.CV_32FC1)
mapRx, mapRy = cv.initUndistortRectifyMap(new_camera_matrixright, Distort_right, Rright, Pright, (w,h), cv.CV_32FC1)

#model = ptlflow.get_model('flownet2', pretrained_ckpt='things') #other models are here: https://ptlflow.readthedocs.io/en/latest/models/models_list.html

#Set disparity parameters
#Note: disparity range is tuned according to specific parameters obtained through trial and error.
win_size = 8
min_disp = 0
max_disp = 7
num_disp = 16*max_disp - 16*min_disp # Needs to be divisible by 16

#Create Block matching object.
#stereo = cv.StereoSGBM_create(minDisparity= min_disp,
 #numDisparities = num_disp,
 #blockSize = 9,
 #uniquenessRatio = 13,
 #speckleWindowSize = 5,
 #speckleRange = 14,
 #disp12MaxDiff = 7,
 #P1 = 8*3*win_size**2,
 #P2 =32*3*win_size**2)

win_size = 11
min_disp = 0
max_disp = 4
num_disp = 16*max_disp - 16*min_disp # Needs to be divisible by 16
#Create semi global block matching object
stereo = cv.StereoSGBM_create(minDisparity= min_disp,
 numDisparities = num_disp,
 blockSize = 11,
 uniquenessRatio = 13,
 speckleWindowSize = 176,
 speckleRange = 3,
 disp12MaxDiff = 6,
 P1 = 8*3*win_size**2,
 P2 =32*3*win_size**2)

#It is based on Gunner Farneback's algorithm which is explained in "Two-Frame Motion Estimation Based on Polynomial Expansion" by Gunner Farneback in 2003.
#date = '2021-10-24_12A'
#vidcapR = cv.VideoCapture('Videos/lowres_C1_'+ date+'.mp4')
#vidcapL = cv.VideoCapture('Videos/C3_'+ date+'.mp4')
#cap = cv.VideoCapture('Videos/C3_'+date+'.mp4')

#cap2 = cv.VideoCapture('depth_10_24_12A.mp4')
#ret, frame1 = cap.read()
#backSub = cv.createBackgroundSubtractorKNN()
#backSub = cv.createBackgroundSubtractorMOG2(detectShadows= True) #default is True, not sure which one to choose


#camera field of views


imgR = cv.imread('cam2.png')
imgL = cv.imread('cam1.png')

#imgR = cv.undistort(imgR, CamM_right, Distort_right, None, new_camera_matrixright)
#imgL = cv.undistort(imgL, CamM_left, Distort_left, None, new_camera_matrixleft)

#prvsR1 = cv.undistort(prvsR1, CamM_right, Distort_right, None, new_camera_matrixright)
#prvsL1 = cv.undistort(prvsL1, CamM_right, Distort_right, None, new_camera_matrixright)

    # remaps each image to the new map
rimgR = cv.remap(imgR, mapRx, mapRy,
                        interpolation=cv.INTER_NEAREST,
                        borderMode=cv.BORDER_CONSTANT,
                        borderValue=(0, 0, 0, 0))
rimgL = cv.remap(imgL, mapLx, mapLy,
                        interpolation=cv.INTER_NEAREST,
                        borderMode=cv.BORDER_CONSTANT,
                        borderValue=(0, 0, 0, 0))


    #previous image mapping

#bgr = backSub.apply(z)


#imgR = cv.bitwise_and(imgR, imgR, mask=mask_C1[:, :, 0])
#imgL = cv.bitwise_and(imgL, imgL, mask=mask_C3[:, :, 0])

#prvsR1  = cv.bitwise_and(prvsR1, prvsR1, mask=mask_C1[:, :, 0])
#prvsL1  = cv.bitwise_and(prvsL1, prvsL1, mask=mask_C3[:, :, 0])

    #Undistort images

                
#maskL = backSub.apply(imgL)
#maskL1 = backSub.apply(prvsL1)
#rimgRm = cv.bitwise_and(frame, frame, mask=fgMask)
#thresh1 = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
#ret,thresh1 = cv.threshold(thresh1,10,255,cv.THRESH_BINARY)
#newthing =  cv.bitwise_and(rimgL, rimgL, mask=thresh1)
#thresh1[thresh1==255] = None
#fig, (ax, ax2) = plt.subplots(1,2)
#ax.imshow(thresh1,cmap='gray')
#ax2.imshow(newthing)
#plt.show()

    #compute disparity map for the rectified images
disparity_map2 = stereo.compute(rimgL, rimgR).astype(np.float32)

        #combine mask1 and mask3 and rectify 
#maskcomb =  mask_C1[:,:,0] & mask_C3[:,:,0]
rg_ratio = imgL[:, :, 1]/imgL[:, :, 2]
new_mask =   (rg_ratio<1.1) & (rg_ratio>0.93)#maskL & thresh1




# disp_mask = cv.remap(new_mask, mapLx, mapLy,
#                         interpolation=cv.INTER_NEAREST,
#                         borderMode=cv.BORDER_CONSTANT,
#                         borderValue=(0, 0, 0, 0))



    #convert to depth
im3d = cv.reprojectImageTo3D((disparity_map2-32)/16, Q, handleMissingValues = True)


# im3d = cv.bitwise_and(im3d, im3d, mask=disp_mask)

    #set out of range depths to 0
im3d[im3d == np.inf] = None
im3d[im3d == -np.inf] = None
#im3d[im3d > 9000] = None
#im3d[im3d == 0]  = None

tilt = 23.7 * np.pi/180
height_camera = 46

depths = np.sqrt(im3d[:,:,0]**2 + im3d[:,:,1]**2 + im3d[:,:,2]**2)
# depths = cv.bitwise_and(depths, depths, mask=disp_mask)
depths[depths > 5000] = None
angle  = np.arccos(im3d[:,:,2]/depths)
z = depths * np.sin(angle + tilt) + height_camera
# z = cv.bitwise_and(z, z, mask=disp_mask)
# z[z==disp_mask]=None

fig, ((ax1,ax2,ax3),(ax4, ax5, ax6)) = plt.subplots(2,3, sharex =True, sharey= True)
rimgL =   cv.cvtColor(rimgL, cv.COLOR_RGB2BGR)
rimgR =   cv.cvtColor(rimgR, cv.COLOR_RGB2BGR)
ax1.imshow(rimgL)
ax2.imshow(rimgR)
disp =  ax3.imshow(disparity_map2-32)
divider = make_axes_locatable(ax3)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(disp, cax=cax, orientation='vertical')

ax4.imshow(depths)
depth =  ax4.imshow(depths, vmin= 0, vmax = 5000)
divider = make_axes_locatable(ax4)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(depth, cax=cax, orientation='vertical')

real  = cv.imread('z_depth.tif', cv.IMREAD_UNCHANGED)
realr = cv.remap(real, mapLx, mapLy,
                        interpolation=cv.INTER_NEAREST,
                        borderMode=cv.BORDER_CONSTANT,
                        borderValue=(0, 0, 0, 0))

rl = ax5.imshow(realr, vmin= 0, vmax = 5000)
divider = make_axes_locatable(ax5)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(rl, cax=cax, orientation='vertical')

difference= realr-depths
dif = ax6.imshow(difference, cmap = 'coolwarm', vmin =-7000, vmax = 7000 )
divider = make_axes_locatable(ax6)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(dif, cax=cax, orientation='vertical')

ax1.set_title('Left')
ax2.set_title('Right')
ax3.set_title('Disparity')
ax4.set_title('Depth')
ax5.set_title('True Depth')
ax6.set_title('Difference')
plt.show()









print('done')