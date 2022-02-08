"""
finds optical flow and depth map for each frame

@author: Erin
"""

import numpy as np
import cv2 as cv
from bearings import rotation_matrix, translation_vector, baseline_dist
from mask2 import mask_C1, mask_C3
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.image import NonUniformImage

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

    #camera field of views
    
horizontal_fov = 62.2 #degrees
vertical_fov = 48.8 #degrees

theta_horizontal = horizontal_fov/w #degree/pixel
theta_vertical = vertical_fov/h #degree/pixel

#It is based on Gunner Farneback's algorithm which is explained in "Two-Frame Motion Estimation Based on Polynomial Expansion" by Gunner Farneback in 2003.

def speed_updraft(vidcapR, vidcapL, cap):
    ret, frame1 = cap.read()
    #ret2, img = cap2.read()

    success, imgR = vidcapR.read()
    success2, imgL = vidcapL.read()

    prvsR1 = imgR
    prvsL1 = imgL

    prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    cloud_speed= []
    cloud_height=[]
    cloud_updraft=[]
    count  = 0
    while(1):
        ret, frame2 = cap.read()
        success, imgR = vidcapR.read()
        success2, imgL = vidcapL.read()
        #ret2, img2 = cap2.read()
        #img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
        if not ret:
            print('No frames grabbed!')
            break
        next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        #print(flow)
        print(count)
        count +=1
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        bgr = cv.bitwise_and(bgr, bgr, mask=mask_C3)
        bgr = cv.remap(bgr, mapLx, mapLy,
                                interpolation=cv.INTER_NEAREST,
                                borderMode=cv.BORDER_CONSTANT,
                                borderValue=(0, 0, 0, 0))
        
        imgR = cv.bitwise_and(imgR, imgR, mask=mask_C1[:, :, 0])
        imgL = cv.bitwise_and(imgL, imgL, mask=mask_C3[:, :, 0])

        prvsR1  = cv.bitwise_and(prvsR1, prvsR1, mask=mask_C1[:, :, 0])
        prvsL1  = cv.bitwise_and(prvsL1, prvsL1, mask=mask_C3[:, :, 0])

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
            #previous image mapping
        rimgR1 = cv.remap(prvsR1 , mapRx, mapRy,
                                interpolation=cv.INTER_NEAREST,
                                borderMode=cv.BORDER_CONSTANT,
                                borderValue=(0, 0, 0, 0))
        rimgL1 = cv.remap(prvsL1 , mapLx, mapLy,
                                interpolation=cv.INTER_NEAREST,
                                borderMode=cv.BORDER_CONSTANT,
                                borderValue=(0, 0, 0, 0))
                    
            #combine mask1 and mask3 and rectify 
        rg_ratio = imgL[:, :, 1]/imgL[:, :, 2]
        new_mask =  mask_C1[:,:,0] & mask_C3[:,:,0] & (rg_ratio<1.1) & (rg_ratio>0.93)

        rg_ratio1 = prvsL1[:, :, 1]/prvsL1[:, :, 2]
        new_mask1 =  mask_C1[:,:,0] & mask_C3[:,:,0] & (rg_ratio1<1.1) & (rg_ratio1>0.93)

        disp_mask = cv.remap(new_mask, mapLx, mapLy,
                                interpolation=cv.INTER_NEAREST,
                                borderMode=cv.BORDER_CONSTANT,
                                borderValue=(0, 0, 0, 0))
        
        disp_mask1 = cv.remap(new_mask1, mapLx, mapLy,
                                interpolation=cv.INTER_NEAREST,
                                borderMode=cv.BORDER_CONSTANT,
                                borderValue=(0, 0, 0, 0))

            #compute disparity map for the rectified images
        disparity_map2 = stereo.compute(rimgL, rimgR).astype(np.float32)
        disparity_map1 = stereo.compute(rimgL1, rimgR1).astype(np.float32)

            #convert to depth
        im3d = cv.reprojectImageTo3D(disparity_map2/32, Q, handleMissingValues = True)
        im3d1 = cv.reprojectImageTo3D(disparity_map1/32, Q, handleMissingValues = True)

            #set out of range depths to 0
        #im3d[im3d == np.inf] = None
        im3d[im3d == -np.inf] = None
        im3d[im3d > 9000] = None
        #im3d[im3d == 0]  = None

        #im3d1[im3d1 == np.inf] = None
        im3d1[im3d1 == -np.inf] = None
        im3d1[im3d1 > 9000] = None

        tilt = 23 * np.pi/180
        height_camera = 46

        depths = np.sqrt(im3d[:,:,0]**2 + im3d[:,:,1]**2 + im3d[:,:,2]**2)
        depths = cv.bitwise_and(depths, depths, mask=disp_mask)
        depths[depths > 9000] = None
        angle  = np.arccos(im3d[:,:,2]/depths)
        z = depths * np.sin(angle + tilt) + height_camera
        z = cv.bitwise_and(z, z, mask=disp_mask)

        depths1 = np.sqrt(im3d1[:,:,0]**2 + im3d1[:,:,1]**2 + im3d1[:,:,2]**2)
        depths1 = cv.bitwise_and(depths1, depths1, mask=disp_mask1)
        depths1[depths1 > 9000] = None
        angle1  = np.arccos(im3d1[:,:,2]/depths1)
        z1 = depths1 * np.sin(angle1 + tilt) + height_camera
        z1 = cv.bitwise_and(z1, z1, mask=disp_mask1)

        #change in depth between 2 frames
        delta_depths =[]
        for  r, valuex in enumerate(depths):
            for  i, value in enumerate(valuex):
                #print(flow[r,i,0])
                x = r+round(flow[r,i,0])
                y = i+round(flow[r,i,1])
                if x<0 or y<0 or x> 479 or y> 639:
                    delta_depth = 50000#depths[r,i] - depths1[r,i]
                    delta_depths.append(delta_depth)
                else:
                    delta_depth = depths[x, y] - depths1[r,i]
                    delta_depths.append(delta_depth)
        delta_depths=np.reshape(delta_depths,(480,640))

        #change in height between 2 frames
        delta_heights = []
        for  p, valuehx in enumerate(z):
            for q, valueh in enumerate(valuehx):
                #print(flow[r,i,0])
                x = p+round(flow[p,q,0])
                y = q+round(flow[p,q,1])
                if x<0 or y<0 or x> 479 or y> 639:
                    delta_height = 50000#depths[r,i] - depths1[r,i]
                    delta_heights.append(delta_height)
                else:
                    delta_height = z[x, y] - z1[p,q]
                    delta_heights.append(delta_height)
        delta_heights=np.reshape(delta_heights,(480,640))
        
            #set non physical changes to none
        delta_heights[delta_heights==0] = None
        #delta_depths[delta_depths==0] = None
        delta_heights[abs(delta_heights)>1000] = None
        delta_depths[abs(delta_depths)>1000] = None

        depths[depths == 0]  = None
        depths1[depths1 == 0]  = None
        
        #convert pixel direction to m
        ang_horizontal = flow[:,:,0]* theta_horizontal *np.pi/180 #rad
        ang_vertical = flow[:,:,1]* theta_vertical *np.pi/180 #rad

        u  = np.tan(ang_horizontal/2) * depths 
        v = np.tan(ang_vertical/2) * depths

        speed = np.sqrt(u**2 + v**2 + delta_depths**2) *1/5 #m/s
        #speed1D = np.sqrt(u**2 + v**2) * 1/5 #m/s

        updraft  = delta_heights* 1/5

        cloud_height.append(z)
        cloud_speed.append(speed)
        cloud_updraft.append(updraft)

            #plot
        #heat map  of change in height vs height
        #updraft_flat = updraft.flatten()
        #xh = z.flatten()
        #idxh = (~np.isnan(xh+updraft_flat))
        
        #fig7, (ax6, ax7) = plt.subplots(1,2)
        #Hh, xedgesh, yedgesh = np.histogram2d(xh[idxh], updraft_flat[idxh], bins=(100, 100))
        #extenth = [xedgesh[0], xedgesh[-1], yedgesh[0], yedgesh[-1]]
        #upd = ax7.imshow(Hh.T, extent=extenth, interpolation='nearest', origin='lower', cmap='coolwarm', aspect='auto')
        #dividerh = make_axes_locatable(ax7)
        #cax7 = dividerh.append_axes('right', size='5%', pad=0.05)
        #fig7.colorbar(upd, cax=cax7, orientation='vertical')

        #ax7.set_xlabel('Height (m)')
        #ax7.set_ylabel('Updraft (m/s)')
        #ax7.set_title('histogram2d')

        #heat map  of speed vs height
        #x = z.flatten()
        #y = speed.flatten()
        #print(x.shape)
        #idx = (~np.isnan(x+y))
        #fig6, ax6 = plt.subplots(1,1)
        #H, xedges, yedges = np.histogram2d(x[idx], y[idx], bins=(100, 100))# range=[[1000, 9000], [0,200]] )
        #extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        #spd = ax6.imshow(H.T, extent=extent, interpolation='nearest', origin='lower', cmap='coolwarm', aspect='auto')
        #divider = make_axes_locatable(ax6)
        #cax6 = divider.append_axes('right', size='5%', pad=0.05)
        #fig7.colorbar(spd, cax=cax6, orientation='vertical')

        #ax6.set_xlabel('Height (m)')
        #ax6.set_ylabel('Speed (m/s)')
        #ax6.set_title('histogram2d')
        #ax6.grid()
        #plt.show()

        #cv.imshow('frame2', bgr)
        #cv.imshow('frame1',frame2)
        #cv.imshow('depths',delta_depths)
        #mag[mag == 0]  = None

        
        #plot the optical flow on the image
        #fig2, ax2 = plt.subplots(1,1)
        #ax2.imshow(disparity_map2) 
        #x,y = np.meshgrid(np.linspace(0,640,640),np.linspace(0,480,480))
        #ax2.quiver(x, y, u, v, color  = 'red')
        #plt.show()
            

        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        #elif k == ord('s'):
            #cv.imwrite('opticalfb.png', frame2)
            #cv.imwrite('opticalhsv.png', bgr)
        prvs = next
        prvsR1 = imgR
        prvsL1 = imgL
    cv.destroyAllWindows()
    cloud_height  = np.reshape(cloud_height, (480*640*716, 1))
    cloud_speed  = np.reshape(cloud_speed, (480*640*716, 1))
    cloud_updraft  = np.reshape(cloud_updraft, (480*640*716, 1))

    np.savetxt(r'data/cloud_height_{videocapR}.csv', cloud_height , delimiter=',', fmt='%s')
    np.savetxt(r'data/cloud_speed_{videocapR}.csv', cloud_speed , delimiter=',', fmt='%s')
    np.savetxt(r'data/cloud_updraft_{videocapR}.csv', cloud_updraft , delimiter=',', fmt='%s')

    return 'dooone'

#cloud_height  = np.reshape(cloud_height, (480*640*716, 1))
#cloud_speed  = np.reshape(cloud_speed, (480*640*716, 1))
#cloud_updraft  = np.reshape(cloud_updraft, (480*640*716, 1))


vidcapR = cv.VideoCapture(r'Videos/lowres_C1_2021-10-20_11A.mp4')
vidcapL = cv.VideoCapture(r'Videos/C3_2021-10-20_11A.mp4')
cap = vidcapL

speed_updraft(vidcapR, vidcapL, cap)





print('done')