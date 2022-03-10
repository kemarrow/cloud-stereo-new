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
import pickle
import ptlflow
from ptlflow.utils import flow_utils
from ptlflow.utils.io_adapter import IOAdapter
import pandas  as pd

#fundamental_matrix = np.loadtxt(r'matrices/fundamental_matrix.csv', delimiter = ',')
##essential_matrix = np.loadtxt(r'matrices/essential_matrix.csv', delimiter = ',')
#pose = np.loadtxt(r'matrices/pose[1].csv', delimiter = ',')
#T = np.loadtxt(r'matrices/T.csv', delimiter = ',')
# Rleft = np.loadtxt(r'matrices/Rleft.csv', delimiter = ',')
# Rright = np.loadtxt(r'matrices/Rright.csv', delimiter = ',')
# Pleft = np.loadtxt(r'matrices/Pleft.csv', delimiter = ',')
# Pright = np.loadtxt(r'matrices/Pright.csv', delimiter = ',')
# Q = np.loadtxt(r'matrices/Q.csv', delimiter = ',')
# Hleft = np.loadtxt(r'matrices/Hleft.csv', delimiter = ',')
# Hright = np.loadtxt(r'matrices/Hright.csv', delimiter = ',')
# roi_left = np.loadtxt(r'matrices/roi_left.csv', delimiter = ',')
# roi_right = np.loadtxt(r'matrices/roi_right.csv', delimiter = ',')

stereo_cal = pickle.load( open( 'stereo_cal_mat.pkl', "rb" ) )

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

model = ptlflow.get_model('flownet2', pretrained_ckpt='things') #other models are here: https://ptlflow.readthedocs.io/en/latest/models/models_list.html

#Set disparity parameters
#Note: disparity range is tuned according to specific parameters obtained through trial and error.
win_size = 8
min_disp = 0
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

#It is based on Gunner Farneback's algorithm which is explained in "Two-Frame Motion Estimation Based on Polynomial Expansion" by Gunner Farneback in 2003.
date = '2021-10-24_12A'
vidcapR = cv.VideoCapture('Videos/lowres_C1_'+ date+'.mp4')
vidcapL = cv.VideoCapture('Videos/C3_'+ date+'.mp4')
cap = cv.VideoCapture('Videos/C3_'+date+'.mp4')

#cap2 = cv.VideoCapture('depth_10_24_12A.mp4')
ret, frame1 = cap.read()
#ret2, img = cap2.read()
backSub = cv.createBackgroundSubtractorKNN()
#backSub = cv.createBackgroundSubtractorMOG2(detectShadows= True) #default is True, not sure which one to choose

success, imgR = vidcapR.read()
success2, imgL = vidcapL.read()

prvsR1 = imgR
prvsL1 = imgL

prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

#camera field of views
horizontal_fov = 62.2 #degrees
vertical_fov = 48.8 #degrees

theta_horizontal = horizontal_fov/w #degree/pixel
theta_vertical = vertical_fov/h #degree/pixel

cloud_speed= []
cloud_height=[]
cloud_updraft=[]
count  = 0
while(1):
    ret, frame2 = cap.read()
    success, imgR = vidcapR.read()
    imgRR =imgR
    imgLL = imgL
    success2, imgL = vidcapL.read()
    #ret2, img2 = cap2.read()
    #img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    if not ret:
        print('No frames grabbed!')
        break

    io_adapter = IOAdapter(model, prvsL1.shape[:2])

    # inputs is a dict {'images': torch.Tensor}
    # The tensor is 5D with a shape BNCHW. In this case, it will have the shape: (1, 2, 3, H, W)
    inputs = io_adapter.prepare_inputs([prvsL1, imgL])

    # Forward the inputs through the model
    predictions = model(inputs)

    # Remove extra padding that may have been added to the inputs
    predictions = io_adapter.unpad_and_unscale(predictions)

    # The output is a dict with possibly several keys,
    # but it should always store the optical flow prediction in a key called 'flows'.
    flows = predictions['flows']
    flow  =  flows.detach().numpy()
   
    #flows will be a 5D tensor BNCHW.
    # This example should print a shape (1, 1, 2, H, W).
    # Create an RGB representation of the flow to show it on the screen
    flow_rgb = flow_utils.flow_to_rgb(flows)
    print(np.shape(flow_rgb))
    # Make it a numpy array with HWC shape
    flow_rgb = flow_rgb[0, 0].permute(1, 2, 0)
    flow_rgb_npy = flow_rgb.detach().cpu().numpy()

    flownet  = cv.cvtColor(flow_rgb_npy, cv.COLOR_BGR2GRAY)

    flow= flow.squeeze()
    flow  = np.swapaxes(flow, 0, 1)
    flow  = np.swapaxes(flow, 1, 2)

    # OpenCV uses BGR format
  
    #rg_ratio = imgL[:, :, 1]/imgL[:, :, 2]
    #new_mask =  mask_C3[:,:,0] & (rg_ratio<1.1) & (rg_ratio>0.93)#maskL & thresh1
    
    #rg_ratio1 = prvsL1[:, :, 1]/prvsL1[:, :, 2]
    #new_mask1 =  maskcomb & (rg_ratio1<1.1) & (rg_ratio1>0.93)

    bgr = cv.cvtColor(flow_rgb_npy, cv.COLOR_RGB2BGR)
    #flow_bgr_npym = cv.bitwise_and(flow_bgr_npy, flow_bgr_npy, mask=new_mask)


    #Farneback algorithm 
    #next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    #flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    #print(flow)
    #mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    #hsv[..., 0] = ang*180/np.pi/2
    #hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    #bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    
    
    #bgr = cv.bitwise_and(bgr, bgr, mask=maskcomb)
    
    #bgr = backSub.apply(z)
    bgr = cv.remap(bgr, mapLx, mapLy,
                            interpolation=cv.INTER_NEAREST,
                            borderMode=cv.BORDER_CONSTANT,
                            borderValue=(0, 0, 0, 0))
    
    print(count)
    count +=1
    #imgR = cv.bitwise_and(imgR, imgR, mask=mask_C1[:, :, 0])
    #imgL = cv.bitwise_and(imgL, imgL, mask=mask_C3[:, :, 0])

    #prvsR1  = cv.bitwise_and(prvsR1, prvsR1, mask=mask_C1[:, :, 0])
    #prvsL1  = cv.bitwise_and(prvsL1, prvsL1, mask=mask_C3[:, :, 0])

    

        #Undistort images
    imgR = cv.undistort(imgR, CamM_right, Distort_right, None, new_camera_matrixright)
    imgL = cv.undistort(imgL, CamM_left, Distort_left, None, new_camera_matrixleft)

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
    disparity_map1 = stereo.compute(rimgL1, rimgR1).astype(np.float32)

         #combine mask1 and mask3 and rectify 
    maskcomb =  mask_C1[:,:,0] & mask_C3[:,:,0]
    rg_ratio = imgL[:, :, 1]/imgL[:, :, 2]
    new_mask =  maskcomb & (rg_ratio<1.1) & (rg_ratio>0.93)#maskL & thresh1
    
    rg_ratio1 = prvsL1[:, :, 1]/prvsL1[:, :, 2]
    new_mask1 =  maskcomb & (rg_ratio1<1.1) & (rg_ratio1>0.93) #maskL1 &thresh1
    

    buildmask = cv.remap(maskcomb, mapLx, mapLy,
                            interpolation=cv.INTER_NEAREST,
                            borderMode=cv.BORDER_CONSTANT,
                            borderValue=(0, 0, 0, 0))

    disp_mask = cv.remap(new_mask, mapLx, mapLy,
                            interpolation=cv.INTER_NEAREST,
                            borderMode=cv.BORDER_CONSTANT,
                            borderValue=(0, 0, 0, 0))
    
    disp_mask1 = cv.remap(new_mask1, mapLx, mapLy,
                            interpolation=cv.INTER_NEAREST,
                            borderMode=cv.BORDER_CONSTANT,
                            borderValue=(0, 0, 0, 0))

    
        #convert to depth
    im3d = cv.reprojectImageTo3D(disparity_map2/16, Q, handleMissingValues = True)
    im3d1 = cv.reprojectImageTo3D(disparity_map1/16, Q, handleMissingValues = True)

    im3d = cv.bitwise_and(im3d, im3d, mask=disp_mask)
    im3d1 = cv.bitwise_and(im3d1, im3d1, mask=disp_mask1)
        #set out of range depths to 0
    im3d[im3d == np.inf] = None
    im3d[im3d == -np.inf] = None
    im3d[im3d > 9000] = None
    #im3d[im3d == 0]  = None

    im3d1[im3d1 == np.inf] = None
    im3d1[im3d1 == -np.inf] = None
    im3d1[im3d1 > 9000] = None

    tilt = 23 * np.pi/180
    height_camera = 46

    depths = np.sqrt(im3d[:,:,0]**2 + im3d[:,:,1]**2 + im3d[:,:,2]**2)
    depths = cv.bitwise_and(depths, depths, mask=disp_mask)
    #depths[depths > 9000] = None
    angle  = np.arccos(im3d[:,:,2]/depths)
    z = depths * np.sin(angle + tilt) + height_camera
    z = cv.bitwise_and(z, z, mask=disp_mask)
    z[z==disp_mask]=None
    

    depths1 = np.sqrt(im3d1[:,:,0]**2 + im3d1[:,:,1]**2 + im3d1[:,:,2]**2)
    depths1 = cv.bitwise_and(depths1, depths1, mask=disp_mask1)
    depths1[depths1 > 9000] = None
    angle1  = np.arccos(im3d1[:,:,2]/depths1)
    z1 = depths1 * np.sin(angle1 + tilt) + height_camera
    z1 = cv.bitwise_and(z1, z1, mask=disp_mask1)
    z1[z1==disp_mask1]=None
    #change in depth between 2 frames
    #creating grids 
    grid1 = np.stack(np.meshgrid(np.arange(480), np.arange( 640))).swapaxes(0, -1)
    grid1= np.round(grid1).astype(int)
    grid = grid1 + flow
    grid = np.round(grid1).astype(int)
    
    #creating empty arrays
    delta_depths = np.empty((480,640))
    delta_depths[:]=np.nan
    delta_heights = np.empty((480,640))
    delta_heights[:]=np.nan

    #filtering pixels which move outside the image frame
    condition = (grid[:,:,1]>639) | (grid[:, :,0]<0) |(grid[:,:, 0]>479)| (grid[:,:, 1]<0)
    condition1 = (grid1[:,:,1]>639) | (grid1[:, :,0]<0) |(grid1[:,:, 0]>479)| (grid1[:,:, 1]<0)
    #change in depth between 2 frames
    delta_depths[~condition] = depths[grid[~condition][:,0], grid[~condition][:,1]] - depths1[grid1[~condition1][:,0], grid1[~condition1][:,1]]    
    delta_heights[~condition] = z[grid[~condition][:,0], grid[~condition][:,1]] - z1[grid1[~condition1][:,0], grid1[~condition1][:,1]]


        #set non physical changes to none
    #delta_heights[delta_heights==0] = None
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
    speed = cv.bitwise_and(speed, speed, mask=disp_mask)
    speed[speed == disp_mask ] = None
    speed[speed>5000] = None
    speed1D = np.sqrt(u**2 + v**2) * 1/5 #m/s

    updraft  = delta_heights* 1/5
    updraft = cv.bitwise_and(updraft, updraft, mask=disp_mask)
    updraft[updraft == disp_mask] = None
    #updraft[updraft == 0] = None
    updraft[updraft > 5000] = None
    #speed[speed1D ==0] = 0

    cloud_height.append(z)
    cloud_speed.append(speed)
    cloud_updraft.append(updraft)


    #plt.show()
    #plot
    fig, (x,ax) = plt.subplots(1,2, sharex= True, sharey= True)
    im = cv.cvtColor(rimgL, cv.COLOR_RGB2BGR)
    x.imshow(im)
    x.set_xlabel('Left camera')

    ax.set_xlabel('Heights (m)')
    disp = ax.imshow(z,'coolwarm', vmin = 800, vmax = 8000)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(disp, cax=cax, orientation='vertical')
    counting = str(count)
    counting.zfill(5)
    file_name = 'gif_plots/heights' + counting + '.png'
    fig.savefig(file_name, format='png')
    #plt.show()

    #fig5, ax5 = plt.subplots(1,1)
    #fig5 = plt.figure()
    #ax5 = fig5.add_subplot()#projection='3d')
    #hist, xedges, yedges = np.histogram2d(x, y, bins=(100, 100), range=[[500, 9000], [0,200]])
    # Construct arrays for the anchor positions of the 16 bars.
    #xpos, ypos = np.meshgrid((xedges[:-1] + xedges[1:])/2, (yedges[:-1] +yedges[1:])/2, indexing="ij")
    #xpos = xpos.ravel()
    #ypos = ypos.ravel()
    #zpos = 
    # Construct arrays with the dimensions for the 16 bars.
    #dx = dy = 0.5 * np.ones_like(zpos)
    #dz = hist.ravel()
    #ax5.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
    #plt.xlabel("Cloud height (m) ")
    ##plt.ylabel("Cloud speed  (m/s) ")
    ###plt.show()
  
    

    #cv.imshow('frame2', bgr)
    #cv.imshow('frame1',frame2)
    #cv.imshow('depths',delta_depths)
    #mag[mag == 0]  = None

    #plot the optical flow on the image
    #fig2, ax2 = plt.subplots(1,1)
    #ax2.imshow(img) 
    #x,y = np.meshgrid(np.linspace(0,640,640),np.linspace(0,480,480))
    #ax2.quiver(x, y, u, v, color  = 'red')
    #plt.show()

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png', frame2)
        cv.imwrite('opticalhsv.png', bgr)
    prvs = next
    prvsR1 = imgR
    prvsL1 = imgL
    if count ==200:
        break
cv.destroyAllWindows()

cloud_height  = np.reshape(cloud_height, (480*640*count, 1))
cloud_speed  = np.reshape(cloud_speed, (480*640*count, 1))
cloud_updraft  = np.reshape(cloud_updraft, (480*640*count, 1))

with open('cloud_height_'+date+'.pkl','wb') as f:
    pickle.dump(cloud_height, f)

with open('cloud_speed_'+date+'.pkl','wb') as f:
    pickle.dump(cloud_speed, f)

with open('cloud_updraft_'+date+'.pkl','wb') as f:
    pickle.dump(cloud_updraft, f)

print('done')