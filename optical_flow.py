"""
finds optical flow and depth map for each frame

@author: Erin
"""

import numpy as np
import cv2 as cv
from mask2 import mask_C1, mask_C3
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.image import NonUniformImage
import pickle
import ptlflow
from ptlflow.utils import flow_utils
from ptlflow.utils.io_adapter import IOAdapter
import pandas  as pd

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
# win_size = 8
# min_disp = 0
# max_disp = 7
# num_disp = 16*max_disp - 16*min_disp # Needs to be divisible by 16

# #Create Block matching object.
# stereo = cv.StereoSGBM_create(minDisparity= min_disp,
#  numDisparities = num_disp,
#  blockSize = 9,
#  uniquenessRatio = 13,
#  speckleWindowSize = 5,
#  speckleRange = 14,
#  disp12MaxDiff = 7,
#  P1 = 8*3*win_size**2,
#  P2 =32*3*win_size**2)

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
date = '2021-10-24_12A'
# vidcapR = cv.VideoCapture('Videos/lowres_C1_'+ date+'.mp4')
# vidcapL = cv.VideoCapture('Videos/C3_'+ date+'.mp4')
# cap = cv.VideoCapture('Videos/C3_'+date+'.mp4')

prefix_right = 'tl'
prefix_left = 'tl4'
vidfolder = "C:/Users/kathe/OneDrive - Imperial College London/MSci Project/Videos"
dtime = '2021-10-24'
hour = 12
vidcapR = cv.VideoCapture(f'{vidfolder}/{prefix_right}_{dtime}_{hour:0>2}A.mp4')
vidcapL = cv.VideoCapture(f'{vidfolder}/{prefix_left}_{dtime}_{hour:0>2}A.mp4')
cap = cv.VideoCapture(f'{vidfolder}/{prefix_left}_{dtime}_{hour:0>2}A.mp4')

#cap2 = cv.VideoCapture('depth_10_24_12A.mp4')
ret, frame1 = cap.read()
#ret2, img = cap2.read()
backSub = cv.createBackgroundSubtractorKNN()
#backSub = cv.createBackgroundSubtractorMOG2(detectShadows= True) #default is True, not sure which one to choose

# for if you need to resize the camera 1 video to 640x480
success, imgRLarge = vidcapR.read()
imgR = cv.resize(imgRLarge,(640,480),fx=0,fy=0, interpolation = cv.INTER_CUBIC)

# success, imgR = vidcapR.read()

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
count = 0
while(1):
    ret, frame2 = cap.read()
    success, imgRLarge = vidcapR.read()
    imgR = cv.resize(imgRLarge,(640,480),fx=0,fy=0, interpolation = cv.INTER_CUBIC)
    imgRR =imgR
    imgLL = imgL
    success2, imgL = vidcapL.read()
    
    if not ret:
        print('No frames grabbed!')
        break

    #undistort all images
    imgR = cv.undistort(imgR, CamM_right, Distort_right, None, new_camera_matrixright)
    imgL = cv.undistort(imgL, CamM_left, Distort_left, None, new_camera_matrixleft)

    prvsR1 = cv.undistort(prvsR1, CamM_right, Distort_right, None, new_camera_matrixright)
    prvsL1 = cv.undistort(prvsL1, CamM_right, Distort_right, None, new_camera_matrixright)

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

    io_adapter = IOAdapter(model, rimgL1.shape[:2])

    # inputs is a dict {'images': torch.Tensor}
    # The tensor is 5D with a shape BNCHW. In this case, it will have the shape: (1, 2, 3, H, W)
    inputs = io_adapter.prepare_inputs([rimgL1, rimgL])

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
    # bgr = cv.remap(bgr, mapLx, mapLy,
    #                         interpolation=cv.INTER_NEAREST,
    #                         borderMode=cv.BORDER_CONSTANT,
    #                         borderValue=(0, 0, 0, 0))
    
    print(count)
    count +=1
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
    delta_depths =[]
    for  r, valuex in enumerate(depths):
        for  i, value in enumerate(valuex):
            #print(flow[r,i,0])
            y = r+round(flow[r,i,0])
            x = i+round(flow[r,i,1])
            if x<0 or y<0 or x> 639 or y>  479:
                delta_depth = 50000#depths[r,i] - depths1[r,i]
                delta_depths.append(delta_depth)
            else:
                delta_depth = depths[y, x] - depths1[r,i]
                delta_depths.append(delta_depth)
    delta_depths=np.reshape(delta_depths,(480,640))

    #change in height between 2 frames
    delta_heights = []
    for  p, valuehx in enumerate(z):
        for q, valueh in enumerate(valuehx):
            #print(flow[r,i,0])
            y = p+round(flow[p,q,0])
            x = q+round(flow[p,q,1])
            if x<0 or y<0 or x> 639 or y> 479:
                delta_height = 50000#depths[r,i] - depths1[r,i]
                delta_heights.append(delta_height)
            else:
                delta_height = z[y, x] - z1[p,q]
                delta_heights.append(delta_height)
    delta_heights=np.reshape(delta_heights,(480,640))
    
    
    fig, ((ax1,ax2), (ax3,ax4))  = plt.subplots(2,2, sharex = True, sharey = True)
    imgRR  = cv.cvtColor(imgRR, cv.COLOR_RGB2BGR)
    imgLL = cv.cvtColor(imgLL, cv.COLOR_RGB2BGR)
    rimgRR = cv.cvtColor(rimgR, cv.COLOR_RGB2BGR)
    rimgLL = cv.cvtColor(rimgL, cv.COLOR_RGB2BGR)

    ax1.imshow(imgRR)
    ax1.set_xlabel('Camera 1 (right)')
    ax2.imshow(imgLL)
    ax2.set_xlabel('Camera 2 (left)')

    ax3.imshow(rimgRR)
    ax3.set_xlabel('Camera 1 rectified')
    ax4.imshow(rimgLL)
    ax4.set_xlabel('Camera 2 rectified')

    plt.show()

    fig, (ax1,ax2,ax3)  = plt.subplots(1,3, sharex = True, sharey = True)
    
    img = cv.bitwise_and(rimgL, rimgL, mask=disp_mask)
    ax1.imshow(img)
    ax1.set_xlabel('Mask')


    disparity_map22  = cv.bitwise_and(disparity_map2, disparity_map2, mask =  buildmask)
    #disparity_map22[disparity_map22== disp_mask]= None
    ax2.set_xlabel('Disparity Map')
    disp = ax2.imshow(disparity_map22,'coolwarm')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(disp, cax=cax, orientation='vertical')
    depths[depths==0] = None
    depth = ax3.imshow(depths, 'coolwarm')
    ax3.set_xlabel('Depth map')
    divider2 = make_axes_locatable(ax3)
    cax2 = divider2.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(depth, cax=cax2, orientation='vertical')
    
    plt.show()


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

    u  = np.tan(ang_horizontal/2) * depths *2
    v = np.tan(ang_vertical/2) * depths *2

    speed = np.sqrt(u**2 + v**2 + delta_depths**2) *1/5 #m/s
    speed = cv.bitwise_and(speed, speed, mask=disp_mask1)
    speed[speed == disp_mask1 ] = None
    speed[speed>5000] = None
    speed1D = np.sqrt(u**2 + v**2) * 1/5 #m/s

    updraft  = delta_heights* 1/5
    updraft = cv.bitwise_and(updraft, updraft, mask=disp_mask1)
    updraft[updraft == disp_mask1] = None
    #updraft[updraft == 0] = None
    updraft[updraft > 5000] = None
    #speed[speed1D ==0] = 0

    cloud_height.append(z1)
    cloud_speed.append(speed)
    cloud_updraft.append(updraft)

    fig3, ((axx, ax3,ax4), (ax5, ax6, ax7)) = plt.subplots(2,3, sharex =  True, sharey =True)
    masked_left = cv.bitwise_and(rimgL1, rimgL1, mask = buildmask)
    masked_left = cv.cvtColor(masked_left, cv.COLOR_RGB2BGR)
    #imgL  = cv.cvtColor(imgL, cv.COLOR_RGB2BGR)
    axx.imshow(imgL)
    #masked_left = cvt.colo5
    ax3.imshow(masked_left)
    spd = ax3.imshow(speed)
    dividerh = make_axes_locatable(ax3)
    cax3 = dividerh.append_axes('right', size='5%', pad=0.05)
    fig3.colorbar(spd, cax=cax3, orientation='vertical')
    ax3.set_title(date) 
    ax3.set_xlabel('speed')

    #fig4, ax4 = plt.subplots(1,1)
    
    ax4.imshow(masked_left)
    upd = ax4.imshow(updraft, cmap = 'coolwarm')
    dividerh = make_axes_locatable(ax4)
    cax4 = dividerh.append_axes('right', size='5%', pad=0.05)
    fig3.colorbar(upd, cax=cax4, orientation='vertical') 
    ax4.set_xlabel('updraft')
    img = cv.bitwise_and(rimgL, rimgL, mask=disp_mask)
    ax5.imshow(img)
    ax5.set_xlabel('Mask')
    ax6.imshow(bgr)
    ax6.set_xlabel('Flownet')
    fig3.tight_layout(pad=0.5)

    ax7.set_xlabel('Disparity Map')
    disp = ax7.imshow(disparity_map22,'coolwarm')
    divider = make_axes_locatable(ax7)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig3.colorbar(disp, cax=cax, orientation='vertical')
    plt.show()
    
    fig5, ax5 = plt.subplots(1,1)
    speed = speed.flatten()
    n, bins, patches = ax5.hist(speed, 2000, facecolor='g', alpha=0.75)
    plt.xlim(0, 25)

    plt.show()

    
    #plot
    fig, ax = plt.subplots(1,1)
    ax.set_xlabel('heights')
    disp = ax.imshow(z,'coolwarm', vmin = 800, vmax = 8000)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(disp, cax=cax, orientation='vertical')
    file_name = 'heights' + str(count) + '.eps'
    fig.savefig(file_name, format='eps')
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
  
    #heat map  of change in height vs height
    updraft_flat = updraft.flatten()
    xh = z.flatten()
    idxh = (~np.isnan(xh+updraft_flat))
    
    fig7, (ax6, ax7) = plt.subplots(1,2)
    Hh, xedgesh, yedgesh = np.histogram2d(xh[idxh], updraft_flat[idxh], bins=(100, 100))
    extenth = [xedgesh[0], xedgesh[-1], yedgesh[0], yedgesh[-1]]
    upd = ax7.imshow(Hh.T, extent=extenth, interpolation='nearest', origin='lower', cmap='coolwarm', aspect='auto')
    dividerh = make_axes_locatable(ax7)
    cax7 = dividerh.append_axes('right', size='5%', pad=0.05)
    fig7.colorbar(upd, cax=cax7, orientation='vertical')

    ax7.set_xlabel('Height (m)')
    ax7.set_ylabel('Updraft (m/s)')
    #ax7.set_title('histogram2d')

    #heat map  of speed vs height
    x = z.flatten()
    y = speed.flatten()
    #print(x.shape)
    idx = (~np.isnan(x+y))
    #fig6, ax6 = plt.subplots(1,1)
    H, xedges, yedges = np.histogram2d(x[idx], y[idx], bins=(100, 100), range=[[500, 9000], [0,50]] )
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    print(extent)
    print(np.shape(extent))
    
    #sum_col= []
    #for i in H: 
        #appen
        #for j in i:
            #speed = (extent[4]  - extent[3])/100 * j  * H.T[i, j]

    print('shape', np.shape(H.T))
    spd = ax6.imshow(H.T, extent=extent, interpolation='nearest', origin='lower', cmap='coolwarm', aspect='auto')
    
    divider = make_axes_locatable(ax6)
    cax6 = divider.append_axes('right', size='5%', pad=0.05)
    fig7.colorbar(spd, cax=cax6, orientation='vertical')
    #ax6.plot(6.687242798, 6.687242798)
    ax6.set_xlabel('Height (m)')
    ax6.set_ylabel('Speed (m/s)')
    #ax6.set_title('histogram2d')
    #ax6.grid()
    plt.show()

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
cv.destroyAllWindows()

cloud_height  = np.reshape(cloud_height, (480*640*716, 1))
cloud_speed  = np.reshape(cloud_speed, (480*640*716, 1))
cloud_updraft  = np.reshape(cloud_updraft, (480*640*716, 1))

#np.savetxt('data/cloud_height_'+date+'.csv', cloud_height , delimiter=',', fmt='%s')
#np.savetxt('data/cloud_speed_'+date+'.csv', cloud_speed , delimiter=',', fmt='%s')
#np.savetxt('data/cloud_updraft_'+date+'.csv', cloud_updraft , delimiter=',', fmt='%s')


print('done')