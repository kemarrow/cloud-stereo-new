"""
finds optical flow and depth map for each frame

@author: Erin
"""

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
#backSub = cv.createBackgroundSubtractorKNN()
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
    im3d = cv.reprojectImageTo3D((disparity_map2-32)/16, Q, handleMissingValues = True)
    im3d1 = cv.reprojectImageTo3D((disparity_map1-32)/16, Q, handleMissingValues = True)

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

    tilt = 23.7 * np.pi/180
    height_camera = 46

    depths = np.sqrt(im3d[:,:,0]**2 + im3d[:,:,1]**2 + im3d[:,:,2]**2)
    depths = cv.bitwise_and(depths, depths, mask=disp_mask)
    depths[depths > 5000] = None
    angle  = np.arccos(im3d[:,:,2]/depths)
    z = depths * np.sin(angle + tilt) + height_camera
    z = cv.bitwise_and(z, z, mask=disp_mask)
    z[z==disp_mask]=None
    

    depths1 = np.sqrt(im3d1[:,:,0]**2 + im3d1[:,:,1]**2 + im3d1[:,:,2]**2)
    depths1 = cv.bitwise_and(depths1, depths1, mask=disp_mask1)
    depths1[depths1 > 5000] = None
    angle1  = np.arccos(im3d1[:,:,2]/depths1)
    z1 = depths1 * np.sin(angle1 + tilt) + height_camera
    z1 = cv.bitwise_and(z1, z1, mask=disp_mask1)
    z1[z1==disp_mask1]=None
    #change in depth between 2 frames
    delta_depths =[]
    for  r, valuex in enumerate(depths1):
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
    for  p, valuehx in enumerate(z1):
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
    speed[speed == disp_mask1] = None
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

    #heat map  of change in height vs height
    # updraft_flat = updraft.flatten()
    # xh = z.flatten()
    # idxh = (~np.isnan(xh+updraft_flat))
    
    # fig7, (ax6, ax7) = plt.subplots(1,2)
    # Hh, xedgesh, yedgesh = np.histogram2d(xh[idxh], updraft_flat[idxh], bins=(100, 100))
    # extenth = [xedgesh[0], xedgesh[-1], yedgesh[0], yedgesh[-1]]
    # upd = ax7.imshow(Hh.T, extent=extenth, interpolation='nearest', origin='lower', cmap='coolwarm', aspect='auto')
    # dividerh = make_axes_locatable(ax7)
    # cax7 = dividerh.append_axes('right', size='5%', pad=0.05)
    # fig7.colorbar(upd, cax=cax7, orientation='vertical')

    # ax7.set_xlabel('Height (m)')
    # ax7.set_ylabel('Updraft (m/s)')
    # #ax7.set_title('histogram2d')

    # #heat map  of speed vs height
    # x = z.flatten()
    # y = speed.flatten()
    # #print(x.shape)
    # idx = (~np.isnan(x+y))
    # #fig6, ax6 = plt.subplots(1,1)
    # H, xedges, yedges = np.histogram2d(x[idx], y[idx], bins=(100, 100), range=[[500, 9000], [0,20]] )
    # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    # print(extent)
    # print(np.shape(extent))

    # print('shape', np.shape(H.T))
    # spd = ax6.imshow(H.T, extent=extent, interpolation='nearest', origin='lower', cmap='coolwarm', aspect='auto')
    
    # divider = make_axes_locatable(ax6)
    # cax6 = divider.append_axes('right', size='5%', pad=0.05)
    
    # fig7.colorbar(spd, cax=cax6, orientation='vertical')
    # #ax6.plot(6.687242798, 6.687242798)
    # #ax6.axvline(x=1280.09, color='black', linestyle='-')
    # #ax6.axvline(x=1066.74, color='black', linestyle='-')
    # #ax6.axhline(y=6.687242798, color='black', linestyle='-')
    # ax6.set_xlabel('Height (m)')
    # ax6.set_ylabel('Speed (m/s)')
    # #ax6.set_title('histogram2d')
    # #ax6.grid()

    # spd = y[idx]
    # alt = x[idx]
    # spd[spd>20] = None
    # idx3 = (~np.isnan(alt+spd))


    # numbins = 100
    
    # df = pd.DataFrame({'altitude': alt[idx3], 'speed':spd[idx3]})
    # bins = pd.cut(df['altitude'], numbins)
    
    # altitude = df.groupby(bins)['speed'].median()
    # count  = df.groupby(bins)['altitude'].count().rename('Count')
    # df2 = pd.concat([altitude, count], axis=1)
    
    # df2.loc[df2.Count < 400, 'speed']  = None
    # print('Im here')
    # print(df2.head())
    # df3 = df2
    # #len(df3.iloc[:,0])
    # minbin = np.nanmin(alt[idx3])
    # maxbin= np.nanmax(alt[idx3])

    # sizeofbin = (maxbin-minbin)/numbins
    # print(minbin)
    # print(maxbin)
    # #minalt = np.nanmin(df3.iloc[:,0])
    # #print(minalt)
    # minalt = min(alt[idx3])
    # alt_bins = []
    # for i in range(0, len(df3.iloc[:,0]), 1):
    #     p = minalt + (2*i+1)*sizeofbin/2
    #     alt_bins.append(p)
    # #print(df2.count())
    # #df2['count_of'] = df.groupby(bins)['speed'].count()
    # #df2 = df2.mean()  #.mean()
    # #df3.plot(y='speed')
    # ax6.plot(alt_bins,df3.iloc[:,0], color='k')

    # plt.show()
    # quit()

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
    #elif k == ord('s'):
        #cv.imwrite('opticalfb.png', frame2)
        #cv.imwrite('opticalhsv.png', bgr)
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

#with open('data/speeds_'+date+'.pkl', 'wb') as f:
    #pickle.dump(data, f)

#np.savetxt('data/cloud_height_'+date+'.csv', cloud_height , delimiter=',', fmt='%s')
#np.savetxt('data/cloud_speed_'+date+'.csv', cloud_speed , delimiter=',', fmt='%s')
#np.savetxt('data/cloud_updraft_'+date+'.csv', cloud_updraft , delimiter=',', fmt='%s')


print('done')