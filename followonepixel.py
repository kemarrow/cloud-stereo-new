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
import math

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

#x = 409#204#260
#y = 45#268#133

#x1='409'

#y1='45'

biglist = []
bigcounts =  []

for x in range(240, 244):
    print('changing x')
    p = x
    for y in range(120, 130):
        q=y
        count=0
        heightpix =[]
        counts=[]
        while(1):
            ret, frame2 = cap.read()
            success, imgR = vidcapR.read()
            success2, imgL = vidcapL.read()

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
            # The output is a dict with possibly several keys, but it should always store the optical flow prediction in a key called 'flows'.
            flows = predictions['flows']
            flow  =  flows.detach().numpy()

            #flows will be a 5D tensor BNCHW.
            # This example should print a shape (1, 1, 2, H, W).
            # Create an RGB representation of the flow to show it on the screen
            flow_rgb = flow_utils.flow_to_rgb(flows)
            #print(np.shape(flow_rgb))
            # Make it a numpy array with HWC shape
            flow_rgb = flow_rgb[0, 0].permute(1, 2, 0)
            flow_rgb_npy = flow_rgb.detach().cpu().numpy()

            flownet  = cv.cvtColor(flow_rgb_npy, cv.COLOR_BGR2GRAY)

            flow = flow.squeeze()
            flow  = np.swapaxes(flow, 0, 1)
            flow  = np.swapaxes(flow, 1, 2)

            # OpenCV uses BGR format
            #bgr = cv.cvtColor(flow_rgb_npy, cv.COLOR_RGB2BGR)
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
            counts.append(count)
            print('Count:', count)
            if count ==0:
                x = p
                y = q
                print('Im here')

            count +=1

                #compute disparity map for the rectified images
            disparity_map1 = stereo.compute(rimgL1, rimgR1).astype(np.float32)
            
                #combine mask1 and mask3 and rectify 
            #maskcomb =  mask_C1[:,:,0] & mask_C3[:,:,0]
            #rg_ratio = imgL[:, :, 1]/imgL[:, :, 2]
            #new_mask =  maskcomb & (rg_ratio<1.1) & (rg_ratio>0.93)#maskL & thresh1
            

            # buildmask = cv.remap(maskcomb, mapLx, mapLy,
            #                         interpolation=cv.INTER_NEAREST,
            #                         borderMode=cv.BORDER_CONSTANT,
            #                         borderValue=(0, 0, 0, 0))

            # disp_mask = cv.remap(new_mask, mapLx, mapLy,
            #                         interpolation=cv.INTER_NEAREST,
            #                         borderMode=cv.BORDER_CONSTANT,
            #                         borderValue=(0, 0, 0, 0))
        
                #convert to depth
            im3d = cv.reprojectImageTo3D((disparity_map1-32)/16, Q, handleMissingValues = True)
            
                #set out of range depths to 0
            im3d[im3d == np.inf] = None
            im3d[im3d == -np.inf] = None
            im3d[im3d > 9000] = None
            im3d[im3d == 0]  = None

            tilt = 23 * np.pi/180
            height_camera = 46

            depths = np.sqrt(im3d[:,:,0]**2 + im3d[:,:,1]**2 + im3d[:,:,2]**2)
            #depths = cv.bitwise_and(depths, depths, mask=disp_mask)

            depths[depths > 9000] = None
            angle  = np.arccos(im3d[:,:,2]/depths)
            z = depths * np.sin(angle + tilt) + height_camera
            #z = cv.bitwise_and(z, z, mask=disp_mask)
            #z[z==disp_mask]=None


            heightpix.append(z[y, x])
            if x<112 or y<0 or x> 479 or y> 639 or math.isnan(z[y,x]):
                biglist.append(heightpix)
                bigcounts.append(count)
                print('stop')
                break
            print('before x:',  x, 'before y', y)
            x = x+round(flow[y, x,0])
            y= y+round(flow[y, x,1])

            print('flowx',flow[y, x,0],'flowy',flow[y,x,1])
            print('x:',  x, 'y', y)
            print('depths', depths[y,x])
            print('height',z[y, x])

            
            k = cv.waitKey(30) & 0xff
            if count == 5:
                biglist.append(heightpix)
                bigcounts.append(count)
                break
            
            prvs = next
            prvsR1 = imgR
            prvsL1 = imgL
            k = cv.waitKey(30) & 0xff
            if k == 27:
                break
            else:
                continue
        
        cv.destroyAllWindows()

#cloud_height  = np.reshape(cloud_height, (480*640*716, 1))
#cloud_speed  = np.reshape(cloud_speed, (480*640*716, 1))
#cloud_updraft  = np.reshape(cloud_updraft, (480*640*716, 1))

#np.savetxt('data/cloud_height_'+date+'.csv', cloud_height , delimiter=',', fmt='%s')
#np.savetxt('data/cloud_speed_'+date+'.csv', cloud_speed , delimiter=',', fmt='%s')
#np.savetxt('data/cloud_updraft_'+date+'.csv', cloud_updraft , delimiter=',', fmt='%s')

with open('data/heightpix_'+date+'2.pkl','wb') as f:
    pickle.dump(biglist, f)

with open('data/framecount_'+date+'2.pkl','wb') as f:
    pickle.dump(bigcounts, f)

#np.savetxt('data/height1pix_'+date+'-'+x1+'-'+y1+'.csv',  , delimiter=',', fmt='%s')
print('done')