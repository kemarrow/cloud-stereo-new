'''Estimate the parameters of the stereo system'''
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
from bearings import rotation_matrix, translation_vector, cleft_elevation, cright_elevation, cleft_bearing, cright_bearing
from mask2 import mask_C1, mask_C3
#from calib_rectify import CamM_1, Distort_1

# camera 3 is on the left, camera 1 is on the right

CamM_1 = [5.115201885073643666e+02,0,3.069700637546008579e+02, 0,5.110005291739051358e+02,2.441830592759120862e+02, 0,0,1]
CamM_1 = np.array(CamM_1).reshape(3,3)
CamM_2 = [5.063166013857816665e+02,0,3.166661931952413056e+02, 1,5.067357512519903935e+02,2.526423174030390157e+02, 0,0,1]
CamM_2 = np.array(CamM_2).reshape(3,3)
CamM_3 = [5.115201885073643666e+02,0,3.069700637546008579e+02, 0,5.110005291739051358e+02,2.441830592759120862e+02, 0,0,1]
CamM_3 = np.array(CamM_1).reshape(3,3)


Distort_1 = np.array([2.791793909906162274e-01,-7.229131626061370275e-01,3.998341915440934737e-03,2.866146329555672653e-03,6.224929102783540724e-01])
Distort_2 = np.array([2.326755157584974587e-01,-6.011054678561147391e-01,3.963575587693899294e-04,-2.566491984608918874e-04,4.822591716560123420e-01])
Distort_3 = np.array([2.791793909906162274e-01,-7.229131626061370275e-01,3.998341915440934737e-03,2.866146329555672653e-03,6.224929102783540724e-01])

#folder = '/net/seldon/disk1/Users/erg10/timelapse/'
#dtime = '2021-03-11_16'
#dtime = '2021-03-13_14'

w, h = 640, 480 #width and height of image

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
    CamM_1, Distort_1, (w, h), 1, (w, h))
mapx, mapy = cv2.initUndistortRectifyMap(
    CamM_1, Distort_1, None, newcameramtx, (w, h), 5)


good = []
ptsR = []
ptsL = []
#for day in range(11, 16):
   # for hour in range(8, 17, 1):
      #  dtime = '2021-03-{:0>2}_{:0>2}'.format(day, hour)


vidcapR = cv2.VideoCapture(r'Videos/lowres_C1_2021-10-03_12A.mp4')  
vidcapL = cv2.VideoCapture(r'Videos/C3_2021-10-03_12A.mp4')

while(vidcapR.isOpened() and vidcapL.isOpened()):
    success, imgR = vidcapR.read()
    success2, imgL = vidcapL.read()
    if success==True and success2==True:
        rg_ratioR = imgR[:, :, 1]/imgR[:, :, 2]
        cloud_mask_right = mask_C1[:, :, 0] & (rg_ratioR<1.1) & (rg_ratioR>0.93) #combining building and sky mask
        imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        imgR = cv2.remap(imgR, mapx, mapy, cv2.INTER_LINEAR)
        imgR = cv2.bitwise_and(imgR, imgR, mask=cloud_mask_right)  #adding mask on img1

        rg_ratioL = imgL[:, :, 1]/imgL[:, :, 2]
        cloud_mask_left = mask_C3[:, :, 0] & (rg_ratioL<1.1) & (rg_ratioL>0.93)  #combining building and sky mask
        imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        imgL = cv2.remap(imgL, mapx, mapy, cv2.INTER_LINEAR)
        imgL = cv2.bitwise_and(imgL, imgL, mask=cloud_mask_left) #adding mask on camera 3 image

        sift = cv2.SIFT_create(contrastThreshold=0.02)
        kpL, desL = sift.detectAndCompute(imgL, None)
        kpR, desR = sift.detectAndCompute(imgR, None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(desL,desR,k=2)
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                good.append(m)
                ptsL.append(kpL[m.queryIdx].pt)
                ptsR.append(kpR[m.trainIdx].pt)
        #fig, (ax1,ax2) = plt.subplots(1,2, sharex=True, sharey = True)
        #cv2.imshow('frame C1', img_right)
        #cv2.imshow( 'frame C2', img_left)
        #plt.show()
        #cv2.imshow('frame',img_left)              # show the video
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #break
    else:
        break
vidcapR.release()
vidcapL.release()
cv2.destroyAllWindows()


#fig, ((ax1,ax2), (ax3, ax4)) = plt.subplots(2,2, sharex=True, sharey = True)
#ax1.imshow(imgR)
#ax3.imshow(img_right,'gray')
#ax3.set_xlabel('Camera 1')
#ax2.imshow(img_left1)
#ax4.imshow(img_left,'gray')
#ax4.set_xlabel('Camera 3')
#plt.show()

#fig, (ax1, ax2) = plt.subplots(1,2, sharex=True, sharey = True)
#imgSift = cv2.drawKeypoints(
    #img_left, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#plt.imshow( imgSift)
#plt.show()

#for i,(m,n) in enumerate(matches):
   # if m.distance < 0.7*n.distance:
       # good.append(m)
       # pts2.append(kp2[m.trainIdx].pt)
       # pts1.append(kp1[m.queryIdx].pt)

ptsL = np.int32(ptsL)
ptsR = np.int32(ptsR)
fundamental_matrix, inlier_mask = cv2.findFundamentalMat(ptsL, ptsR, cv2.FM_RANSAC)
essential_matrix, inlier_mask = cv2.findEssentialMat(ptsL, ptsR, cameraMatrix=newcameramtx)

relDirs = np.rad2deg(cv2.Rodrigues(cv2.decomposeEssentialMat(essential_matrix)[0])[0])

#pts1_norm = cv2.undistortPoints(pts1, cameraMatrix=CamM_1, distCoeffs=Distort_1)
#pts2_norm = cv2.undistortPoints(pts2, cameraMatrix=CamM_1, distCoeffs=Distort_1)

pose = cv2.recoverPose(essential_matrix, ptsL , ptsR, newcameramtx)
relDirs2 = np.rad2deg(cv2.Rodrigues(pose[1])[0])
# Euler angles are roll, pitch, yaw

inter_cam_dist = np.sqrt(np.sum(translation_vector**2))

Rleft, Rright, Pleft, Pright, Q, roi_left, roi_right = cv2.stereoRectify(
    newcameramtx, None,
    newcameramtx, None,
    imageSize=(w,h),
    R=pose[1],
    T=pose[2]*inter_cam_dist)

success, Hleft, Hright = cv2.stereoRectifyUncalibrated(
    np.float32(ptsL), np.float32(ptsR),
    fundamental_matrix, imgSize=(w, h)
    )
T = pose[2]*inter_cam_dist
stereo_cal = {'F': fundamental_matrix,
              'E': essential_matrix,
              'R': pose[1],
              'T': T,
              'Rleft': Rleft,
              'Rright': Rright,
              'Pleft': Pleft,
              'Pright': Pright,
              'Q': Q,
              'Hleft': Hleft,
              'Hright': Hright,
              'roi_left': roi_left,
              'roi_right': roi_right}

#with open('stereo_cal_mat.pkl', 'wb') as f:
    #pickle.dump(stereo_cal, f)



np.savetxt(r'matrices/fundamental_matrix.csv', fundamental_matrix, delimiter=',')
np.savetxt(r'matrices/essential_matrix.csv', essential_matrix, delimiter=',')
np.savetxt(r'matrices/pose[1].csv', pose[1], delimiter=',')
np.savetxt(r'matrices/T.csv', T, delimiter=',')
np.savetxt(r'matrices/Rleft.csv', Rleft, delimiter=',')
np.savetxt(r'matrices/Rright.csv', Rright, delimiter=',')
np.savetxt(r'matrices/Pleft.csv', Pleft, delimiter=',')
np.savetxt(r'matrices/Pright.csv', Pright, delimiter=',')
np.savetxt(r'matrices/Q.csv', Q, delimiter=',')
np.savetxt(r'matrices/Hleft.csv', Hleft, delimiter=',')
np.savetxt(r'matrices/Hright.csv', Hright, delimiter=',')
np.savetxt(r'matrices/roi_left.csv', roi_left, delimiter=',')
np.savetxt(r'matrices/roi_right.csv', roi_right, delimiter=',')

print('F:', fundamental_matrix,
              'E:', essential_matrix,
              'R:', pose[1],
              'T:',  pose[2]*inter_cam_dist,
              'Rleft:', Rleft,
              'Rright:', Rright,
              'Pleft:', Pleft,
              'Pright:', Pright,
              'Q:', Q,
              'Hleft:', Hleft,
              'Hright:', Hright,
              'roi_left:', roi_left,
              'roi_right:', roi_right)
#camera 1 is right
