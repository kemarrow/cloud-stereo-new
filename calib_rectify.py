import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img1 = cv.imread('img1_masked.png', 0)      #cv.imread(r'frames/frame_1.jpg', 0)  # camera 1 image
img2 = cv.imread(r'Frames/C2_frame_1.jpg', 0) # camera 2 image
img3= cv.imread('img3_masked.png', 0) #img3 = cv.imread(r'frames/C3_frame_1.jpg', 0)

CamM_1 = [5.115201885073643666e+02,0,3.069700637546008579e+02, 0,5.110005291739051358e+02,2.441830592759120862e+02, 0,0,1]
CamM_1 = np.array(CamM_1).reshape(3,3)
CamM_2 = [5.063166013857816665e+02,0,3.166661931952413056e+02, 1,5.067357512519903935e+02,2.526423174030390157e+02, 0,0,1]
CamM_2 = np.array(CamM_2).reshape(3,3)
CamM_3 = [5.115201885073643666e+02,0,3.069700637546008579e+02, 0,5.110005291739051358e+02,2.441830592759120862e+02, 0,0,1]
CamM_3 = np.array(CamM_1).reshape(3,3)


Distort_1 = np.array([2.791793909906162274e-01,-7.229131626061370275e-01,3.998341915440934737e-03,2.866146329555672653e-03,6.224929102783540724e-01])
Distort_2 = np.array([2.326755157584974587e-01,-6.011054678561147391e-01,3.963575587693899294e-04,-2.566491984608918874e-04,4.822591716560123420e-01])
Distort_3 = np.array([2.791793909906162274e-01,-7.229131626061370275e-01,3.998341915440934737e-03,2.866146329555672653e-03,6.224929102783540724e-01])
#assume both images have same height and width
h,w = img1.shape[:2]

def gaussian_contrast(img):
    img = cv.GaussianBlur(img, (7,7),0)
    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    l, a, b = cv.split(img) #Splitting the LAB image to different channels
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(1,1)) #Applying CLAHE to L-channel
    cl = clahe.apply(l)
    limg = cv.merge((cl,a,b)) #Merge the CLAHE enhanced L-channel with the a and b channel
    final = cv.cvtColor(limg, cv.COLOR_BGR2GRAY)
    return final 

img1= gaussian_contrast(img1)
#img2= gaussian_contrast(img2)
img3= gaussian_contrast(img3)

def threshold(img, limit):
    thresh_img = cv.threshold(gaussianL, limit,255, cv.THRESH_TOZERO)[1]
    return thresh_img

#def masking(img):
    #testing in other script

def undistort(img, CamM, Distort, width, height):
    new_cam_matrix, roi = cv.getOptimalNewCameraMatrix(CamM,Distort,(width,height),1,(width,height))
    img_undistorted = cv.undistort(img1, CamM, Distort, None, new_cam_matrix)
    return new_cam_matrix, img_undistorted

new_camera_matrix1, img_1_undistorted= undistort(img1,CamM_1,Distort_1,w,h)
new_camera_matrix2, img_2_undistorted= undistort(img2, CamM_2,Distort_2,w,h)
new_camera_matrix3, img_3_undistorted= undistort(img3, CamM_3,Distort_3,w,h)
#
#img_1_thresh = cv.threshold(img1, 100, 255, cv.THRESH_TOZERO)[1]
#img_2_thresh = cv.threshold(img2, 120, 255, cv.THRESH_TOZERO)[1]
#fig1, (ax4,ax5) = plt.subplots(1,2)
#ax4.imshow(img_1_thresh)
#ax5.imshow(img_2_thresh)
#plt.show()

#Set disparity parameters
#Note: disparity range is tuned according to specific parameters obtained through trial and error. 
win_size = 5
min_disp = -1
max_disp = 63 #min_disp * 9
num_disp = max_disp - min_disp # Needs to be divisible by 16
#Create Block matching object. 
stereo = cv.StereoSGBM_create(minDisparity= min_disp,
 numDisparities = num_disp,
 blockSize = 5,
 uniquenessRatio = 5,
 speckleWindowSize = 5,
 speckleRange = 5,
 disp12MaxDiff = 1,
 P1 = 8*3*win_size**2,#8*3*win_size**2,
 P2 =32*3*win_size**2) #32*3*win_size**2)
#Compute disparity map
#print ("\nComputing the disparity  map...")
#disparity_map = stereo.compute(img_1_undistorted, img_2_undistorted)

# Find Fundamental matrix by calculating image points with SIFT algorithm, 
# then matching with Flann based matcher
# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
kp3, des3 = sift.detectAndCompute(img3, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des3, k=2)

# Keep good matches: calculate distinctive image features
# Lowe, D.G. Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision 60, 91–110 (2004). https://doi.org/10.1023/B:VISI.0000029664.99615.94
# https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
matchesMask = [[0, 0] for i in range(len(matches))]
good = []
pts1 = []
pts2 = []
pts3 = []

for i, (m, n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        # Keep this keypoint pair
        matchesMask[i] = [1, 0]
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)
        pts3.append(kp3[m.queryIdx].pt)
        
# ---------STEREO RECTIFICATION---------

# Calculate the fundamental matrix for the cameras
# https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
pts3 = np.int32(pts3)
fundamental_matrix, inliers = cv.findFundamentalMat(pts1, pts3, cv.FM_RANSAC)

# We select only inlier points
pts1 = pts1[inliers.ravel() == 1]
pts2 = pts2[inliers.ravel() == 1]
pts3 = pts3[inliers.ravel() == 1]

# Stereo rectification (uncalibrated variant)
# Adapted from: https://stackoverflow.com/a/62607343
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]
h3, w3 = img3.shape[:2]
_, H1, H2 = cv.stereoRectifyUncalibrated(
    np.float32(pts1), np.float32(pts3), fundamental_matrix, imgSize=(w1, h1)
)

# Undistort (rectify) the images and save them
# Adapted from: https://stackoverflow.com/a/62607343
img1_rectified = cv.warpPerspective(img1, H1, (w1, h1))
img2_rectified = cv.warpPerspective(img2, H2, (w2, h2))
img3_rectified = cv.warpPerspective(img3, H2, (w3, h3))
cv.imwrite("rectified_1.png", img1_rectified)
cv.imwrite("rectified_2.png", img2_rectified)
cv.imwrite("rectified_3.png", img3_rectified)

disparity_map2 = stereo.compute(img1, img3)
fig, (ax1,ax2,ax3) = plt.subplots(1,3) 
ax1.imshow(disparity_map2,'coolwarm')
ax2.imshow(img1)
ax3.imshow(img3)
plt.show()