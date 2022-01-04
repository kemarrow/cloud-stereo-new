import numpy as np
import matplotlib.path as mpltPath
import cv2 as cv
#import csat2.misc.geo
from scipy.spatial.transform import Rotation as R
import pickle
from matplotlib import pyplot as plt

#note: different resolutions have different coordinate systems...

#timestamp location for 1280x960
mask_points_time1 = np.array([[587.5, 19.8],
                             [587.5, 32.2],
                             [689.8, 32.2],
                             [689.8, 19.8]])

#timestamp location for 640x480
mask_points_time3 = np.array([[267, 32],
                             [267, 20],
                             [372, 20],
                             [372, 32],
                             [272, 32]])

#mask for buildings for camera 1, 1280x960 --> need to divde coordinates by 2 to get the 640x480 coordinates
img1_mask = np.array([[  2.78 , 869.98],
                             [ 3.40, 863.84],
                             [ 5.93, 863.07],
                             [ 6.7, 855.32],
                             [ 7.77,854.01],
                             [ 16.60, 853.17],
                             [ 41.55, 815.55],
                             [ 54.83, 805.65],
                             [ 56.14, 822.31],
                             [ 58.44, 840.66],
                             [ 64.28, 881.04],
                             [ 67.65, 881.81],
                             [ 73.18, 875.2],
                             [ 75.1 , 889.17],
                             [ 79.55 , 890.94],
                             [ 81.01, 894.47],
                             [90.38, 895.24],
                             [ 91.76, 907.29],
                             [96.44, 907.52 ],
                             [102.28, 899.62],
                             [110.18, 881.34],
                             [111.8, 882.8 ],
                             [115.17, 906.45],
                             [167.92, 906.68],
                             [179.92, 907.29],
                             [181.43, 902.61],
                             [185.03, 904.38],
                             [185.11, 910.36],
                             [191.64, 910.44 ],
                             [206.61, 889.41],
                             [214.74, 889.94],
                             [231.8, 908.2],
                             [246.5, 906.2],
                             [317.7, 906.4],
                             [319 , 879.3],
                             [328.9, 879.3],
                             [331.5 , 875.2],
                             [339.2 , 873.9],
                             [342.0, 862.4],
                             [355.6, 882.3],
                             [354.1, 912.3],
                             [376.4, 911.9 ],
                             [379.2, 902],
                             [392.1, 903.7],
                             [392.1, 909.2],
                             [415.2, 910.3],
                             [415.9, 918.7],
                             [469.2, 917.4],
                             [470.5, 912.1],
                             [476.5, 911.6],
                             [482.8, 779],
                             [893.2, 779],
                             [910.0, 938.5],
                             [1022.9, 939.0],
                             [1025.2, 952.8],
                             [1055.4, 952.8],
                             [1052.9, 929.3],
                             [1054.8, 929.3],
                             [1059.3, 956.1],
                             [1093.7, 955.9],
                             [1112.1, 957.8],
                             [1119.1, 941.2],
                             [1119.8, 916.2],
                             [1123.8, 916.8],
                             [1143.7, 956.9],
                             [1156.7, 958.2],
                             [1093.3, 624.8],
                             [1130.8, 551.1],
                             [1125.0, 518.1],
                             [1128.2, 478.4],
                             [1154.5, 526],
                             [1161.8, 551.1],
                             [1202.9, 587.3],
                             [1225.0, 593.7],
                             [1276.8, 801.0]])

#use this for the low res image (640x480) (i think)
img1_mask_lowres = img1_mask/2

#mask for buildings for camera 3 
img3_mask = np.array([[  0.0001 , 431.9],
                             [ 7.78, 397.65],
                             [ 15.16, 399.28],
                             [ 6.05, 445.8],
                             [ 27.63,448.1],
                             [ 28.01, 452.52],
                             [ 33.39, 452.99],
                             [ 34.73, 450.11],
                             [ 69.93, 451.65],
                             [ 71.18, 458.27],
                             [ 75.78, 457.31],
                             [ 76.74, 453.53],
                             [ 83.93, 453.47],
                             [ 84.32 , 457.12],
                             [ 89.11 , 457.6],
                             [ 90.07, 452.71],
                             [97.94, 453.09],
                             [ 98.23, 457.02],
                             [101.49, 456.45 ],
                             [101.49, 451.17],
                             [112.23, 452.62],
                             [114.92, 466.04 ],
                             [124.32, 449.93],
                             [124.70, 459.91],
                             [153.76, 459.52],
                             [154.82, 450.02],
                             [155.39, 451.46],
                             [162.2, 462.97],
                             [174.58, 436.5 ],
                             [176.88, 424.79],
                             [186.18, 424.41],
                             [190.21, 452.52],
                             [192.42, 434.29],
                             [196.25, 434.67],
                             [197.12 ,426.71],
                             [214.86, 427.29],
                             [215.63 , 451.17],
                             [220.23 , 439.28],
                             [226.66, 439.37],
                             [230.88, 460.38],
                             [240.66, 460.19],
                             [243.16, 454.53 ],
                             [251.89, 454.15],
                             [256.87, 446.57],
                             [263.97, 454.43],
                             [264.64, 462.68],
                             [268.1, 462.97],
                             [269.44, 456.45],
                             [278.65, 462.78],
                             [278.84, 459.33],
                             [281.24, 457.85],
                             [296.88, 457.6],
                             [296.1, 462.78],
                             [311.07, 463.55],
                             [313.27, 459.9],
                             [341.67, 460.57],
                             [342.34, 447.82],
                             [346.56, 447.91],
                             [346.37, 462.3],
                             [369.0, 460.76],
                             [368.81, 455.1],
                             [375.62, 455.01],
                             [375.81, 462.30],
                             [395.09, 462.11],
                             [395.0, 449],
                             [399.6, 449.35],
                             [399.5, 460.8],
                             [413.03, 462.3],
                             [412.16, 445.61],
                             [416.0, 444.07],
                             [415.04, 426.91],
                             [425.21, 426.04],
                             [427.13, 432.95],
                             [425.21, 426.04],
                             [427.13, 432.37],
                             [430.96, 426.04],
                             [433.46, 432.37],
                             [439.12,431.99],
                             [438.45, 399.95],
                             [443.05, 400.72],
                             [444.58, 430.93],
                             [555.56, 434.1],
                             [556.62, 437.94],
                             [567.74, 437.07],
                             [580.02, 430.26],
                             [599.88,429.40],
                             [604.19, 437.17],
                             [612.44, 437.46],
                             [613.3, 430.74],
                             [612.46, 432.76],
                             [622.03, 438.42],
                             [628.84, 438.61],
                             [636.04, 470.84],
                             [639.3, 470.55]])


def gen_mask(points, imshape, timemask=None):
    # make blank mask
    ind_y, ind_x = np.meshgrid(np.arange(0.5, imshape[1]),
                               np.arange(0.5, imshape[0]))
    # Ensure the points go to the edge of the image
    newpoints = [[641, points[-1, 1]],
                 [641, -1],
                 [-1, -1],
                 [-1, points[0, 1]],
                 [points[0, 0], points[0, 1]]]
    points = np.concatenate((points, newpoints), axis=0)
    points_poly = mpltPath.Path(points)
    mask = points_poly.contains_points(np.array(list(zip(ind_x.ravel(), ind_y.ravel())))).reshape(imshape)

    if timemask is not None:
        points_poly = mpltPath.Path(timemask)
        tmask = points_poly.contains_points(np.array(list(zip(ind_x.ravel(), ind_y.ravel())))).reshape(imshape)
        mask = mask & ~tmask
        
    return mask.transpose()[:, :, None].astype('uint8')

mask_C1 = gen_mask(img1_mask_lowres, (640, 480), timemask=mask_points_time1/2)
mask_C3 = gen_mask(img3_mask, (640, 480), timemask=mask_points_time3)

#img1 = cv.imread(r'Frames/lowres_output_C1_20211004_11/C1_041021_frame_0.jpg')  # camera 1 image
#img3 = cv.imread(r'Frames/output_C3_20211004_11/C3_041021_frame_0.jpg') # camera 3 image

img1 = cv.imread(r'Frames/lowres_C1_frame_0.jpg')
img3 = cv.imread(r'Frames/C3_frame_0.jpg')
#img1 = cv.imread(r"C:\Users\kathe\OneDrive - Imperial College London\MSci Project\lowres_output_C1_20211004_11\C1_041021_frame_8.jpg")
#img3 = cv.imread(r"C:\Users\kathe\OneDrive - Imperial College London\MSci Project\output_C3_20211004_11\C3_041021_frame_8.jpg") # camera 2 image

#print(mask_left)
img1_masked = cv.bitwise_and(img1, img1, mask=mask_C1)
img3_masked = cv.bitwise_and(img3, img3, mask=mask_C3)
#img3_masked = img3+mask_right[0:1]
# fig, ((ax1,ax2,ax3), (ax4, ax5, ax6)) = plt.subplots(2,3) 
# ax1.imshow(img1)
# ax2.imshow(mask_C1, 'gray')
# ax3.imshow(img1_masked)
# ax4.imshow(img3)
# ax5.imshow(mask_C3, 'gray')
# ax6.imshow(img3_masked)
#plt.show()

cv.imwrite("img1_masked.png", img1_masked)
cv.imwrite("img3_masked.png", img3_masked)







#rpi2_camera_matrix = np.array([
    #[7.081227818517781998e+02,0.000000000000000000e+00,3.166872807864161814e+02],
   # [0.000000000000000000e+00,7.076462869624876930e+02,2.362412288395926794e+02],
   # [0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00]])
#rpi2_camera_dist = np.array([
  #  5.802509039934340418e-01,-2.661186729637604387e+00,-2.576286519795654550e-03,4.426548550162914993e-03,4.419816555469236796e+00])

#camera_right = np.array([51.498793613638746, -0.17884208525378673, 30])
#camera_left = np.array([51.49950846148529, -0.17902071837790667, 42])
#queens_tower = np.array([51.49836466489024, -0.17684224462817338, 89])
#eee_right = np.array([51.49886030527626, -0.1764474234277123, 56])
#eee_left = np.array([51.49922293990841, -0.1765219561557504, 56])

#cg_building = np.array([51.49841140556489, -0.174782312358097])
#eee_roof = np.array([51.49915027210576, -0.17648417121867988])

#baseline_bearing = csat2.misc.geo.bearing(camera_left[1], camera_left[0],
                                         # camera_right[1], camera_right[0])
#cright_bearing = csat2.misc.geo.bearing(camera_right[1], camera_right[0],
                                       # cg_building[1], cg_building[0])
#cleft_bearing = csat2.misc.geo.bearing(camera_left[1], camera_left[0],
                            ##           eee_roof[1], eee_roof[0])

# Queens tower top is almost at centre of image (pixel 224)
#qttop_right = 224
#_dheight = (queens_tower[2]-camera_right[2])/1000
#_ddepth = csat2.misc.geo.haversine(camera_right[1], camera_right[0],
                                  # queens_tower[1], queens_tower[0])
##cright_elevation = np.rad2deg(np.arctan2(_dheight, _ddepth)) + (qttop_right-240)*(48.8/480)

#qttop_left = 304
#_dheight = (queens_tower[2]-camera_left[2])/1000
#_ddepth = csat2.misc.geo.haversine(camera_left[1], camera_left[0],
#                                   queens_tower[1], queens_tower[0])
#cleft_elevation = np.rad2deg(np.arctan2(_dheight, _ddepth)) + (qttop_left-240)*(48.8/480)

#translation_vector = np.array([1000*csat2.misc.geo.haversine(
#    camera_left[1], camera_left[0],
#    camera_right[1], camera_right[0]),
 #                              (camera_right[2]-camera_left[2]), 0])

#cright_rotation = R.from_euler('y', (cright_bearing-baseline_bearing+90), degrees=True)*R.from_euler('x', cright_elevation, degrees=True)
#cleft_rotation = R.from_euler('y', (cleft_bearing-baseline_bearing+90), degrees=True)*R.from_euler('x', cleft_elevation, degrees=True)

#rotation_matrix = np.dot(np.linalg.inv(cright_rotation.as_matrix()), cleft_rotation.as_matrix())

##try:
   # with open('stereo_cal_mat.pkl', 'rb') as f:
   #     stereo_cal = pickle.load(f)
#except:
   # print('No Stereo_cal')