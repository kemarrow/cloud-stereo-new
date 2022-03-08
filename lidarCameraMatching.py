# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 18:22:19 2021

@author: kathe
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle

fp = r"C:\Users\kathe\OneDrive - Imperial College London\MSci Project\output_C1_20211003_12\C1_031021_frame_84.jpg"
imgLarge = cv2.imread(fp) # camera image
img = cv2.resize(imgLarge,(640,480),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
h, w = img.shape[:2]

stereo_cal = pickle.load( open( 'stereo_cal_mat.pkl', "rb" ) )
Rright = stereo_cal.get('Rright')
Pleft = stereo_cal.get('Pright')
R_right_ronnie = np.array([[ 0.9893863, -0.06083473,-0.1319617 ],
 [ 0.14528628 , 0.4302662, 0.89093363],
 [ 0.00257894,-0.9006498, 0.43453792]])


cameraH = np.array([[ 1.03570118e+00,  2.20209220e-01, -6.56401026e+01],
 [-2.47759214e-01,  9.72997542e-01,  7.11706249e+01],
 [ 1.01353307e-04, -5.38746250e-05,  1.00000000e+00]])


lidar_coords = np.array([(206.27500926955878, 21.99513607712273),
 (209.06611049314054, 16.929804226918797),
 (203.2771598071932, 16.516307749351128),
 (208.8593622543567, 14.655573600296623),
 (203.8974045235447, 14.345451242120873),
 (192.00938079347424, 8.349752317389694),
 (170.71431219873935, 9.590241750092694),
 (189.4250278086763, 9.486867630700779),
 (188.39128661475715, 9.486867630700779),
 (192.11275491286614, 2.3540533926585105),
 (170.4041898405636, 3.284420467185763),
 (201.82992213570634, 2.974298109010011),
 (203.17378568780128, 1.940556915090843),
 (164.4084909158324, 3.594542825361513),
 (163.58149796069708, 5.662025213199852),
 (162.65113088616982, 3.38779458657768),
 (157.89592139414165, 4.421535780496848),
 (153.34746014089728, 4.7316581386726),
 (149.93611420096403, 7.522759362254355),
 (151.27997775305897, 4.7316581386726),
 (148.90237300704487, 5.662025213199852),
 (172.67842046718576, 6.902514645902855),
 (197.1780867630701, 2.3540533926585105),
 (206.9986281053022, 15.379192436040043),
 (207.20537634408603, 10.210486466444198),
 (206.9986281053022, 6.799140526510939),
 (184.5664441972562, 9.38349351130886),
 (183.01583240637746, 9.176745272525029),
 (181.77534297367447, 9.280119391916944),
 (197.281460882462, 2.3540533926585105)])

camera_coords = np.array([(568.8506493506495, 251.9675324675324),
  (611.7077922077923, 298.72077922077915),
  (550.6688311688313, 309.1103896103896),
  (613.0064935064936, 322.09740259740255),
  (558.4610389610391, 331.1883116883116),
  (444.17532467532476, 401.31818181818176),
  (241.57792207792212, 393.525974025974),
  (420.7987012987014, 393.525974025974),
  (409.1103896103897, 392.2272727272727),
  (453.2662337662339, 467.551948051948),
  (237.68181818181822, 459.7597402597402),
  (562.357142857143, 461.05844155844153),
  (580.5389610389611, 475.3441558441558),
  (176.6428571428572, 454.564935064935),
  (171.44805194805198, 425.99350649350646),
  (157.1623376623377, 451.96753246753246),
  (105.21428571428575, 444.17532467532465),
  (55.863636363636374, 446.77272727272725),
  (24.694805194805213, 403.91558441558436),
  (31.1883116883117, 441.57792207792204),
  (3.9155844155844335, 427.29220779220776),
  (261.0584415584416, 428.59090909090907),
  (507.8116883116884, 468.8506493506493),
  (592.2272727272729, 316.90259740259734),
  (605.2142857142859, 366.2532467532467),
  (614.3051948051949, 403.91558441558436),
  (375.3441558441559, 396.12337662337654),
  (359.7597402597403, 396.12337662337654),
  (346.77272727272737, 396.12337662337654),
  (506.5129870129871, 468.8506493506493)])

# azi = data_locs[:,1]
# elev = data_locs[:,2]


lidarH, status = cv2.findHomography(lidar_coords, camera_coords, cv2.FM_RANSAC, 16.0)

# array of all of the azimuth and elevation coordinates in the lidar scan
# points = []
# for a in azi:
#     for e in elev:
#         points.append((a,e))
# points = np.array(points)
    
# makes array of the list of found coordinates
lidar_points = []
for a in lidar_coords[:,0]:
    pp = []
    for e in lidar_coords[:,1]:
        pp.append([a,e])
    lidar_points.append(pp)
        
lidar_points = np.array(lidar_points)

# add a z coordinate of 1 to every (x,y) lidar coordinate so that matrix multiplication works
lidarz = []
for point in list(lidar_coords):
    point = np.append(point, [1])
    lidarz.append(point)
    
# remaps the lidar coordinates according to the homography
remapped = []
for p in lidarz:
    val = (np.matmul(lidarH,p))
    val[0] = val[0]/val[2]  #see documentation of cv2.findHomography
    val[1] = val[1]/val[2]
    remapped.append(val)
remapped = np.array(remapped)


# transforms the full array of lidar coordinates to map to the camera image
def transform(coords, mat):
    z = []
    for point in list(coords):
        point = np.append(point, [1])
        z.append(point)
    newmap = []
    for p in z:
        val = np.matmul(mat, p)
        val[0] = val[0]/val[2]
        val[1] = val[1]/val[2]
        newmap.append(val)
    newmap = np.array(newmap)
    return newmap 

    
def lidarToRectifiedSingle(lidar, H_L, H_C):
    z = np.append(lidar, [1])
    val = np.matmul(H_L, z)
    val[0] = val[0]/val[2]
    val[1] = val[1]/val[2]
    newval = np.array([val[0], val[1], 1])
    rectval = np.matmul(H_C, newval)
    # rectval[0] = rectval[0]/rectval[2]
    # rectval[1] = rectval[1]/rectval[2]
    return rectval

#test = transform(points, lidarH)  

rect = []
# for l in lidar_coords:
#     rot = lidarToRectifiedSingle(l, lidarH, Rright) 
#     rot = np.append(rot, [1])
#     rect.append(np.matmul(Pleft, rot))

for c in camera_coords:
    c = np.append(c,1)
    rect.append(np.matmul(Rright, c))
    
rect = np.array(rect)

rectim = cv2.imread(r"C:\Users\kathe\OneDrive - Imperial College London\MSci Project\rectifiedC1forLidarCalibration.png")

### Uncomment to plot things! ###

fig, ax = plt.subplots(1,1)
ax.imshow(rectim)
# ax.plot(camera_coords[:,0], camera_coords[:,1], 'rx')
# ax.plot(remapped[:,0], remapped[:,1], 'bx')
ax.plot(rect[:,0], rect[:,1], 'rx')
plt.show()

