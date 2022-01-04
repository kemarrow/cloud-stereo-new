import numpy as np
import cv2 
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
import tqdm
import pyproj
from haversine import haversine
from scipy.spatial.transform import Rotation as R


#fp1 = r"C:\Users\kathe\OneDrive - Imperial College London\MSci Project\lowres_output_C1_20211004_11\\"
#fp2 = r"C:\Users\kathe\OneDrive - Imperial College London\MSci Project\output_C3_20211004_11\\"
#img1 = cv2.imread(fp1+'C1_041021_frame_10.jpg')  #queryimage # camera 1 image
#img2 = cv2.imread(fp2+'C3_041021_frame_10.jpg') #trainimage # camera 2 image

imgR = cv2.imread(r'Frames/frame_0.jpg')  # camera 1 image
imgL = cv2.imread(r'Frames/C3_frame_0.jpg')



camera1 = np.array([51.49880908055068, -0.1788492157810761, 30])
camera2 = np.array([51.49782099271191, -0.17618060849936956, 33])
camera3 = np.array([51.4993318750954, -0.17901837289811393, 46])
queens_tower = np.array([51.49836466489024, -0.17684224462817338, 89])
eee_right = np.array([51.49886030527626, -0.1764474234277123, 56])
eee_left = np.array([51.49922293990841, -0.1765219561557504, 56])

cg_building = np.array([51.49841140556489, -0.174782312358097])
eee_roof = np.array([51.49915027210576, -0.17648417121867988, 56])

# baseline_bearing = csat2.misc.geo.bearing(camera_left[1], camera_left[0],
#                                           camera_right[1], camera_right[0])
# cright_bearing = csat2.misc.geo.bearing(camera_right[1], camera_right[0],
#                                         cg_building[1], cg_building[0])
# cleft_bearing = csat2.misc.geo.bearing(camera_left[1], camera_left[0],
                                       # eee_roof[1], eee_roof[0])
                                       
                                       
geodesic = pyproj.Geod(ellps='WGS84')
baseline_bearing, baseline_inv, baseline_dist = geodesic.inv(camera1[1], camera1[0], 
                                        camera3[1], camera3[0])
cright_bearing, cright_inv, cright_dist = geodesic.inv(camera1[1], camera1[0],
                                        eee_left[1], eee_left[0])
cleft_bearing, cleft_inv, cleft_dist = geodesic.inv(camera3[1], camera3[0],
                                        eee_left[1], eee_left[0])


# Queens tower top is almost at centre of image (pixel 224)
# for now calibrate with top LHS of EEE building which we call eeetop
eeetop_right = 395
_dheight = (eee_roof[2]-camera1[2])/1000
_ddepth = haversine((camera1[1], camera1[0]),
                                   (eee_left[1], eee_left[0]))
cright_elevation = np.rad2deg(np.arctan2(_dheight, _ddepth)) + (eeetop_right-240)*(48.8/480)

eeetop_left = 429
_dheight = (eee_roof[2]-camera3[2])/1000
_ddepth = haversine((camera3[1], camera3[0]),
                                   (eee_left[1], eee_left[0]))
cleft_elevation = np.rad2deg(np.arctan2(_dheight, _ddepth)) + (eeetop_left-240)*(48.8/480)

translation_vector = np.array([1000*haversine(
    (camera3[1], camera3[0]),
    (camera1[1], camera1[0])),
                               (camera1[2]-camera3[2]), 0])

cright_rotation = R.from_euler('y', (cright_bearing-baseline_bearing+90), degrees=True)*R.from_euler('x', cright_elevation, degrees=True)
cleft_rotation = R.from_euler('y', (cleft_bearing-baseline_bearing+90), degrees=True)*R.from_euler('x', cleft_elevation, degrees=True)

rotation_matrix = np.dot(np.linalg.inv(cright_rotation.as_matrix()), cleft_rotation.as_matrix())

print(cleft_elevation)
print(cright_elevation)
print(rotation_matrix)
print(translation_vector)
print(baseline_dist)