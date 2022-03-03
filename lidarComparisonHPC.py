# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 10:51:36 2022

@author: kathe
"""
import numpy as np
import cv2 as cv
import pickle
from tqdm import tqdm
import sys
import datetime
import glob
import os.path
#%%
# functions
def flattenList(t):
    flat_list = []
    for sublist in t:
        for item in sublist:
            flat_list.append(item)
    return flat_list

def singleTransform(coord, mat):
    coord = list(coord)
    coord = np.append(coord, [1])
    val = np.matmul(mat, coord)
    val[0] = val[0]/val[2]
    val[1] = val[1]/val[2]
    val = np.array(val)
    return val

def gaussian_contrast(img):
    img = cv.GaussianBlur(img, (7,7),0)
    #img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    l, a, b = cv.split(img) #Splitting the LAB image to different channels
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(1,1)) #Applying CLAHE to L-channel
    cl = clahe.apply(l)
    limg = cv.merge((cl,a,b)) #Merge the CLAHE enhanced L-channel with the a and b channel
    #final = cv.cvtColor(limg, cv.COLOR_BGR2GRAY)
    return limg

def findCloud(backscatter, dist_thresh=300):
    # np.argmax returns the index of where the backscatter is highest
    # index in this case = range gate i.e. distance
    if np.max(backscatter) > 10:
        # print('building reflection')
        return (None, None)
    cloud = np.argmax(backscatter[dist_thresh:])
    if backscatter[cloud+dist_thresh] - np.median(backscatter[dist_thresh:]) > 0.03: 
        # print((cloud+dist_thresh+20)*3)
        return cloud+dist_thresh, (cloud+dist_thresh+20)*3 #cloud index, cloud distance
    else:
        # print('no clouds')
        return (None, None)
    
def lidarToRectifiedSingle(lidar, H_L, H_C):
    z = np.append(lidar, [1])
    val = np.matmul(H_L, z)
    val[0] = val[0]/val[2]
    val[1] = val[1]/val[2]
    newval = np.array([val[0], val[1], 1])
    rectval = np.matmul(H_C, newval)
    rectval[0] = rectval[0]/rectval[2]
    rectval[1] = rectval[1]/rectval[2]
    return rectval


# get lidar data
class LidarData():
    def __init__(self,
                 fname=None,
                 system_id=None,
                 num_gates=0,
                 gate_length=0,
                 gate_pulse_length=0,
                 pulses_per_ray=0,
                 start_time=None,
                 data=None,
                 data_locs=None):
        self.fname = fname
        self.system_id = system_id
        self.num_gates = num_gates
        self.gate_length = gate_length
        self.gate_pulse_length = gate_pulse_length
        self.pulses_per_ray = pulses_per_ray
        self.start_time = start_time
        self.data = data
        self.data_locs = data_locs

    @classmethod
    def fromfile(cls, filename):
        with open(filename) as f:
            header = [f.readline().split(':', maxsplit=1) for i in range(17)]
            
            fname = header[0][1].strip()
            system_id = header[1][1].strip()
            num_gates = int(header[2][1].strip())
            gate_length = header[3][1].strip()
            gate_pulse_length = header[4][1].strip()
            pulses_per_ray = header[5][1].strip()
            start_time = header[9][1].strip()
            data_locs_format = header[13][0].split(' ')[0].strip()
            data_format = header[15][0].split(' ')[0].strip()

            data = []
            data_locs = []
            while True:
                try:
                    data_locs_in = np.array(f.readline().split()).astype('float')
                    if len(data_locs_in) == 0:
                        break
                    data_locs.append(data_locs_in)
                    data.append(np.array(
                        [f.readline().split() for i in range(num_gates)]).astype('float'))
                except:
                    break
            data = np.array(data)
            data_locs = np.array(data_locs)

            return cls(
                 fname=fname,
                 system_id=system_id,
                 num_gates=num_gates,
                 gate_length=gate_length,
                 gate_pulse_length=gate_pulse_length,
                 pulses_per_ray=pulses_per_ray,
                 start_time=start_time,
                 data=data,
                 data_locs=data_locs)
        
    # starting all these at 20 means we avoid the really large peak at zero distance    
    def getDistance(self):
        return self.data[:,20:,0]*3 #multiply by 3 to get distance in m
    
    def getDoppler(self):
         return self.data[:,20:,1]
    
    def getBackscatter(self):
         return self.data[:,20:,2]
     
    def getBeta(self):
         return self.data[:,20:,3]
     
# homography mapping from rectified to unrectified pixel coords
cameraH = np.array([[ 1.03570118e+00,  2.20209220e-01, -6.56401026e+01],
 [-2.47759214e-01,  9.72997542e-01,  7.11706249e+01],
 [ 1.01353307e-04, -5.38746250e-05,  1.00000000e+00]])

# homography mapping from lidar to rectified pixel coords
lidarH = np.array([[ 1.36071800e+01, -9.17182893e-01, -2.00941892e+03],
       [ 7.76186504e-01, -1.45996552e+01,  5.07045411e+02],
       [ 1.67058286e-03,  7.20348355e-04,  1.00000000e+00]])

# load extrinsic and intrinsic matrices
stereo_cal = pickle.load( open( 'stereo_cal_mat.pkl', "rb" ) )

R_left_absolute = stereo_cal.get('R_left_absolute') # absolute rotation of camera 3 (as calculated by Ronnie)
camera_matrix_left = stereo_cal.get('camera_matrix')
camera_matrix_right = stereo_cal.get('camera_matrix')
distortion_left = stereo_cal.get('distortion')
distortion_right = stereo_cal.get('distortion')

w, h = 640, 480
#%%
### change these to set the time and date you want ###
year = 2021
tasks = []
for doy in range(277,350):
    for hour in range(6,19):
        tasks.append((doy, hour))
task = tasks[int(sys.argv[1])]
#%%
doy, hour = task

def doy_to_date(year, doy):
    '''doy_to_date(year,doy)    Converts a date from DOY representation to day.month.year    returns tuple(year,month,day)    Raises ValueError if the doy is not valid for that year'''    
    dat = datetime.date(year, 1, 1)
    dat += datetime.timedelta(int(doy) - 1)
    if dat.year != year:
        raise ValueError('Day not within year')
    return (dat.year, dat.month, dat.day)

year, month, day = doy_to_date(year, doy)

file_output = f'/rds/general/user/kem3418/home/lidarOutput/{year}-{month}-{day}_{hour}_output.pkl'
if os.path.exists(file_output):
    print(f'{file_output} already exists')
    exit()

# load videos
prefix_right = 'tl'
prefix_left = 'tl4'
vidfolder = '/rds/general/user/erg10/home/CREATE/video/'
dtime = f'{year}-{month}-{day}'
vidcapR = cv.VideoCapture(f'{vidfolder}{dtime}/{prefix_right}_{dtime}_{hour:0>2}A.mp4')
vidcapL = cv.VideoCapture(f'{vidfolder}{dtime}/{prefix_left}_{dtime}_{hour:0>2}A.mp4')

# load corresponding lidar points (could improve time matching?)

# lists to extract from lidar
decimal_time = []
azi = []
elev = []
distance = []
backscatter = []

folder = '/rds/general/user/erg10/home/CREATE/lidar/'
for time in [0,1,2,3,4,5]:
        pattern = f'{folder}{year}{month:0>2}/{year}{month}{day:0>2}/User5_18_{year}{month:0>2}{day:0>2}_{hour:0>2}{time}*.hpl'
        filenames = glob.glob(pattern)
        if len(filenames) != 1:
            continue
        else:
            filename = filenames[0]
        ld = LidarData.fromfile(filenames)
        data_locs = ld.data_locs
        decimal_time.append(list(data_locs[:,0]))
        azi.append(list(data_locs[:,1]))
        elev.append(list(data_locs[:,2]))
        distance.append(list(ld.getDistance()))
        backscatter.append(list(ld.getBackscatter()))

if len(backscatter) == 0:
    print(f'no lidar data for {year}-{month}-{day}_{hour}')
    exit()

decimal_time = np.array(flattenList(decimal_time), ndmin=1)
azi = np.array(flattenList(azi), ndmin=1)
elev = np.array(flattenList(elev), ndmin=1)
distance = np.array(flattenList(distance), ndmin=1)
backscatter = np.array(flattenList(backscatter), ndmin=1)

# Create stereo matching object and set disparity parameters
win_size = 3
min_disp = 0
max_disp = 4
num_disp = 16*max_disp - 16*min_disp # Needs to be divisible by 16
# Create semi global block matching object.
stereo = cv.StereoSGBM_create(minDisparity=min_disp,
 numDisparities = num_disp,
 blockSize = 3,
 uniquenessRatio = 7,
 speckleWindowSize = 75,
 speckleRange = 1,
 disp12MaxDiff = 6,
 P1 = 8*3*win_size**2,
 P2 =32*3*win_size**2)

# video does not start precisely on the hour
camera_time = hour + 10/3600

# output lists 
stereo_x = []
stereo_y = []
stereo_z = []
stereo_coords = []
stereo_depths = []
lidar_depths = []
time_points = []

bar = tqdm(total=717)
while(vidcapR.isOpened() and vidcapL.isOpened()):
    bar.update(1)
    success, imgRLarge = vidcapR.read()
    success2, imgL = vidcapL.read()
    if success==True and success2==True:
        imgR = cv.resize(imgRLarge,(640,480),fx=0,fy=0, interpolation = cv.INTER_CUBIC)
        imgR = gaussian_contrast(imgR)
        imgL = gaussian_contrast(imgL)

        #assume both images have same height and width

        new_camera_matrix_left, roi = cv.getOptimalNewCameraMatrix(camera_matrix_left,distortion_left,(w,h),1,(w,h))
        new_camera_matrix_right, roi = cv.getOptimalNewCameraMatrix(camera_matrix_right, distortion_right,(w,h),1,(w,h))

        #Undistort images
        imgR_undistorted = cv.undistort(imgR, camera_matrix_right, distortion_right, None, new_camera_matrix_right)
        imgL_undistorted = cv.undistort(imgL, camera_matrix_left, distortion_left, None, new_camera_matrix_left)

        #creates new map for each camera with the rotation and pose (R and P) values
        mapLx, mapLy = cv.initUndistortRectifyMap(new_camera_matrix_left, distortion_left, stereo_cal.get('Rleft'), stereo_cal.get('Pleft'), (w,h), cv.CV_32FC1)
        mapRx, mapRy = cv.initUndistortRectifyMap(new_camera_matrix_right, distortion_right, stereo_cal.get('Rright'), stereo_cal.get('Pright'), (w,h), cv.CV_32FC1)

        # remaps each image to the new map
        rimgR = cv.remap(imgR, mapRx, mapRy,
                              interpolation=cv.INTER_NEAREST,
                              borderMode=cv.BORDER_CONSTANT,
                              borderValue=(0, 0, 0, 0))
        rimgL = cv.remap(imgL, mapLx, mapLy,
                              interpolation=cv.INTER_NEAREST,
                              borderMode=cv.BORDER_CONSTANT,
                              borderValue=(0, 0, 0, 0))

        # compute disparity map for the rectified images
        disparity_map = stereo.compute(rimgL, rimgR).astype(np.float32) - 32    # 2px * 16
        # note: -32 accounts for mean zero error of 2px in star rectification
        
        im3d = cv.reprojectImageTo3D(disparity_map/16, stereo_cal.get('Q'), handleMissingValues = True)
        
        # remove areas where matching failed
        im3d[im3d == np.inf] = 0
        im3d[im3d > 9_000] = 0
        im3d[im3d < -9_000] = 0
        im3d[im3d == -np.inf] = 0
        
        # account for camera absolute rotation
        for pos in im3d.reshape(307200,3):
            pos = np.matmul(R_left_absolute, pos)
        im3d.reshape(480,640,3)
        depths = np.sqrt( im3d[:,:,0]**2 + im3d[:,:,1]**2 + im3d[:,:,2]**2)        

        camera_time += 5/3600
        for time in decimal_time:
            if np.abs(time-camera_time) < 2.5/3600:
                i = list(decimal_time).index(time)
                cloudindex, cloud = findCloud(backscatter[i], 500)  
                point = np.array((azi[i], elev[i]))
                new = lidarToRectifiedSingle(point, lidarH, cameraH)
                if cloud is not None:
                    if int(new[0]) < 640 and int(new[1]) < 480:
                        if depths[int(new[1]), int(new[0])] > 1000:
                            if im3d[:,:,2][int(new[1]), int(new[0])] > 0:
                                if im3d[:,:,0][int(new[1]), int(new[0])] > 0:
                                    stereo_depths.append(depths[int(new[1]), int(new[0])])
                                    stereo_x.append(im3d[:,:,0][int(new[1]), int(new[0])])
                                    stereo_y.append(im3d[:,:,1][int(new[1]), int(new[0])])
                                    stereo_z.append(im3d[:,:,2][int(new[1]), int(new[0])])
                                    stereo_coords.append(np.array([new[0], new[1]]))
                                    time_points.append(camera_time) #currently in decimal time
                                    lidar_depths.append(cloud)
                else:
                    pass

        if cv.waitKey(1) & 0xFF == ord('q'):
            break 
    else:
        break
bar.close()  

    # When everything done, release the video capture object
vidcapL.release()
vidcapR.release()

# Closes all the frames
cv.destroyAllWindows()


vid_output = {'stereo_depths': stereo_depths,
              'lidar_depths': lidar_depths,
              'stereo_x': stereo_x,
              'stereo_y': stereo_y,
              'stereo_z': stereo_z,
              'stereo_coords': stereo_coords,
              'time_points': time_points}

with open(file_output, 'wb') as f:
    pickle.dump(vid_output, f)
