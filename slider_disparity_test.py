import sys
import cv2
#from camera_data import get_stereo_cal, rpi2_camera_dist, rpi2_camera_matrix
from calib_rectify import CamM_1, Distort_1
rpi2_camera_matrix = CamM_1
rpi2_camera_dist = Distort_1
import numpy as np

# 1. Import `QApplication` and all the required widgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QVBoxLayout, QHBoxLayout
from PyQt5.QtWidgets import QLabel, QSlider, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
#from fundamental_estimate import fundamental_matrix, inlier_mask, pts1, pts2,essential_matrix,T, R1, R2, P1, P2, Q, H1, H2, roi_left, roi_right, mapx, mapy


fundamental_matrix = np.loadtxt(r'matrices/fundamental_matrix.csv', delimiter = ',')
essential_matrix = np.loadtxt(r'matrices/essential_matrix.csv', delimiter = ',')
T = np.loadtxt(r'matrices/T.csv', delimiter = ',')
R1 = np.loadtxt(r'matrices/R1.csv', delimiter = ',')
R2 = np.loadtxt(r'matrices/R2.csv', delimiter = ',')
P1 = np.loadtxt(r'matrices/P1.csv', delimiter = ',')
P2 = np.loadtxt(r'matrices/P2.csv', delimiter = ',')
Q = np.loadtxt(r'matrices/Q.csv', delimiter = ',')
H1 = np.loadtxt(r'matrices/H1.csv', delimiter = ',')
H2 = np.loadtxt(r'matrices/H2.csv', delimiter = ',')
roi_left = np.loadtxt(r'matrices/roi_left.csv', delimiter = ',')
roi_right = np.loadtxt(r'matrices/roi_right.csv', delimiter = ',')

#def get_stereo_cal(prefixleft, prefixright):
    #with open(f'stereo_cal_mat.pkl', 'rb') as f:
        #return pickle.load(f)

class DisplayUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Block Matcher Tuner')
        self.setFixedSize(600, 600)
        # Set the central widget
        self.generalLayout = QGridLayout()
        self._centralWidget = QWidget(self)
        self.setCentralWidget(self._centralWidget)
        self._centralWidget.setLayout(self.generalLayout)
        self._createImageDisplay()
        #self._createDisparityDisplay()
        self._createSliders()
                 
    def _createImageDisplay(self):
        self.displayLeft = MplCanvasWidget(self)
        self.generalLayout.addWidget(self.displayLeft, 0, 0)
        self.displayRight = MplCanvasWidget(self)
        self.generalLayout.addWidget(self.displayRight, 0, 1)
        self.displayDisparity = MplCanvasWidget(self)
        self.generalLayout.addWidget(self.displayDisparity, 1, 1)

    def _createSliders(self):
        self._sliderLayout = QVBoxLayout()
        self.sliders = {}
        for name, valrange, default in [['minDisparity', (-32, 32, 1), 0],
                                        ['numDisparities', (0, 256, 16), 64],
                                        ['blockSize', (1, 11, 2), 3],
                                        ['P1', (0, 600, 1), 216],
                                        ['P2', (0, 600, 1), 480],
                                        ['disp12MaxDiff', (-1, 200, 1), 1],
                                        ['preFilterCap', (0, 600, 1), 0],
                                        ['uniquenessRatio', (0, 30, 1), 10],
                                        ['speckleWindowSize', (0, 500, 5), 0],
                                        ['speckleRange', (0, 5, 1), 2]]:
            labelbox = QHBoxLayout()
            label = QLabel(name)
            labelbox.addWidget(label)
            self.sliders[name] = QSlider(Qt.Horizontal)
            self.sliders[name].setMinimum(valrange[0])
            self.sliders[name].setMaximum(valrange[1])
            self.sliders[name].setValue(default)
            self.sliders[name].setSingleStep(valrange[2])
            labelbox.addWidget(self.sliders[name])
            self._sliderLayout.addLayout(labelbox)
        self.generalLayout.addLayout(self._sliderLayout, 1, 0)

    def printSliders(self):
        print(' '.join([str(self.sliders[name].value()) for name in self.sliders.keys()]))
            

    def setDisplayImageLeft(self, data):
        self.displayLeft.axes.imshow(data)
        self.displayLeft.axes.set_xticks([])
        self.displayLeft.axes.set_yticks([])

    def setDisplayImageRight(self, data):
        self.displayRight.axes.imshow(data)
        self.displayRight.axes.set_xticks([])
        self.displayRight.axes.set_yticks([])

    def setDisplayImageDisparity(self, data):
        try:
            self.displayDisparity.axes.set_data(data)
        except:
            self.displayDisparity.axes.imshow(data)
        self.displayDisparity.axes.set_xticks([])
        self.displayDisparity.axes.set_yticks([])
        
        
class MplCanvasWidget(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_axes([0, 0, 1, 1])
        #self.compute_initial_figure()

        super().__init__(fig)
        self.setParent(parent)

        #super().updateGeometry()
        #fig.canvas.mpl_connect('button_press_event', self.button_press_event)

from undistortRectify import rimgR, rimgL, imgR, imgL
#def opencv2_setup():
    #w, h = 640, 480
    
    #mapxl, mapyl = cv2.initUndistortRectifyMap(
        #rpi2_camera_matrix, rpi2_camera_dist,
        #R1,P1, (w, h), 5)
    #mapxr, mapyr = cv2.initUndistortRectifyMap(
        #rpi2_camera_matrix, rpi2_camera_dist,
        #R2, P2, (w, h), 5)
    
    #vidcap = cv2.VideoCapture('lowres_C1_2021-10-03_12A.mp4')
    #vidcap2 = cv2.VideoCapture('C3_2021-10-03_12A.mp4')

    #success, img_right = vidcap.read()
    #rg_ratio = img_right[:, :, 1]/img_right[:, :, 2]
    #cloud_mask_right = mask_right[:, :, 0] & (rg_ratio<1.1) & (rg_ratio>0.95)
    #img_right_rect = (cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY))
    #img_right_rect = cv2.remap(img_right_rect, mapxr, mapyr, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)#[50:200]

    #success, img_left = vidcap2.read()
    #rg_ratio = img_left[:, :, 1]/img_left[:, :, 2]
    #cloud_mask_left = mask_left[:, :, 0] & (rg_ratio<1.1) & (rg_ratio>0.95)
    #img_left_rect = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    #img_left_rect = cv2.remap(img_left_rect, mapxl, mapyl, cv2.INTER_LINEAR)#[50:200]

    #return img_left, img_right, img_left_rect, img_right_rect


class disparityController:
    def __init__(self, model, view):
        self._evaluate = model
        self._view = view
        self._connectSignals()

    def _connectSignals(self):
        for slider_name in self._view.sliders.keys():
            print(slider_name)
            self._view.sliders[slider_name].valueChanged.connect(
                lambda: self.disparity_calc()) #self._view.printSliders())
            #lambda: printVal(self._view.sliders[slider_name].value()))

    def disparity_calc(self):
        params = {name: self._view.sliders[name].value() for name in self._view.sliders.keys()}
        left_matcher = cv2.StereoSGBM_create(**params)
        disparity_left = left_matcher.compute(self._view.img_right_rect,
                                              self._view.img_left_rect)
        print('Disp calculated', np.isfinite(disparity_left).sum(), disparity_left.sum())
        self._view.setDisplayImageDisparity(disparity_left)
        self._view.displayDisparity.draw()
        
def main():
    """Main function."""
    # Create an instance of QApplication
    bmtuner = QApplication(sys.argv)
    # Show the calculator's GUI
    view = DisplayUI()
    view.show()
    img_left, img_right, img_left_rect, img_right_rect = imgL, imgR, rimgL, rimgR
    view.setDisplayImageLeft(img_left_rect)
    view.setDisplayImageRight(img_right_rect)
    view.img_left_rect = img_left_rect
    view.img_right_rect = img_right_rect    
    disparityController(model=0, view=view)
    # Execute the calculator's main loop
    sys.exit(bmtuner.exec_())
    
#if __name__ == '__main__':
main()

    

# class ImageWidget
        
    
# # 2. Create an instance of QApplication
app = QApplication(sys.argv)

# # 3. Create an instance of your application's GUI
window = QWidget()
layout = QGridLayout()
layout.addWidget(QPushButton('Button (0, 0)'), 0, 0)


window.setWindowTitle('PyQt5 App')
window.setGeometry(100, 100, 280, 80)
window.move(60, 15)
helloMsg = QLabel('<h1>Hello World!</h1>', parent=window)
helloMsg.move(60, 15)

# # 4. Show your application's GUI
window.show()

# # 5. Run your application's event loop (or main loop)
sys.exit(app.exec_())
