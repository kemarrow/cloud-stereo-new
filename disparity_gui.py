# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 13:26:55 2022

@author: kathe
"""

import sys
import cv2
import numpy as np

# 1. Import `QApplication` and all the required widgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QVBoxLayout, QHBoxLayout
from PyQt5.QtWidgets import QLabel, QSlider, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class DisplayUI(QMainWindow):
    def __init__(self, initial_params=None):
        super().__init__()
        self.setWindowTitle('Block Matcher Tuner')
        self.setFixedSize(600, 600)
        # Set the central widget
        self.generalLayout = QGridLayout()
        self._centralWidget = QWidget(self)
        self.setCentralWidget(self._centralWidget)
        self._centralWidget.setLayout(self.generalLayout)
        self._createImageDisplay()
        self._createDisparityDisplay()
        self._createSliders(initial_params=initial_params)
                 
    def _createImageDisplay(self):
        self.displayLeft = MplCanvasWidget(self)
        self.generalLayout.addWidget(self.displayLeft, 0, 0)
        self.displayRight = MplCanvasWidget(self)
        self.generalLayout.addWidget(self.displayRight, 0, 1)
        self.displayDisparity = MplCanvasWidget(self)
        self.generalLayout.addWidget(self.displayDisparity, 1, 1)

    def _createSliders(self, initial_params=None):
        if initial_params is None:
            initial_params = {}
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
            if name in initial_params.keys():
                self.sliders[name].setValue(initial_params[name])
            else:
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
        super().__init__(fig)
        self.setParent(parent)

class disparityController:
    def __init__(self, model, view):
        self._evaluate = model
        self._view = view
        self._connectSignals()

    def _connectSignals(self):
        for slider_name in self._view.sliders.keys():
            print(slider_name)
            self._view.sliders[slider_name].valueChanged.connect(
                #lambda: self.disparity_calc()) 
                self._view.printSliders())
            #lambda: printVal(self._view.sliders[slider_name].value()))

    def disparity_calc(self):
        params = {name: self._view.sliders[name].value() for name in self._view.sliders.keys()}
        left_matcher = cv2.StereoSGBM_create(**params)
        disparity_left = left_matcher.compute(self._view.img_left_rect,
                                              self._view.img_right_rect)
        print('Disp calculated', np.isfinite(disparity_left).sum(), disparity_left.sum())
        print(params)
        self._view.printSliders()
        self._view.setDisplayImageDisparity(disparity_left)
        self._view.displayDisparity.draw()
        
def match_gui(img_left, img_right, img_left_rect, img_right_rect, initial_params=None):
    """Main function."""
    # Create an instance of QApplication
    bmtuner = QApplication(sys.argv)
    # Show the calculator's GUI
    view = DisplayUI(initial_params)
    view.show()
    view.setDisplayImageLeft(img_left_rect)
    view.setDisplayImageRight(img_right_rect)
    view.img_left_rect = img_left_rect
    view.img_right_rect = img_right_rect    
    disparityController(model=0, view=view)
    # Execute the calculator's main loop
    sys.exit(bmtuner.exec_())