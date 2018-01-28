import sys
from PIL import Image, ImageFont, ImageDraw
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PIL.ImageQt import ImageQt

class CuratorGui(QMainWindow):

    def __init__(self, data):
        QMainWindow.__init__(self)
        self.setMinimumSize(QSize(640, 480))
        self.setWindowTitle('Curator')
        self.data = data
        self.dIdx = 0
        #
        # Create central widget + layout.
        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)
        gridLayout = QGridLayout(centralWidget)
        centralWidget.setLayout(gridLayout)
        #
        # Rest of the GUI.
        self.imLab = QLabel('Image', self)
        self.dispImg = QLabel('', self)
        self.dispImg.setAlignment(QtCore.Qt.AlignCenter)
        gridLayout.addWidget(self.imLab, 0, 0)
        gridLayout.addWidget(self.dispImg, 1, 0)
        # 
        # Keyboard shortcuts.
        self.rArrow = QShortcut(QKeySequence("right"), self)
        self.rArrow.activated.connect(self.rightArrowCB)
        self.lArrow = QShortcut(QKeySequence("left"), self)
        self.lArrow.activated.connect(self.leftArrowCB)
        self.space = QShortcut(QKeySequence(Qt.Key_Space), self)
        self.space.activated.connect(self.spaceCB)
        self.show()

    def visualize(self):
        #
        # Generate image.
        img = self.data.getImage(self.dIdx)
        draw = ImageDraw.Draw(img)
        #
        # Convert to Qt for presentation.
        imgqt = ImageQt(img)
        pix = QtGui.QPixmap.fromImage(imgqt)
        self.dispImg.setPixmap(pix)
        # 
        # Process all draw events. 
        QApplication.processEvents()

    def rightArrowCB(self):
        print('right')
        self.dIdx += 1
        # 
        # Bounds.
        if self.dIdx > self.data.getSize():
            self.dIdx -= 1

    def leftArrowCB(self):
        print('left')
        self.dIdx -= 1
        # 
        # Bounds.
        if self.dIdx < 0:
            self.dIdx += 1

    def spaceCB(self):
        print('space')