import sys
from PyQt5.QtCore import QSize
from PIL.ImageQt import ImageQt
import tools.visualization as visutil
from PyQt5 import QtCore, QtWidgets, QtGui
from PIL import Image, ImageFont, ImageDraw
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QGridLayout, QWidget

class Visualizer(QMainWindow):

    def __init__(self):
        QMainWindow.__init__(self)
        self.setMinimumSize(QSize(640, 480))
        self.setWindowTitle('Agent View')
        #
        # Create central widget + layout.
        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)
        gridLayout = QGridLayout(centralWidget)
        centralWidget.setLayout(gridLayout)
        #
        # Data structures for generating visualization.
        self.rgbTable = visutil.rtobTable()
        #
        # Rest of the GUI.
        self.imLab = QLabel('Image', self)
        self.dispImg = QLabel('', self)
        self.dispImg.setAlignment(QtCore.Qt.AlignCenter)
        gridLayout.addWidget(self.imLab, 0, 0)
        gridLayout.addWidget(self.dispImg, 1, 0)
        self.show()

    def visualize(self, obs, action, agent):
        #
        # Generate image.
        # img = Image.fromarray(obs['img'].decompressPNG()[:,:,0:3])
        # draw = ImageDraw.Draw(img)
        # print(action.meta['activations'])
        # visutil.drawTrajectoryDots(0, 0, 7, img.size, self.rgbTable, draw, agent.nnconf, action.meta['activations'])
        # #
        # # Convert to Qt for presentation.
        # imgqt = ImageQt(img)
        # pix = QtGui.QPixmap.fromImage(imgqt)
        # self.dispImg.setPixmap(pix)
        # # self.title.setText('index: %s'%idx.val)
        # QApplication.processEvents()

        if obs['img'].uint8Img is not None:
            img = Image.fromarray(obs['img'].uint8Img)
            draw = ImageDraw.Draw(img)
            # print(action.meta['activations'])
            # visutil.drawTrajectoryDots(0, 0, 7, img.size, self.rgbTable, draw, agent.nnconf, action.meta['activations'])
            #
            # Convert to Qt for presentation.
            imgqt = ImageQt(img)
            pix = QtGui.QPixmap.fromImage(imgqt)
            self.dispImg.setPixmap(pix)
            # self.title.setText('index: %s'%idx.val)
            QApplication.processEvents()
