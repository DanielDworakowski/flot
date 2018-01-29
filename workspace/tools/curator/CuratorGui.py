import sys
from PIL import Image, ImageFont, ImageDraw
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PIL.ImageQt import ImageQt
import numpy as np

class LedIndicator(QAbstractButton):
    scaledSize = 1000.0

    def __init__(self, parent=None, initialColour = 0):
        QAbstractButton.__init__(self, parent)
        self.setMinimumSize(24, 24)
        self.setCheckable(True)
        self.colourPicker = {
            0: QColor(127, 127, 127),
            1: QColor(20, 200, 60)
        }
        self.colour = self.colourPicker[initialColour]

    def setColour(self, colCode):
        self.colour = self.colourPicker[colCode]

    def resizeEvent(self, QResizeEvent):
        self.update()

    def paintEvent(self, QPaintEvent):
        realSize = min(self.width(), self.height())
        painter = QPainter(self)
        pen = QPen(Qt.black)
        pen.setWidth(1)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.translate(self.width() / 2, self.height() / 2)
        painter.scale(realSize / self.scaledSize, realSize / self.scaledSize)
        painter.setBrush(QBrush(self.colour))
        painter.drawEllipse(QPointF(0, 0), 400, 400)

class PicButton(QAbstractButton):

    def __init__(self, parent=None, width = 700, height = 20, ):
        super(PicButton, self).__init__(parent)
        self.parent = parent
        self.data = np.zeros((height, width, 3), dtype=np.uint8)
        self.pix = QtGui.QPixmap.fromImage(ImageQt(Image.fromarray(self.data)))
        self.width = width

    def paintEvent(self, event):
        # 
        # TODO: Fix this to be wider.
        painter = QPainter(self)
        self.pix = QtGui.QPixmap.fromImage(ImageQt(Image.fromarray(np.expand_dims(self.parent.data.df['usable'].as_matrix().astype(np.uint8), axis = 1))))
        painter.drawPixmap(event.rect(), self.pix)

    def sizeHint(self):
        return self.pix.size()

    def mousePressEvent(self, e):
        relLoc = e.x() / (self.width + 1)
        self.parent.setIdx(relLoc)

class CuratorGui(QMainWindow):

    def __init__(self, data):
        QMainWindow.__init__(self)
        self.setMinimumSize(QSize(700, 550))
        self.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        self.setWindowTitle('Curator')
        self.data = data
        self.dIdx = 0
        self.usableImageFlag = True
        self.labelOnOffFlag = False
        #
        # Create central widget + layout.
        centralWidget = QWidget()
        hlayout = QHBoxLayout()
        self.setCentralWidget(centralWidget)
        gridLayout = QGridLayout(centralWidget)
        centralWidget.setLayout(gridLayout)
        #
        # Rest of the GUI.
        self.dispImg = QLabel('', self)
        self.labelBox = QGroupBox('Label Status')
        self.dispImg.setAlignment(QtCore.Qt.AlignCenter)
        self.dispImg.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # 
        # Set the indicator.
        self.labOnOff = QLabel('Status', self)
        self.curLab = QLabel('Current Label', self)
        self.labelOnOffIndicator = LedIndicator(self, self.labelOnOffFlag)
        self.curLabelIndicator = LedIndicator(self, self.usableImageFlag)
        # 
        # Create the data scroller.
        self.pixButton = PicButton(self, 700, 20)
        # 
        # Setup the top bar.
        hlayout.addWidget(self.labOnOff)
        hlayout.addWidget(self.labelOnOffIndicator)
        hlayout.addWidget(self.curLab)
        hlayout.addWidget(self.curLabelIndicator)
        gridLayout.addWidget(self.labelBox, 0, 0)
        gridLayout.addWidget(self.dispImg, 1, 0)
        gridLayout.addWidget(self.pixButton, 2, 0)
        self.labelBox.setLayout(hlayout)
        # 
        # Keyboard shortcuts.
        self.rArrow = QShortcut(QKeySequence("right"), self)
        self.lArrow = QShortcut(QKeySequence("left"), self)
        self.uArrow = QShortcut(QKeySequence("up"), self)
        self.dArrow = QShortcut(QKeySequence("down"), self)
        self.enter = QShortcut(QKeySequence(Qt.EnterKeyDefault), self)
        self.space = QShortcut(QKeySequence(Qt.Key_Space), self)
        self.rArrow.activated.connect(self.rightArrowCB)
        self.lArrow.activated.connect(self.leftArrowCB)
        self.uArrow.activated.connect(self.upArrowCB)
        self.dArrow.activated.connect(self.downArrowCB)
        self.space.activated.connect(self.spaceCB)
        self.enter.activated.connect(self.enterCB)
        self.jumpSize = 60
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

    def paintEvent(self, event):
        pass
        # 
        # Paint indicators.
        # painter = QtGui.QPainter(self)
        # painter.setPen(QtGui.QPen(QtCore.Qt.red))
        # painter.setRenderHint(QPainter.Antialiasing)
        # painter.setPen(QPen(Qt.NoPen))
        # painter.setBrush(QBrush(QColor(127, 127, 127)))
        # painter.drawEllipse(20, 20, 20, 20)
        # print(self.imLab.frameGeometry())

    def setIdx(self, relIdx):
        old = self.dIdx
        self.dIdx = int(self.data.getSize() * relIdx)
        self.data.setUsable(self.flag, min(oldVal, self.dIdx), max(self.dIdx, oldVal))

    def upArrowCB(self):
        oldVal = self.dIdx
        self.dIdx += self.jumpSize
        # 
        # Bounds.
        if self.dIdx > self.data.getSize():
            self.dIdx = self.data.getSize - 1
        # 
        # Label data as requested. 
        if self.labelOnOffFlag:
            self.data.setUsable(self.flag, oldVal, self.dIdx)

    def downArrowCB(self):
        oldVal = self.dIdx
        self.dIdx -= self.jumpSize
        # 
        # Bounds.
        if self.dIdx < 0:
            self.dIdx = 0
        # 
        # Label data as requested. 
        if self.labelOnOffFlag:
            self.data.setUsable(self.flag, self.dIdx, oldVal)

    def rightArrowCB(self):
        oldVal = self.dIdx
        self.dIdx += 1
        # 
        # Bounds.
        if self.dIdx > self.data.getSize():
            self.dIdx -= 1
        # 
        # Label data as requested. 
        if self.labelOnOffFlag:
            self.data.setUsable(self.flag, oldVal, self.dIdx)

    def leftArrowCB(self):
        oldVal = self.dIdx
        self.dIdx -= 1
        # 
        # Bounds.
        if self.dIdx < 0:
            self.dIdx += 1
        # 
        # Label data as requested. 
        if self.labelOnOffFlag:
            self.data.setUsable(self.flag, self.dIdx, oldVal)

    def spaceCB(self):
        self.usableImageFlag = not self.usableImageFlag
        print(self.usableImageFlag)
        self.curLabelIndicator.setColour(int(self.usableImageFlag))

    def enterCB(self):
        self.labelOnOffFlag = not self.labelOnOffFlag
        self.labelOnOffIndicator.setColour(int(self.labelOnOffFlag))
