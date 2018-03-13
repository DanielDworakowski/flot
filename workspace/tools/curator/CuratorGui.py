import sys
import numpy as np
from util.debug import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PIL.ImageQt import ImageQt
from torchvision import transforms
from torch.autograd import Variable
import tools.visualization as visutil
from PyQt5 import QtCore, QtWidgets, QtGui
from PIL import Image, ImageFont, ImageDraw

class LedIndicator(QAbstractButton):
    scaledSize = 1000.0

    def __init__(self, parent=None, initialColour = 0):
        QAbstractButton.__init__(self, parent)
        self.setMinimumSize(24, 24)
        self.setCheckable(True)
        self.colourPicker = {
           -1: QColor(255, 30, 30),
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

class UsableButton(QAbstractButton):

    def __init__(self, parent=None, width = 700, height = 20):
        super(UsableButton, self).__init__(parent)
        self.parent = parent
        self.data = np.zeros((height, width, 3), dtype=np.uint8)
        self.pix = QtGui.QPixmap.fromImage(ImageQt(Image.fromarray(self.data)))
        self.width = width
        self.curIdx = 0
        self.maxIdx = self.parent.data.getSize() - 1
        self.lineWidth = int(0.001 * self.maxIdx)
        idx = self.parent.data.df['usable'].as_matrix().astype(np.uint8)
        self.rgbMatrix = np.zeros((1 ,idx.shape[0], 3)).astype(np.uint8)

    def paintEvent(self, event):
        painter = QPainter(self)
        mask = self.parent.data.df['labelled'].as_matrix().astype(np.uint8)
        idx = self.parent.data.df['usable'].as_matrix().astype(np.uint8)
        self.rgbMatrix[0, (idx < 1), :] = np.array([200, 19, 56]) # Bad.
        self.rgbMatrix[0, (idx > 0), :] = np.array([20, 200, 60]) # Good.
        self.rgbMatrix[0, :, :] = self.rgbMatrix[0, :] * mask[:, np.newaxis] # Mask everything out that was not labeled.
        #
        # Set a white line at the current index.
        self.rgbMatrix[0, max(0, self.curIdx - self.lineWidth):min(self.maxIdx - 1, self.curIdx + self.lineWidth), :] = 255
        self.pix = QtGui.QPixmap.fromImage(ImageQt(Image.fromarray(self.rgbMatrix)))
        painter.drawPixmap(event.rect(), self.pix)

    def setCurIdx(self, idx):
        self.curIdx = idx

    def sizeHint(self):
        return self.pix.size()

    def mousePressEvent(self, e):
        relLoc = e.x() / (self.width + 1)
        self.parent.setIdx(relLoc)

class ForceButton(QAbstractButton):

    def __init__(self, parent=None, width = 700, height = 20):
        super(ForceButton, self).__init__(parent)
        self.parent = parent
        self.data = np.zeros((height, width, 3), dtype=np.uint8)
        self.pix = QtGui.QPixmap.fromImage(ImageQt(Image.fromarray(self.data)))
        self.width = width
        self.curIdx = 0
        self.maxIdx = self.parent.data.getSize() - 1
        self.lineWidth = int(0.001 * self.maxIdx)
        idx = self.parent.data.df['forcedLabel'].as_matrix().astype(np.int8)
        self.rgbMatrix = np.zeros((1 ,idx.shape[0], 3)).astype(np.uint8)

    def paintEvent(self, event):
        painter = QPainter(self)
        idx = self.parent.data.df['forcedLabel'].as_matrix().astype(np.int8)
        self.rgbMatrix[0, (idx <  0), :] = np.array([200, 19, 56]) # Bad.
        self.rgbMatrix[0, (idx == 0), :] = np.array([120, 120, 120]) # neutral.
        self.rgbMatrix[0, (idx >  0), :] = np.array([20, 200, 60]) # Good.
        #
        # Set a white line at the current index.
        self.rgbMatrix[0, max(0, self.curIdx - self.lineWidth):min(self.maxIdx - 1, self.curIdx + self.lineWidth), :] = 255
        self.pix = QtGui.QPixmap.fromImage(ImageQt(Image.fromarray(self.rgbMatrix)))
        painter.drawPixmap(event.rect(), self.pix)

    def setCurIdx(self, idx):
        self.curIdx = idx

    def sizeHint(self):
        return self.pix.size()

    def mousePressEvent(self, e):
        relLoc = e.x() / (self.width + 1)
        self.parent.setIdx(relLoc)

class NNVis(object):

    def __init__(self, conf, model):
        self.conf = conf
        self.model = model
        self.visModelCB = lambda img, idx: (Image.fromarray(np.zeros((1,1,3), dtype=np.uint8)), img)
        self.rgbTable = visutil.rtobTable()
        self.sm = None
        self.lIdx = -1
        self.lastImg = None
        self.lastNet = None
        self.toPIL = transforms.ToPILImage()
        if self.conf != None:
            import torch
            from config.DefaultNNConfig import getDefaultTransform
            # self.t = getDefaultTransform(conf)
            self.t = self.conf.transforms
            self.sm = torch.nn.Softmax(dim = 1)
            self.visModelCB = self.visModel

    def visModel(self, img, idx):
        #
        # If the current image being looked at is the same as the last image, do not process.
        if self.lIdx == idx:
            return self.lastNet, self.lastImg
        #
        # Process the new image.
        draw = ImageDraw.Draw(img)
        sample = {'img': img, 'labels': np.array([0]), 'meta': {'shift':(0,0)}}
        data = self.t(sample)
        if self.conf.usegpu:
            labels = Variable(data['labels'].squeeze_()).cuda(async = True)
            out = self.conf.hyperparam.model(Variable(data['img']).unsqueeze_(0).cuda(async = True))
        else:
            out = self.conf.hyperparam.model(Variable(data['img']).unsqueeze_(0))
        posClasses = self.model.getClassifications(out, self.sm).squeeze_()
        visutil.drawTrajectoryDots(0, 0, 12, img.size, self.rgbTable, draw, self.conf, posClasses)
        self.lIdx = idx
        retnet = self.toPIL(data['img'].squeeze_())
        self.lastImg = img
        self.lastNet = retnet
        return retnet, img

    def getTrainThreshold(self):
        if self.conf != None:
            return self.conf.distThreshold
        return None

class CuratorGui(QMainWindow):

    def __init__(self, data):
        QMainWindow.__init__(self)
        self.running = True
        self.setMinimumSize(QSize(700, 550))
        self.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        self.setWindowTitle('Curator')
        self.data = data
        self.dIdx = 0
        self.usableImageFlag = True
        self.usableOnOffFlag = False
        self.forceOnOffFlag = False
        self.forceLab = 0
        self.modelVis = NNVis(None, None)
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
        self.savebutton = QPushButton('Save', self)
        #
        # Network image.
        self.netimg = QLabel('', self)
        self.netimg.setAlignment(QtCore.Qt.AlignCenter)
        self.netimg.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        #
        # Set the indicator.
        self.labOnOff = QLabel('Usable Status', self)
        self.usableLab = QLabel('Usable Label', self)
        self.forceOnOffLab = QLabel('Forced status', self)
        self.forceText = QLabel('Forced Label', self)
        self.dist = QLabel('Distance:', self)
        self.labelOnOffIndicator = LedIndicator(self, self.usableOnOffFlag)
        self.usableLabelIndicator = LedIndicator(self, self.usableImageFlag)
        self.forceOnOffIndicator = LedIndicator(self, self.forceOnOffFlag)
        self.forceLabelIndicator = LedIndicator(self, self.forceLab)
        self.distTxt = QLabel('tx', self)
        #
        # Create the data scroller.
        self.usableBar = UsableButton(self, 700, 20)
        self.forceBar = ForceButton(self, 700, 20)
        #
        # Setup the top bar.
        hlayout.addWidget(self.labOnOff)
        hlayout.addWidget(self.labelOnOffIndicator)
        hlayout.addWidget(self.usableLab)
        hlayout.addWidget(self.usableLabelIndicator)
        hlayout.addWidget(self.forceOnOffLab)
        hlayout.addWidget(self.forceOnOffIndicator)
        hlayout.addWidget(self.forceText)
        hlayout.addWidget(self.forceLabelIndicator)
        hlayout.addWidget(self.dist)
        hlayout.addWidget(self.distTxt)
        hlayout.addWidget(self.savebutton)
        gridLayout.addWidget(self.labelBox, 0, 0)
        gridLayout.addWidget(self.dispImg, 1, 0)
        gridLayout.addWidget(self.usableBar, 2, 0)
        gridLayout.addWidget(self.forceBar, 3, 0)
        gridLayout.addWidget(self.netimg, 4 , 0)
        self.labelBox.setLayout(hlayout)
        #
        # Keyboard shortcuts.
        self.rArrow = QShortcut(QKeySequence("right"), self)
        self.lArrow = QShortcut(QKeySequence("left"), self)
        self.uArrow = QShortcut(QKeySequence("up"), self)
        self.dArrow = QShortcut(QKeySequence("down"), self)
        self.one = QShortcut(QKeySequence(Qt.Key_1), self)
        self.two = QShortcut(QKeySequence(Qt.Key_2), self)
        self.three = QShortcut(QKeySequence(Qt.Key_3), self)
        self.four = QShortcut(QKeySequence(Qt.Key_4), self)
        self.returnKey = QShortcut(QKeySequence(Qt.Key_Return), self)
        self.enter = QShortcut(QKeySequence(Qt.Key_Enter), self)
        self.space = QShortcut(QKeySequence(Qt.Key_Space), self)
        self.rArrow.activated.connect(self.forwardOneCV)
        self.lArrow.activated.connect(self.backOneCB)
        self.uArrow.activated.connect(self.skipCB)
        self.dArrow.activated.connect(self.skipBackCB)
        self.space.activated.connect(self.usableFlagCB)
        self.enter.activated.connect(self.labelOnOffCB)
        self.returnKey.activated.connect(self.labelOnOffCB)
        self.savebutton.clicked.connect(self.saveCB)
        self.one.activated.connect(self.forceNegativeCB)
        self.two.activated.connect(self.forceNeutralCB)
        self.three.activated.connect(self.forcePositiveCB)
        self.four.activated.connect(self.forceOnOffCB)
        self.jumpSize = 60
        self.show()

    def closeEvent(self, event):
        if self.running:
            self.running = False
            self.data.saveData()

    def visualize(self):
        #
        # Generate image.
        img, dist, self.dIdx = self.data.getData(self.dIdx)
        draw = ImageDraw.Draw(img)
        #
        # Convert to Qt for presentation.
        netData, img = self.modelVis.visModelCB(img, self.dIdx)
        #
        # Show all images.
        imgqt = ImageQt(img)
        pix = QtGui.QPixmap.fromImage(imgqt)
        self.dispImg.setPixmap(pix)
        imgqt = ImageQt(netData)
        pix = QtGui.QPixmap.fromImage(imgqt)
        self.netimg.setPixmap(pix)
        #
        # Create the distance information text.
        self.distTxt.setText('%0.2f'%dist)
        palette = self.distTxt.palette()
        col = None
        if dist > self.data.labelConf.distanceThreshold:
            col = QColor(20, 200, 60)
        else:
            col = QColor(255, 0, 0)
        palette.setColor(self.distTxt.foregroundRole(), col)
        self.distTxt.setPalette(palette)
        #
        # Update the bottom bar on the current location.
        self.usableBar.setCurIdx(self.dIdx)
        #
        # Process all draw events.
        QApplication.processEvents()
        self.labelOnOffIndicator.update()
        self.usableLabelIndicator.update()
        self.forceOnOffIndicator.update()
        self.forceLabelIndicator.update()
        self.usableBar.update()
        self.forceBar.update()

    def setIdx(self, relIdx):
        oldVal = self.dIdx
        self.dIdx = int(self.data.getSize() * relIdx)
        #
        # Set Flags.
        if self.usableOnOffFlag:
            self.data.setUsable(self.usableImageFlag, min(oldVal, self.dIdx), max(self.dIdx, oldVal))

    def skipCB(self):
        oldVal = self.dIdx
        self.dIdx += self.jumpSize
        #
        # Bounds.
        if self.dIdx > self.data.getSize():
            self.dIdx = self.data.getSize() - 1
        #
        # Label data as requested.
        if self.usableOnOffFlag:
            self.data.setUsable(self.usableImageFlag, oldVal, self.dIdx)
        if self.forceOnOffFlag:
            self.data.setForce(self.forceLab, oldVal, self.dIdx)

    def skipBackCB(self):
        oldVal = self.dIdx
        self.dIdx -= self.jumpSize
        #
        # Bounds.
        if self.dIdx < 0:
            self.dIdx = 0
        #
        # Label data as requested.
        if self.usableOnOffFlag:
            self.data.setUsable(self.usableImageFlag, self.dIdx, oldVal)
        if self.forceOnOffFlag:
            self.data.setForce(self.forceLab, self.dIdx, oldVal)


    def forwardOneCV(self):
        oldVal = self.dIdx
        self.dIdx += 1
        #
        # Bounds.
        if self.dIdx > self.data.getSize():
            self.dIdx = self.data.getSize() - 1
        #
        # Label data as requested.
        if self.usableOnOffFlag:
            self.data.setUsable(self.usableImageFlag, oldVal, self.dIdx)
        if self.forceOnOffFlag:
            self.data.setForce(self.forceLab, oldVal, self.dIdx)

    def backOneCB(self):
        oldVal = self.dIdx
        self.dIdx -= 1
        #
        # Bounds.
        if self.dIdx < 0:
            self.dIdx = 0
        #
        # Label data as requested.
        if self.usableOnOffFlag:
            self.data.setUsable(self.usableImageFlag, self.dIdx, oldVal)
        if self.forceOnOffFlag:
            self.data.setForce(self.forceLab, self.dIdx, oldVal)


    def usableFlagCB(self):
        self.usableImageFlag = not self.usableImageFlag
        self.usableLabelIndicator.setColour(int(self.usableImageFlag))

    def labelOnOffCB(self):
        self.usableOnOffFlag = not self.usableOnOffFlag
        self.labelOnOffIndicator.setColour(int(self.usableOnOffFlag))

    def forceNegativeCB(self):
        self.forceLab = -1
        self.forceLabelIndicator.setColour(self.forceLab)

    def forceNeutralCB(self):
        self.forceLab = 0
        self.forceLabelIndicator.setColour(self.forceLab)

    def forcePositiveCB(self):
        self.forceLab = 1
        self.forceLabelIndicator.setColour(self.forceLab)

    def forceOnOffCB(self):
        self.forceOnOffFlag = not self.forceOnOffFlag
        self.forceOnOffIndicator.setColour(int(self.forceOnOffFlag))

    def setModel(self, model, conf):
        if model != None:
            #
            # Resize the window for the bottom.
            self.setMinimumSize(QSize(700, 550+224))
            #
            # Setup visualization for the network.
            self.modelVis = NNVis(conf, model)
            thresh = self.modelVis.getTrainThreshold()
            if thresh is not None:
                self.data.labelConf.distanceThreshold = thresh


    def saveCB(self, event):
        print('Saving data!')
        self.data.saveData(True)