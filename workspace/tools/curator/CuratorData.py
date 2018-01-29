import pandas as pd
import pathlib2 as pathlib
from PIL import Image

class CuratorData(object):

    def __init__(self, dataFolder):
        self.folder = str(pathlib.Path(dataFolder).resolve())
        dataFile = self.folder + '/out.csv'
        self.df = pd.read_csv(dataFile)
        self.df['usable'] = 1
        self.png = self.df['PNG']
        self.size = len(self.png)

    def getImage(self, idx):
        return Image.open(self.folder + '/' + self.png[idx])

    def getSize(self):
        return self.size

    def saveData(self):
        self.df.to_csv(self.folder + '/processed.csv')

    def setUsable(self, flag, startRange, endRange):
        # 
        # Ensure that the bounds are respected. 
        if startRange < 0:
            startRange = 0
        if endRange > self.size:
            endRange = self.size - 1
        # 
        # Write the data.
        self.df.loc[startRange:endRange, ('usable')] = flag