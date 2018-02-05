import pandas as pd
import pathlib2 as pathlib
from PIL import Image
import os.path
import re
import glob

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside'''
    return [atoi(c) for c in re.split('(\d+)', text)]

class AutoLabelConf(object):
    distanceThreshold = 0.5

class CuratorData(object):

    def __init__(self, dataFolder):
        self.folder = str(pathlib.Path(dataFolder).resolve())
        dataFile = self.folder + '/out.csv'
        self.df = None
        self.dataIdx = 0
        self.touched = False
        files = glob.glob(self.folder + '/processed*.csv')
        idxcol = 'idx'
        if len(files) > 0:
            files.sort(key = natural_keys)
            dataFile = files[-1]
            self.dataIdx = int(dataFile.split('processed')[1].split('.')[0])
            self.df = pd.read_csv(dataFile, index_col=idxcol)
        else:
            self.df = pd.read_csv(dataFile)
            self.df['usable'] = 1
            self.df['labelled'] = 0 
        self.png = self.df['PNG']
        self.dist = self.df['Sonar:Value']
        self.size = len(self.png)
        self.df.index.name = idxcol
        self.labelConf = AutoLabelConf()

    def getData(self, idx):
        return Image.open(self.folder + '/' + self.png[idx]), self.dist[idx]

    def getSize(self):
        return self.size

    def saveData(self):
        # 
        # Dont save anything if the data was not touched. 
        if not self.touched:
            return
        # 
        # Save the processed / labelled data.
        self.dataIdx += 1
        saveLoc = self.folder + '/processed' + str(self.dataIdx) + '.csv'
        print('saving to ' + saveLoc)
        self.df.to_csv(saveLoc)
        # 
        # Reduce the data to only that what was labelled to be usable. 
        saveLoc = self.folder + '/labels.csv'
        labelDf = self.df[self.df['usable'] == 1] # Convert to logical indexing first.
        labelDf.to_csv(saveLoc)

    def setUsable(self, flag, startRange, endRange):
        # 
        # Ensure that the bounds are respected. 
        if startRange < 0:
            startRange = 0
        if endRange > self.size:
            endRange = self.size - 1
        # 
        # Write the data.
        self.df.loc[startRange:endRange, ('usable')] = int(flag)
        self.df.loc[startRange:endRange, ('labelled')] = 1
        self.touched = True

    def autoLabel(self):
        self.df['collision_free'] = self.df['Sonar:Value'] > self.labelConf.distanceThreshold
        self.df['collision_free'] = self.df['collision_free'].astype(int)