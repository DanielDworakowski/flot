import re
import glob
import os.path
import numpy as np
import pandas as pd
import numpy.ma as ma
from PIL import Image
import pathlib2 as pathlib
from pykalman import KalmanFilter

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside'''
    return [atoi(c) for c in re.split('(\d+)', text)]

class AutoLabelConf(object):
    distanceThreshold = 0.7
    consecutiveZeroForPositive = 5
    medianFilterSizer = 5
    maxDistanceMarker = 9999
    tSkip = 1/30.0 * 4

class CuratorData(object):

    def __init__(self, dataFolder):
        self.folder = str(pathlib.Path(dataFolder).resolve())
        self.df = None
        self.dataIdx = 0
        self.touched = False
        dataFile = self.folder + '/out.csv'
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
        #
        # Set label forcing.
        if 'forcedLabel' not in self.df:
            self.df['forcedLabel'] = 0
        try:
            self.df = self.df.drop(['collision_free'], axis=1)
        except:
            pass
        self.png = self.df['PNG']
        self.size = len(self.png)
        self.df.index.name = idxcol
        self.labelConf = AutoLabelConf()
        self.remap = {
            False: -1,
            True: 1
        }

    def getLabel(self, idx):
        idx = min(self.size - 1, max(0, idx))
        lab = self.df['forcedLabel'][idx]
        if lab == 0: # Case for no forced label.
            dist = self.df['Sonar:Smoothed'][idx]
            lab = self.remap[dist > self.labelConf.distanceThreshold]
        return lab

    def getData(self, idx):
        idx = min(self.size - 1, max(0, idx))
        return Image.open(self.folder + '/' + self.png[idx]), self.df['Sonar:Smoothed'][idx], idx

    def getSize(self):
        return self.size

    def _prune(self, pruneDf):
        ts = pruneDf['video_ts:Timestamp']
        mask = np.zeros_like(ts, dtype=bool)
        ts_shift = np.delete(np.roll(np.copy(ts), -1), -1)
        ts = ts[:-1]
        tDiff= ts_shift - ts
        #
        # Iterate over all of the indexes and remove if there is not enough of a tdiff.
        runDiff = 0
        mask[0] = True
        for idx, diff in enumerate(tDiff):
            if runDiff > self.labelConf.tSkip:
                mask[idx] = 1
                runDiff = False
            runDiff += diff
        #
        # Balance the dataset.
        label = pruneDf['collision_free']
        positives = label==1
        negatives = label==0
        positives[mask == 0] = False
        negatives[mask == 0] = False
        labels_diff = label[positives].shape[0] - label[negatives].shape[0]
        if labels_diff > 0:
            posShape = label[positives].shape
            new_positive_label = np.ones(posShape)
            new_positive_label[np.random.choice(posShape[0],labels_diff,replace=False)]= 0
            mask[positives] = new_positive_label
        elif labels_diff < 0:
            labels_diff = -1*labels_diff
            negShape = label[negatives].shape
            new_negative_label = np.ones(negShape)
            new_negative_label[np.random.choice(negShape[0],labels_diff,replace=False)]= 0
            mask[negatives] = new_negative_col_traj
        return mask

    def saveData(self, force = False):
        #
        # Dont save anything if the data was not touched.
        if not self.touched and not force:
            print('No change was made to the labels, not saving.')
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
        print('Saving labels to ', saveLoc)
        labelDf = self.df[self.df['usable'] == 1] # Convert to logical indexing first.
        # mask = self._prune(labelDf)
        # labelDf = labelDf[mask.astype(bool)]
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

    def setForce(self, flag, startRange, endRange):
        #
        # Ensure that the bounds are respected.
        if startRange < 0:
            startRange = 0
        if endRange > self.size:
            endRange = self.size - 1
        #
        # Write the data.
        self.df.loc[startRange:endRange, ('forcedLabel')] = int(flag)
        self.touched = True

    @staticmethod
    def __consecutive(data, stepsize=1):
        # https://stackoverflow.com/questions/7352684/how-to-find-the-groups-of-consecutive-elements-from-an-array-in-numpy
        return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

    def autoLabel(self):
        sonarKey = 'sonar-observation:Value'
        smoothedKey = 'Sonar:Smoothed'
        arrays = self.__consecutive(np.where(self.df[sonarKey] == 0)[0])
        mask = np.zeros(len(self.df[sonarKey]))
        self.df[smoothedKey] = self.df[sonarKey]
        self.df.loc[np.isnan(self.df[smoothedKey]), smoothedKey] = self.labelConf.maxDistanceMarker
        masked = ma.masked_array(self.df[smoothedKey], mask = mask)
        #
        # Assume that if there are multiple zeros in a row this indicates that the
        # there is nothing there. Obstacles are beyond the max range of the sensor.
        # If there are too few measurements in a row, extend the last measurement.
        if len(arrays[0]) > 0:
            for array in arrays:
                masked.mask[array] = 1
                if len(array) > self.labelConf.consecutiveZeroForPositive:
                    self.df.loc[array, (smoothedKey)] = self.labelConf.maxDistanceMarker
                else:
                    self.df.loc[array, (smoothedKey)] = self.df[sonarKey][max(array[0] - 1, 0)]
            kf = KalmanFilter([1], [1], [0.2**2], [0.2**2])
            means, cov = kf.smooth(masked)
            #
            # Create the labels based on a distance threshold.
            self.df[smoothedKey] = np.squeeze(means)
        # self.df['collision_free'] = self.df[smoothedKey] > self.labelConf.distanceThreshold
        # self.df['collision_free'] = self.df['collision_free'].astype(int)