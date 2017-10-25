import os
import pandas as pd
from pandas import rolling_median
import numpy as np

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import make_scorer
from detect_peaks import detect_peaks

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.datasets import dump_svmlight_file

from sklearn.feature_selection import RFECV

## consts :
dataSource = r'../raw-data/utf8'
SAMPLE_FREQ = 50
FILE_MARGINES = 5* SAMPLE_FREQ  ## number of samples to ignore in the  start and in the end of the file (5 seconds )
WINDOW_SIZE = 128  ## sliding window size
PEAKS_WINDOW_SIZE = 5*WINDOW_SIZE  ## sliding window size for peaks count feature

DEVICE_MODE_LABELS = ['pocket','swing','texting','talking','whatever']
USER_MODE_LABELS = ['walking','fastwalking','stairs','static','whatever']

FEATURES = ['agforce','agyro',                          ## avarage
            'mgforce','mgyro',                       ## median
            'vgforce','vgyro',                         ## variance
            'iqrforce','iqqgyro',                    ## iqr
            'entforce','entgyro',                      ## entropy
            'gforcegyrocorr', ## correlation
            'skforce', 'skgyro',                       ## skewness
            'kuforce','kugyro',                         ## kurtosis
            'MultiGyroAcc',      ## max  gyro * max acc
            ##'MultiVarGyroAcc', ## variance  gyro * variance acc
            'maxgforce','maxgyro',                      ## max
               # 'maxgforceabs','maxgyroabs',     ## abs max
            'mingforce','mingyro',                      ## min
               # 'mingforceabs','mingyroabs',     ## abs min
            # 'aadforce','aadgyro',            ## average absolute dfference
            # 'smaforce','smagyro',            ## signal magnitude area
               # 'zcrforce','zcrgyro',           ## zero crossing rate
            'gcrgforce',                    ## g-crossing rate
            'mcrgforce','mcrgyro',          ## medain-crossing rate
            'peaksgforce','peaksgyro',      ## peaks count in PEAKS_WINDOW_SIZE
            #'enyforce','enygyro',                       ## signal energy
            #'amAccGyro',               ## amplitude Acc Gyro
            'ampgforce','ampgyro']                      ## amplitude |max


__FEATURES = ['agforce','agyro',               ## avarage
            'mgforce','mgyro',               ## median
            'vgforce','vgyro',               ## variance
            'iqrforce','iqqgyro',            ## iqr
            'entforce','entgyro',            ## entropy
            #'gforcegyrocorr',               ## correlation
            'skforce', 'skgyro',             ## skewness
            'kuforce','kugyro',              ## kurtosis
            'MultiGyroAcc',                  ## max  gyro * max acc
            'maxgforce','maxgyro',           ## max
            'maxgforceabs','maxgyroabs',     ## abs max
            'mingforce','mingyro',           ## min
            'mingforceabs','mingyroabs',     ## abs min
            'ampgforce','ampgyro' ,           ## amplitude |max - min|
            'peaksgforce','peaksgyro'       ## peaks count in PEAKS_WINDOW_SIZE
#            'light' ,                         ## embient light sensor
#             'alight',
#             'mlight',
#             'vlight',
#             'maxlight',
#             'minlight',
#             'amplight'
]

## Calulates high level features and add to given data frame add norm feature for g-force , gyro vectors
## calculates additional statistics features on the norm properties using sliding window fill NaN values
def RollingPercentile(values,a):
    res = np.percentile(values,a)
    return res

def RollingEntropy(values):
    res = (-values*np.log2(values)).sum(axis=0)
    return res

def NormValues(values):
    Res = values/values.max()
    return Res

def LowPassFilter(values):
    threshold = values.mean()+3*values.std()
    ResE = rolling_median(values, window=15, center=True).fillna(method='bfill').fillna(method='ffill')
    #ResE = NormValues(ResEtemp)
    return ResE

def RemoveOutlier (values):
    threshold = values.mean()+3*values.std()
    ResEtemp = rolling_median(values, window=15, center=True).fillna(method='bfill').fillna(method='ffill')
    difference = np.abs(values - ResEtemp)
    outlier_idx = difference > threshold
    values[outlier_idx] = threshold
    #ResE = NormValues(ResEtemp)
    return values

 ## peaks detection :
    mph = 0 ## minimum peak height
    mpd = 5 ## minimum peak distance = 20
    def peaks(values):
        return len(detect_peaks(values, mph, mpd, show=False))

def AveAbsDiff(values):
    res = np.mean(np.abs(values - values.mean()))
    return res

# Signal Magnitude Area
def SigMagArea(values):
    res = np.sum(values.mean())
    return res

# Zero crossing rate
def ZeroCorsRate(values):
    res = (np.nonzero(np.diff(values > 0))[0]).size
    return res

# g-crossing rate
def GCorsRate(values):
    res = (np.nonzero(np.diff(values > 1))[0]).size
    return res

# medain-crossing rate
def MCorsRate(values):
    tem = np.median(values)
    res = (np.nonzero(np.diff(values > tem))[0]).size
    return res

# rolling window on 2d numpy array. return crrelation between first and second array colomn
# input 2d array with 2 colomns and rolling window size
# output: 1d array with correlation results

# peaks detection :
mph = 0 ## minimum peak height
mpd = 5 ## minimum peak distance = 20
def peaks(values):
    return len(detect_peaks(values, mph, mpd, show=False))

def window_correlation(a,window_size):
    ## pad array start with zeros for rolling window compatibility
    c = np.append(np.zeros((window_size,2),np.float64 ), a , axis=0)

    ## stacks windows :
    interlaceing = np.hstack(c[i:1+i-window_size or None:1] for i in range(0,window_size) )

    # get left and right indices :
    l = interlaceing[:,range(0, 2*window_size, 2)]
    r = interlaceing[:,range(1, 2*window_size, 2)]

    ## correlates left and right values
    return  np.array([np.corrcoef(l[i] , r[i],rowvar=0)[0,1] for i in range(len(l)-1)])

def addFeatures(_rdf):

    idf = pd.DataFrame()

    idf['timestamp'] = _rdf['time']

    ## norm calculations :
    g = 9.8
    gfx = _rdf['gfx'] / g
    gfy = _rdf['gFy'] / g
    gfz = _rdf['gFz']  / g
    idf['gforce'] = np.sqrt(gfx**2 + gfy**2 + gfz**2)
    idf['gyro'] = np.sqrt(_rdf['wx']**2 + _rdf['wy']**2 + _rdf['wz']**2)

    ## calculates statistics features on rolling window :

    idf['agforce'] = idf['gforce'].rolling(window=WINDOW_SIZE,min_periods=1,center=False).mean()
    idf['agyro']   = idf['gyro'].rolling(window=WINDOW_SIZE,min_periods=1 ,center=False).mean()

    idf['mgforce'] = idf['gforce'].rolling(window=WINDOW_SIZE,min_periods=1,center=False).median()
    idf['mgyro']   = idf['gyro'].rolling(window=WINDOW_SIZE,min_periods=1 ,center=False).median()

    idf['vgforce'] = idf['gforce'].rolling(window=WINDOW_SIZE,min_periods=1,center=False).var()
    idf['vgyro']   = idf['gyro'].rolling(window=WINDOW_SIZE,min_periods=1,center=False).var()

    idf['maxgforce'] = idf['gforce'].rolling(window=WINDOW_SIZE,min_periods=1,center=False).max()
    idf['maxgyro']   = idf['gyro'].rolling(window=WINDOW_SIZE,min_periods=1 ,center=False).max()

    idf['maxgforceabs'] = abs(idf['maxgforce'])
    idf['maxgyroabs']   = abs(idf['maxgyro'])

    idf['mingforce'] = idf['gforce'].rolling(window=WINDOW_SIZE,min_periods=1,center=False).min()
    idf['mingyro']  = idf['gyro'].rolling(window=WINDOW_SIZE,min_periods=1 ,center=False).min()

    idf['mingforceabs'] = abs(idf['mingforce'])
    idf['mingyroabs']   = abs(idf['mingyro'])

    idf['ampgforce'] = idf['maxgforce'] - idf['mingforce']
    idf['ampgyro']  = idf['maxgyro'] - idf['mingyro']

    roll_force = idf['gforce'].rolling(window=WINDOW_SIZE,min_periods=1,center=False)
    roll_gyro = idf['gyro'].rolling(window=WINDOW_SIZE,min_periods=1,center=False)

    ## iqr calculations:
    q25 = roll_gyro.apply(RollingPercentile, [25.0])
    q75 = roll_gyro.apply(RollingPercentile, [75.0])
    idf['iqrforce'] = q75 - q25
    idf['iqqgyro'] = q75 - q25

    ## entropy calculations:
    idf['entforce'] = roll_force.apply(RollingEntropy);
    idf['entgyro'] = roll_gyro.apply(RollingEntropy);

    ## ratio calculations
    idf['MultiGyroAcc'] = idf['maxgforce']*idf['maxgyro']

    idf['skforce'] = roll_force.skew()
    idf['skgyro'] = roll_gyro.skew()

    idf['kuforce'] = roll_force.kurt()
    idf['kugyro'] = roll_gyro.kurt()

    idf['light'] = rdf['I'] if 'I' in rdf else 0.0

    idf['peaksgforce'] = idf['gforce'].rolling(window=PEAKS_WINDOW_SIZE,min_periods=1 ,center=False).apply(peaks)
    idf['peaksgyro'] = idf['gyro'].rolling(window=PEAKS_WINDOW_SIZE,min_periods=1 ,center=False).apply(peaks)

    ## gforce gyro correlation :
    idf['gforcegyrocorr'] = window_correlation(idf[['gforce','gyro']].values,WINDOW_SIZE)

    idf['alight'] = idf['light'].rolling(window=WINDOW_SIZE, min_periods=1, center=False).mean()
    idf['mlight'] = idf['light'].rolling(window=WINDOW_SIZE, min_periods=1, center=False).median()
    idf['vlight'] = idf['light'].rolling(window=WINDOW_SIZE, min_periods=1, center=False).var()
    idf['maxlight'] = idf['light'].rolling(window=WINDOW_SIZE, min_periods=1, center=False).max()
    idf['minlight'] = idf['light'].rolling(window=WINDOW_SIZE, min_periods=1, center=False).min()
    idf['amplight'] = idf['maxlight'] - idf['minlight']

    # new features 22.10.17
    idf['gcrgforce'] = roll_force.apply(GCorsRate)
    idf['mcrgforce'] = roll_force.apply(MCorsRate)
    idf['mcrgyro'] = roll_force.apply(MCorsRate)

    idf.fillna(method='ffill', axis=0, inplace=True)

    return idf


## get latest sensors samples in json format
import urllib2
import sys

if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

def getLatest():
    req = urllib2.Request('https://us-central1-sensors-efc67.cloudfunctions.net/latest')
    response = urllib2.urlopen(req)
    return response.read()

def getTrain():
    req = urllib2.Request('https://us-central1-sensors-efc67.cloudfunctions.net/train')
    response = urllib2.urlopen(req)
    return response.read()


def readCsvString(data):
   dataio = StringIO(data)
   dataFrame = pd.read_csv(dataio, sep=",")
   dataFrame.columns = ['time','gfx', 'gFy','gFz','wx','wy','wz','I']
   return dataFrame


# calssification
## ====================================================================================================================
# load model from file
import time
import pickle
from collections import Counter

def loadModel(modelFile):
    loaded_model = pickle.load(open(modelFile, "rb"))
    return loaded_model

def predict(xgbmodel,dataFrame):
    dtest = xgb.DMatrix(dataFrame)
    preds = xgbmodel.predict(dtest)
    best_preds = np.asarray([np.argmax(line) for line in preds])
    return best_preds

def trace(msg):
    if DEBUG == True :
        print (msg)

def plot(data):
    plt.clf()
    #plt.axis([-1,7,-10,10])
    plt.ion()
    plt.show()
    plt.plot(data.agyro, 'b') #data.time,
    plt.draw()
    plt.pause(0.001)

## main loop
DEBUG = False
modelFile= r'model/xgb-no-light.pickle.dat' ##  'r'model/xgb-light.pickle.dat'
rfmodelFile= r'model/rf.pickle.dat' ##  'r'model/xgb-light.pickle.dat'


print ("loading models : ",modelFile)
xgbloaded = loadModel(modelFile)
rfloaded = loadModel(rfmodelFile)

plt.ion()
plt.show()

while True :
    trace("fetch sensor data " )
    sensorData = getLatest()

    trace("convert to dataFrame ")
    rdf = readCsvString(sensorData)
    trace(str(len(rdf)) + ' sampls loaded ')

    trace("add features ")
    df = addFeatures(rdf)

    trace("select relevant features ")
    tdf = df[FEATURES]
    tdf.fillna(value=0 , axis=0, inplace=True) # method='bfill',
    # print tdf.head(100)

    plot(tdf)

    forest_val = rfloaded.predict(tdf)
    rfpredNames = [DEVICE_MODE_LABELS[x] for x in forest_val]
    rfmode = Counter(rfpredNames)
    print 'rf : ' , rfmode

    trace("predict xgb")
    pred = predict(xgbloaded,tdf)

    # convert to mode names and print counts for each predicted mode
    predNames = [DEVICE_MODE_LABELS[x] for x in pred]
    mode = Counter(predNames)
    print 'xgb: ' , mode

    # Wait for 2 seconds
    time.sleep(2.0)


print (' bye bye..')