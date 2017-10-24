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
WINDOW_SIZE = 2 * 128  ## sliding window size
PEAKS_WINDOW_SIZE = 5*WINDOW_SIZE  ## sliding window size for peaks count feature

DEVICE_MODE_LABELS = ['pocket','swing','texting','talking','whatever']
USER_MODE_LABELS = ['walking','fastwalking','stairs','static','whatever']

FEATURES = [#'timestamp',
            'agforce','agyro',               ## avarage
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
            'peaksgforce','peaksgyro',       ## peaks count in PEAKS_WINDOW_SIZE
            'light'                          ## embient light sensor
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

def addFeatures(df):

    df['timestamp'] = df['time']

    ## norm calculations :
    df['gforce'] = np.sqrt(df['gfx']**2 + df['gFy']**2 + df['gFz']**2)
    df['gyro'] = np.sqrt(df['wx']**2 + df['wy']**2 + df['wz']**2)

   # remove outliers
    rawforce = np.sqrt(df['gfx']**2 + df['gFy']**2 + df['gFz']**2)
    df['gforce'] = RemoveOutlier(rawforce)
    rawgyro = np.sqrt(df['wx']**2 + df['wy']**2 + df['wz']**2)
    df['gyro'] =RemoveOutlier(rawgyro)

    ## calculates statistics features on rolling window :

    df['agforce'] = df['gforce'].rolling(window=WINDOW_SIZE,min_periods=1,center=False).mean()
    df['agyro']   = df['gyro'].rolling(window=WINDOW_SIZE,min_periods=1 ,center=False).mean()

    df['mgforce'] = df['gforce'].rolling(window=WINDOW_SIZE,min_periods=1,center=False).median()
    df['mgyro']   = df['gyro'].rolling(window=WINDOW_SIZE,min_periods=1 ,center=False).median()

    df['vgforce'] = df['gforce'].rolling(window=WINDOW_SIZE,min_periods=1,center=False).var()
    df['vgyro']   = df['gyro'].rolling(window=WINDOW_SIZE,min_periods=1,center=False).var()

    df['maxgforce'] = df['gforce'].rolling(window=WINDOW_SIZE,min_periods=1,center=False).max()
    df['maxgyro']   = df['gyro'].rolling(window=WINDOW_SIZE,min_periods=1 ,center=False).max()

    df['maxgforceabs'] = abs(df['maxgforce'])
    df['maxgyroabs']   = abs(df['maxgyro'])

    df['mingforce'] = df['gforce'].rolling(window=WINDOW_SIZE,min_periods=1,center=False).min()
    df['mingyro']  = df['gyro'].rolling(window=WINDOW_SIZE,min_periods=1 ,center=False).min()

    df['mingforceabs'] = abs(df['mingforce'])
    df['mingyroabs']   = abs(df['mingyro'])

    df['ampgforce'] = df['maxgforce'] - df['mingforce']
    df['ampgyro']  = df['maxgyro'] - df['mingyro']

    roll_force = df['gforce'].rolling(window=WINDOW_SIZE,min_periods=1,center=False)
    roll_gyro = df['gyro'].rolling(window=WINDOW_SIZE,min_periods=1,center=False)

    ## iqr calculations:

    q25 = roll_force.apply(RollingPercentile, [25.0])
    q75 = roll_force.apply(RollingPercentile, [75.0])
    q25 = roll_gyro.apply(RollingPercentile, [25.0])
    q75 = roll_gyro.apply(RollingPercentile, [75.0])
    df['iqrforce'] = q75 - q25
    df['iqqgyro'] = q75 - q25



    ## entropy calculations:
    df['entforce'] = roll_force.apply(RollingEntropy);
    df['entgyro'] = roll_gyro.apply(RollingEntropy);

    ## ratio calculations
    df['MultiGyroAcc'] = df['maxgforce']*df['maxgyro']

    df['skforce'] = roll_force.skew()
    df['skgyro'] = roll_gyro.skew()

    df['kuforce'] = roll_force.kurt()
    df['kugyro'] = roll_gyro.kurt()

    df['light'] = df['I'] if 'I' in df else 0.0

    df['peaksgforce'] = df['gforce'].rolling(window=PEAKS_WINDOW_SIZE,min_periods=1 ,center=False).apply(peaks)
    df['peaksgyro'] = df['gyro'].rolling(window=PEAKS_WINDOW_SIZE,min_periods=1 ,center=False).apply(peaks)

    ## gforce gyro correlation :
    df['gforcegyrocorr'] = window_correlation(df[['gforce','gyro']].values,WINDOW_SIZE)

    df.fillna(method='ffill', axis=0, inplace=True)


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

def readCsvString(data):
   dataio = StringIO(data)
   ## StringIO("""time;gfx;gFy;gFz;wx;wy;wz""")

   dataFrame = pd.read_csv(dataio, sep=",")
   dataFrame.columns = ['time','gfx', 'gFy','gFz','wx','wy','wz']
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

def predict(dataFrame):
    ## TODO : plot data
    dtest = xgb.DMatrix(dataFrame)
    preds = xgbmodel.predict(dtest)
    best_preds = np.asarray([np.argmax(line) for line in preds])
    return best_preds

def trace(msg):
    if DEBUG == True :
        print (msg)

def plot(data):
    plt.plot(data.time, data.gfx, 'b')
    plt.show()

## main loop
DEBUG = False
modelFile=r'model/xgb.pickle.dat'

print ("loading model : ",modelFile)
xgbmodel = loadModel(modelFile)

while True :
    trace("fetch sensor data " )
    sensorData = getLatest()

    trace("convert to dataFrame ")
    rdf = readCsvString(sensorData)
    trace(str(len(rdf)) + ' sampls loaded ')
    ## plot(rdf) 

    trace("add features ")
    addFeatures(rdf)

    trace("select relevant features ")
    df = rdf[FEATURES]

    trace("predict ")
    pred = predict(df)

    # convert to mode names and print counts for each predicted mode
    predNames = [DEVICE_MODE_LABELS[x] for x in pred]
    mode = Counter(predNames)
    print mode

    # Wait for 2 seconds
    time.sleep(2.0)


print (' bye bye..')