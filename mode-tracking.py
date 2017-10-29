import os
import pandas as pd
from pandas import rolling_median
import numpy as np

# Visualisation
import matplotlib.pyplot as plt
from detect_peaks import detect_peaks

import xgboost as xgb
import loadFiles
from const import *

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
    df = loadFiles.addFeatures(rdf)

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