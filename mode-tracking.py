# load data :
import loadData

# feature engineering :
from consts import *
import features

## prediction :
import numpy as np
import xgboost as xgb

# Visualisation :
import matplotlib.pyplot as plt



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
    sensorData = loadData.getLatest()

    trace("convert to dataFrame ")
    rdf = loadData.csvStringToDataframe(sensorData)
    trace(str(len(rdf)) + ' sampls loaded ')

    trace("add features ")
    df = features.addFeatures(rdf)

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