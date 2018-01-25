## prediction :
import numpy as np
import xgboost as xgb

# Visualisation :
import matplotlib.pyplot as plt

import pandas as pd
# load data :
import loadData

# feature engineering :
import consts
import features




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
    plt.plot(data.timestamp,data.wx,'b') #data.time,
    plt.draw()
    plt.pause(0.001)

## main loop

DEBUG = False
modelFile= r'cloud-model/xgb-model.dat' ##  'r'model/xgb-light.pickle.dat'
rfmodelFile= r'cloud-model/rf-model.dat' ##  'r'model/xgb-light.pickle.dat'

print ("loading models : ",modelFile)
xgbloaded = loadModel(modelFile)
rfloaded = loadModel(rfmodelFile)

plt.ion()
plt.show()

## save 3 windows length history :
last_samples = pd.DataFrame()

while True :
    trace("fetch sensor data " )
    sensorData = loadData.getLatest()

    trace("convert to dataFrame ")
    samples = loadData.csvStringToDataframe(sensorData.decode("ascii"))
    trace(str(len(samples)) + ' samples loaded ')

    # continue on first loop entry
    if len(last_samples) == 0:
        last_samples = samples
        continue

    ## shift sample timestamp forward to fit last_samples
    shifted_samples = samples.copy(deep=True)
    shifted_samples['timestamp'] = samples['timestamp'] + samples['timestamp'].max()

    ## combine samples with last_samples :
    combined_samples = pd.concat([last_samples,shifted_samples])
    plot(combined_samples)

    trace("add features ")
    combined_ftrs = features.addFeatures(combined_samples,g=9.8)

    trace("select relevant features ")
    selected_ftrs = combined_ftrs[consts.FEATURES]
    selected_ftrs.fillna(value=0 , axis=0, inplace=True) # method='bfill',

    ## remove boundries before prediction
    boundery = 50 ## consts.WINDOW_SIZE / 2.0
    central_selected_ftrs = selected_ftrs[boundery:-boundery]

    # predict  :
    forest_val = rfloaded.predict(central_selected_ftrs)
    rfpredNames = [consts.DEVICE_MODE_LABELS[x] for x in forest_val]
    rfmode = Counter(rfpredNames)
    print ('rf : ' , rfmode)

    trace("predict xgb")
    pred = predict(xgbloaded, central_selected_ftrs)

    # save for next time :
    last_samples = samples

    # convert to mode names and print counts for each predicted mode
    predNames = [consts.DEVICE_MODE_LABELS[x] for x in pred]
    mode = Counter(predNames)
    print('xgb: ' , mode)

    selected_mode = max(mode)

    time.sleep(2.0)


print (' bye bye..')