# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 15:15:43 2017

"""
###############################################################################
import pandas as pd

import os
import urllib2
import sys
import json

import features

if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

import consts

## load realtime data from cloud :
##---------------------------------------------------------------------------------------------------------------------

def getLatest():
    req = urllib2.Request('https://us-central1-sensors-efc67.cloudfunctions.net/latest')
    response = urllib2.urlopen(req)
    return response.read()

## convert data from csv format string to data frmae :
def csvStringToDataframe(csvStr):
   dataio = StringIO(csvStr)
   dataFrame = pd.read_csv(dataio, sep=",")
   dataFrame.columns = ['time','gfx', 'gFy','gFz','wx','wy','wz','I']
   return dataFrame


## load training data from cloud :
##---------------------------------------------------------------------------------------------------------------------


def getTrainingData():
    req = urllib2.Request('https://us-central1-sensors-efc67.cloudfunctions.net/trainingData')
    response = urllib2.urlopen(req)
    return response.read()

## convert json to data frame :
##
def jsonToDataframe(jsonStr):
    dataFrame = pd.DataFrame()
    jsonDict = json.loads(jsonStr)
    for i in range(0,len(jsonDict)):
        rec = jsonDict[i]
        df = csvStringToDataframe(rec['data'])
        df['timestamp'] = df['time']  ## naming conventions
        df.drop('time',axis=1,inplace=True)
        modeName = rec['mode'] ## mode name
        if modeName == 'text':
            modeName = 'texting'
        df['devicemodeDescription'] = modeName
        df['devicemode'] = consts.DEVICE_MODE_LABELS.index(modeName)  ## mode index
        dataFrame = pd.concat([dataFrame,df])

    print (len(dataFrame), ' records loaded ')
    return dataFrame

## load trainig data from csv files
##--------------------------------------------------------------------------------------------------------------------

def loadFiles(inputDir):
    print ('loading data from : ' , inputDir )
    data =  pd.concat([loadFile(inputDir,f) for f in os.listdir(inputDir) if f.lower().endswith('.csv')])
    print (len(data) , ' samples loaded ')
    return data

def loadFile(root,file):
    data=pd.read_csv(os.path.join(root,file))
    if len(data) < 400 :
        print (' only ' , len(data) , ' samples in file ', file , ' pass ')
        return pd.DataFrame()

    print('loading : ' , file)
    print('loading : ' , len(data) , ' samples from ', file)

    ## usefull property :
    data['source']=file

    data['timestamp'] = data['time']  ## naming conventions
    data.drop('time',axis=1)

    ## default label values in case file name not contains label
    data['devicemodeDescription']=consts.DEVICE_MODE_LABELS[-1] ## 'whatever' label
    data['devicemode'] = len(consts.DEVICE_MODE_LABELS)

    ## search device mode label in file name and add as new properties :
    for label in consts.DEVICE_MODE_LABELS:
        if label.lower() in file.lower():
            data['devicemodeDescription']=label         ## label name
            data['devicemode'] = consts.DEVICE_MODE_LABELS.index(label)    ## label index
            break

    ## add high level features
    features.addFeatures(data)

    ## print( len(data) , ' samples loaded ')
    ## print('all records labeld as ', data['devicemodeDescription'][0])

    ## crop samples from start and from the end of the file :
    margin = min(len(data) / 2 - 1 , consts.FILE_MARGINES)
    data.drop(data.index[range(0,margin)],axis=0,inplace=True)
    data.drop(data.index[range(-margin,-1)],axis=0,inplace=True)
    ##  print(len(data) , ' samples after cropping ' , margin , 'samples from start-end of the file  ')
    return data

