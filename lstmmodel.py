from __future__ import print_function
import pandas as pd
import os
import sys
import json
import urllib.request as urllib2
from io import StringIO
import consts
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.layers import Bidirectional
from sklearn.model_selection import train_test_split
import numpy as np
import features
from scipy import stats
from loadData import *

"""Convert an iterable of indices to one-hot encoded labels."""
def indices_to_one_hot(data, nb_classes):
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

def loadSensorData(inputDir = None):
    if inputDir == None:
        data_res = getTrainingData()
        return jsonToDataframe(data_res)
    return loadFiles(inputDir,add_features=False)


def toLstmFormat(data):
    x = data[sensor].as_matrix(columns=None)
    x = np.reshape(x, (x.shape[0], 1, x.shape[1]))
    y = data[mode].as_matrix(columns=None)
    y = indices_to_one_hot(y, 4)
    return x , y


# TODO : consts
batch_size = 64
epochs=3

sensor = ['timestamp','gfx','gFy','gFz','wx','wy','wz']
mode   = ['devicemode']


def runLSTM(trainSource, testSource):
    print('Load train data from : {}'.format(trainSource if trainSource != None else ' cloud ' ))

    data = loadSensorData(trainSource)

    if testSource != None :
        print('Load test data from : {}'.format(testSource))
        testData = loadSensorData(testSource)
        x_train, y_train = toLstmFormat(data)
        x_test , y_test  = toLstmFormat(testData)
    else :
        print('split loaded data to train and test : ')
        train, test = train_test_split(data, test_size=0.2)
        x_train, y_train = toLstmFormat(train)
        x_test , y_test = toLstmFormat(test)

    print('Build model...')
    model = Sequential()
    model.add(LSTM(512, input_shape=(x_train.shape[1], x_train.shape[2]),dropout=0.2,return_sequences=True,activation='relu'))
    model.add(LSTM(input_dim=512, output_dim=128,dropout=0.02,return_sequences=False,activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Train...')
    model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)


def run(argv):
    if len(argv) == 3:
        runLSTM(argv[1], argv[2])
        return

    if len(argv) == 2 and argv[1] == 'cloud':
        runLSTM(None, None)
        return

    print ('usage: python ' , argv[0] , ' <train directory> <test directory>')


## main
import sys

if __name__ == '__main__':
    run(sys.argv)

