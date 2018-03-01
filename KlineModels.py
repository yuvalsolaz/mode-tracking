from __future__ import print_function
import numpy as np
np.random.seed(2)
import tensorflow as tf
tf.set_random_seed(2)

from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import GRU
from keras.layers import  Lambda
from keras.layers.merge import add
from keras.layers import Input
from keras.models import Model
from keras.layers import ConvLSTM2D
import numpy as np
from keras.layers import Flatten
from keras.layers import Dropout
#from keras.layers import Bidirectional
from keras.models import model_from_json
from keras.optimizers import Adamax
#from numpy.core.numeric import full
from keras import callbacks
from sklearn.model_selection import train_test_split
from loadData import *

from keras.layers.advanced_activations import LeakyReLU, PReLU










"""Convert an iterable of indices to one-hot encoded labels."""
def indices_to_one_hot(data, nb_classes):
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

def getLocalTrainingData(localFile):
    fullpath = os.path.join(os.getcwd(),localFile)
    with open(fullpath, 'r') as file:
        return file.read()

def loadSensorData(inputDir = None):
    if inputDir == None:
        data_res = getLocalTrainingData('training.json')
        return jsonToDataframe(data_res)
    return loadFiles(inputDir,add_features=False)


def toLstmFormat(data):
    x = data[sensor].as_matrix(columns=None)
    x = np.reshape(x, (x.shape[0], 1, x.shape[1]))
    y = data[mode].as_matrix(columns=None)
    y = indices_to_one_hot(y, 4)
    return x , y


# TODO : consts
epochs=10000

#sensor = ['timestamp','gfx','gFy','gFz','wx','wy','wz','gyroraw','gforceraw']
#sensor = ['timestamp','gfx','gFy','gFz','wx','wy','wz']
#sensor = ['gfx','gFy','gFz','wx','wy','wz','gyroraw','gforceraw']
sensor = ['gfx','gFy','gFz','wx','wy','wz']

mode   = ['devicemode']

import pandas as pd
import os

SAMPLE_FREQ = 50 
FILE_MARGINES = 5*SAMPLE_FREQ
MODE_LABELS = ['pocket','swing','texting','talking','whatever'] 

def RemoveOutlier (values):
    threshold = values.mean()+3*values.std()
    outlier_idx = values > threshold
    values[outlier_idx] = threshold
#    outlier_idx = values < -threshold
#    values[outlier_idx] = -threshold
    return values

def loadFile(root,file):
    data=pd.read_csv(os.path.join(root,file))
    if len(data) < 400 :
        print (' only ' , len(data) , ' samples in file ', file , ' pass ')
        return pd.DataFrame()
    
#    print('loading : ' , file) 

#    print('loading : ' , len(data) , ' samples from ', file) 
    
        ## usefull property : 
    data['source']=file  

        ## default label values in case file name not contains label  
    data['devicemodeDescription']=MODE_LABELS[-1] ## 'whatever' label 
    data['devicemode'] = len(MODE_LABELS)
   # data['timestamp'] = data['time']
    data['gfx'] = RemoveOutlier(data['gfx'])
    data['gFy'] = RemoveOutlier(data['gFy'])
    data['gFz'] = RemoveOutlier(data['gFz'])
    data['wx'] = RemoveOutlier(data['wx'])
    data['wy'] = RemoveOutlier(data['wy'])
    data['wz'] = RemoveOutlier(data['wz'])
   

#    rawgyro = np.sqrt(data['wx']**2 + data['wy']**2 + data['wz']**2) 
#    data['gyroraw'] = rawgyro
#    rawforce = np.sqrt(data['gfx']**2 + data['gFy']**2 + data['gFz']**2) 
#    data['gforceraw'] = rawforce
    #data['gyro_acc'] = rawgyro*rawforce

    
        ## search device mode label in file name and add as new properties :
    for label in MODE_LABELS:
        if label.lower() in file.lower():  
           data['devicemodeDescription']=label         ## label name 
           data['devicemode'] = MODE_LABELS.index(label)    ## label index 
           break
 
        ## crop samples from start and from the end of the file :
    margin = min(len(data) / 2 - 1 , FILE_MARGINES)
    data.drop(data.index[range(0,margin)],axis=0,inplace=True)
    data.drop(data.index[range(-margin,-1)],axis=0,inplace=True)   
        ##  print(len(data) , ' samples after cropping ' , margin , 'samples from start-end of the file  ')
    return data 


def loadFiles(inputDir):
    print ('loading files from : ' , inputDir )
    return pd.concat([loadFile(inputDir,f) for f in os.listdir(inputDir) if f.lower().endswith('.csv')])





def make_residual_lstm_layers(input, rnn_width, rnn_depth, rnn_dropout):
    """
    The intermediate LSTM layers return sequences, while the last returns a single element.
    The input is also a sequence. In order to match the shape of input and output of the LSTM
    to sum them we can do it only for all layers but the last.
    """
    x = input
    for i in range(rnn_depth):
        return_sequences = i < rnn_depth - 1
        x_rnn = LSTM(rnn_width, recurrent_dropout=rnn_dropout, dropout=rnn_dropout, return_sequences=return_sequences)(x)
        if return_sequences:
            # Intermediate layers return sequences, input is also a sequence.
            if i > 0 or input.shape[-1] == rnn_width:
                x = add([x, x_rnn])
            else:
                # Note that the input size and RNN output has to match, due to the sum operation.
                # If we want different rnn_width, we'd have to perform the sum from layer 2 on.
                x = x_rnn
        else:
            # Last layer does not return sequences, just the last element
            # so we select only the last element of the previous output.
            def slice_last(x):
                return x[..., -1, :]
            x = add([Lambda(slice_last)(x), x_rnn])
            
#    x = add([Dense(4, activation='softmax'), x_rnn])        
    return x









def runLSTM(x_train, y_train, x_test , y_test , batch_size, factor1, dpValue, lrValue):
    print('Load train data from : {}'.format(trainSource if trainSource != None else ' cloud ' ))

    class LossHistory(callbacks.Callback):
          def on_train_begin(self, logs={}):
              self.losses = []
          def on_batch_end(self, batch, logs={}):
              self.losses.append(logs.get('loss'))
   
   
#    data = loadFiles(trainSource)
#    testData = loadFiles(testSource)
#    x_train, y_train = toLstmFormat(data)
#    x_test , y_test  = toLstmFormat(testData)
    

    
    print('Build model...')
    model = Sequential()

    model.add(LSTM(32*factor1, dropout=dpValue , return_sequences=True,  activation='relu', input_shape=(x_train.shape[1],x_train.shape[2])))
#    model.add(LSTM(64, dropout=0.3 , return_sequences=True, activation='relu'))
    model.add(LSTM(32, dropout=0.1 , return_sequences=False, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    optimizer = Adamax(lr=lrValue, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0) # 'adam'
    
#    from keras.optimizers import SGD
#    optimizer = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
#    from keras.optimizers import Nadam
#    optimizer = Nadam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.01)

    
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    print(model.summary())

    print('Train...')

   # tbCallBack = callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=False)
    aCallback = callbacks.EarlyStopping(monitor='val_acc',
                              min_delta=0,
                              patience=5,
                              verbose=0, mode='auto')
    
#    mcCallback = callbacks.ModelCheckpoint(os.path, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
#    loggerCallback = callbacks.CSVLogger(os.path.join(".", "training.csv"))
#    historyCallback = LossHistory()

    model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          callbacks = [aCallback] )
    
    BestEpochNumber = aCallback.stopped_epoch+1 - aCallback.patience
    BestEpochNumberAcc = aCallback.best
    
#    score, acc = model.evaluate(x_test, y_test,
#                            batch_size=batch_size)
#    print('Test score:', score)
#    print('Test accuracy:', acc)
    return model , BestEpochNumber, BestEpochNumberAcc

def runLSTM2(x_train, y_train, x_test , y_test , batch_size, factor1, dpValue, lrValue, factor2, dpValue2):
    #print('Load train data from : {}'.format(trainSource if trainSource != None else ' cloud ' ))

    class LossHistory(callbacks.Callback):
          def on_train_begin(self, logs={}):
              self.losses = []
          def on_batch_end(self, batch, logs={}):
              self.losses.append(logs.get('loss'))
   
#   
#    data = loadFiles(trainSource)
#    testData = loadFiles(testSource)
#    x_train, y_train = toLstmFormat(data)
#    x_test , y_test  = toLstmFormat(testData)
    

    
    print('Build model...')
    model = Sequential()

#    model.add(LSTM(32*factor1, dropout=dpValue , return_sequences=True,  activation='relu', input_shape=(x_train.shape[1],x_train.shape[2])))
#    model.add(LSTM(32*factor2, dropout=dpValue2 , return_sequences=False, activation='sigmoid'))
#    model.add(Dense(4, activation='softmax'))



    model.add(LSTM(factor1, dropout=dpValue , return_sequences=True,  activation='linear', input_shape=(x_train.shape[1],x_train.shape[2])))
    act = PReLU(alpha_initializer='zeros', weights=None)
    # act = LeakyReLU(alpha=0.3)
    model.add(act)
    model.add(LSTM(factor2, dropout=dpValue2 , return_sequences=False, activation='sigmoid'))
    model.add(Dense(4, activation='softmax'))






    optimizer = Adamax(lr=lrValue, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0) # 'adam'
    
#    from keras.optimizers import SGD
#    optimizer = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
#    from keras.optimizers import Nadam
#    optimizer = Nadam(lr=lrValue, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.01)

    
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    print(model.summary())

    print('Train...')

   # tbCallBack = callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=False)
    aCallback = callbacks.EarlyStopping(monitor='val_acc',
                              min_delta=0,
                              patience=5,
                              verbose=0, mode='auto')
    
#    mcCallback = callbacks.ModelCheckpoint(os.path, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
#    loggerCallback = callbacks.CSVLogger(os.path.join(".", "training.csv"))
#    historyCallback = LossHistory()

    model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          callbacks = [aCallback] )
    
    BestEpochNumber = aCallback.stopped_epoch+1 - aCallback.patience
    BestEpochNumberAcc = aCallback.best
    
#    score, acc = model.evaluate(x_test, y_test,
#                            batch_size=batch_size)
#    print('Test score:', score)
#    print('Test accuracy:', acc)
    return model , BestEpochNumber, BestEpochNumberAcc





def runGRU(x_train, y_train, x_test , y_test , batch_size, factor1, dpValue, lrValue, factor2, dpValue2):
    #print('Load train data from : {}'.format(trainSource if trainSource != None else ' cloud ' ))

    class LossHistory(callbacks.Callback):
          def on_train_begin(self, logs={}):
              self.losses = []
          def on_batch_end(self, batch, logs={}):
              self.losses.append(logs.get('loss'))
   
#   
#    data = loadFiles(trainSource)
#    testData = loadFiles(testSource)
#    x_train, y_train = toLstmFormat(data)
#    x_test , y_test  = toLstmFormat(testData)
#    

    
    print('Build model...')
    model = Sequential()

#    model.add(LSTM(32*factor1, dropout=dpValue , return_sequences=True,  activation='relu', input_shape=(x_train.shape[1],x_train.shape[2])))
#    model.add(LSTM(32*factor2, dropout=dpValue2 , return_sequences=False, activation='sigmoid'))
#    model.add(Dense(4, activation='softmax'))



#, stateful=True

    model.add(GRU(factor1, dropout=dpValue , return_sequences=True,  activation='linear', input_shape=(x_train.shape[1],x_train.shape[2])))
    act = PReLU(alpha_initializer='zeros', weights=None)
    # act = LeakyReLU(alpha=0.3)
    model.add(act)
    model.add(GRU(factor2, dropout=dpValue2 , return_sequences=False, activation='sigmoid'))
    model.add(Dense(4, activation='softmax'))






    optimizer = Adamax(lr=lrValue, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0) # 'adam'
    
#    from keras.optimizers import SGD
#    optimizer = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
#    from keras.optimizers import Nadam
#    optimizer = Nadam(lr=lrValue, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.01)

    
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    print(model.summary())

    print('Train...')

   # tbCallBack = callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=False)
    aCallback = callbacks.EarlyStopping(monitor='val_acc',
                              min_delta=0,
                              patience=5,
                              verbose=0, mode='auto')
    
#    mcCallback = callbacks.ModelCheckpoint(os.path, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
#    loggerCallback = callbacks.CSVLogger(os.path.join(".", "training.csv"))
#    historyCallback = LossHistory()

    model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          callbacks = [aCallback] )
    
    BestEpochNumber = aCallback.stopped_epoch+1 - aCallback.patience
    BestEpochNumberAcc = aCallback.best
    
#    score, acc = model.evaluate(x_test, y_test,
#                            batch_size=batch_size)
#    print('Test score:', score)
#    print('Test accuracy:', acc)
    return model , BestEpochNumber, BestEpochNumberAcc


def runResLSTM(x_train, y_train, x_test , y_test , batch_size, factor1, dpValue, lrValue, factor2, dpValue2):

    class LossHistory(callbacks.Callback):
          def on_train_begin(self, logs={}):
              self.losses = []
          def on_batch_end(self, batch, logs={}):
              self.losses.append(logs.get('loss'))  
    print('Build model...')
    input = Input(shape=(1, 8))
    output = make_residual_lstm_layers(input, rnn_width=4, rnn_depth=6, rnn_dropout=dpValue)
    model = Sequential()
    model = Model(inputs=input, outputs=output)
    model.add(Dense(4, activation='softmax'))
    model.summary()
    
    optimizer = Adamax(lr=lrValue, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0) # 'adam'
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    print(model.summary())
    print('Train...')
    aCallback = callbacks.EarlyStopping(monitor='val_acc',
                              min_delta=0,
                              patience=5,
                              verbose=0, mode='auto')
    model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          callbacks = [aCallback] )
    BestEpochNumber = aCallback.stopped_epoch+1 - aCallback.patience
    BestEpochNumberAcc = aCallback.best
    
    return model , BestEpochNumber, BestEpochNumberAcc


# save model as JSON , save weights as hdf5
def saveModel(model,outDir):
    model_json = model.to_json()
    json_file_name = os.path.join(outDir ,"model.json")
    with open(json_file_name, "w") as json_file:
        print("Save model to {}".format(json_file_name))
        json_file.write(model_json)

    # serialize weights to HDF5
    weights_file_name = os.path.join(outDir, "model.h5")
    print("Save weights to {}".format(weights_file_name))
    model.save_weights(weights_file_name)

# load json and create model
def loadModel(inputDir):
    json_file_name = os.path.join(inputDir ,"model.json")
    print("Load model from {}".format(json_file_name))
    json_file = open(json_file_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    weights_file_name = os.path.join(inputDir, "model.h5")
    print("Load weights from {}".format(weights_file_name))
    loaded_model.load_weights(weights_file_name)
    return loaded_model

def run(argv):
    if len(argv) == 3:
        model = runLSTM(argv[1], argv[2])
        return

    if len(argv) == 2 and argv[1] == 'cloud':
        runLSTM(None, None)
        return

    print ('usage:')
    print ('for data from files: python ' , argv[0] , ' <train directory> <test directory>')
    print ('for data from cloud: python ' , argv[0] , ' cloud ')


## main
trainSource = r'C:\Users\itzikkl.RF\Desktop\LSTM_SmartphoneMode\mode-tracking-master\utf8itzNewWithLight3'
testSource =  r'C:\Users\itzikkl.RF\Desktop\LSTM_SmartphoneMode\mode-tracking-master\ValForAccGyroNoStairs'
#tensorboard(r'C:\Users\itzikkl.RF\Desktop\LSTM_SmartphoneMode\mode-tracking-master\Graph')
#batch_size = 128
#factor1 = 1
#ModelOut, acc = runLSTM(trainSource, testSource, batch_size, factor1)

#
### step 1
#batch_size = 128
#factor1 = 1
#dpvValue_vec=[0,0.1,0.2,0.3]
#lrValue_vec= [0.0005,0.001, 0.0015,0.002,0.003,0.005,0.007,0.01] 
#
#BestEpochNumberMat = np.zeros(shape=(len(dpvValue_vec),len(lrValue_vec)))
#BestEpochNumberAccMat = np.zeros(shape=(len(dpvValue_vec),len(lrValue_vec)))
#
#for kk in range(0,len(dpvValue_vec)):
#    for jj in range(0,len(lrValue_vec)):
#        dpValue = dpvValue_vec[kk]
#        lrValue = lrValue_vec[jj]
#        ModelOut, BestEpochNumber, BestEpochNumberAcc = runLSTM(trainSource, testSource, batch_size, factor1,  dpValue, lrValue)
#        BestEpochNumberMat[kk,jj] = BestEpochNumber
#        BestEpochNumberAccMat[kk,jj] = BestEpochNumberAcc
#        print('Test:',dpvValue_vec[kk],lrValue_vec[jj], 'Best Acc:', BestEpochNumberAcc)
#        print(BestEpochNumberAccMat)
#i_ind,j_ind = np.unravel_index(BestEpochNumberAccMat.argmax(), BestEpochNumberAccMat.shape)
#print('LSTM Max Acc:',BestEpochNumberAccMat.max(),' index:',i_ind,j_ind)

## step 2
#batch_size = 256
#dpValue = 0.2
#lrValue_vec= [0.0005,0.001, 0.0015,0.002,0.003,0.005,0.007,0.01] 
#factor1_vec = [2,4,8,16]
#
#BestEpochNumberMat = np.zeros(shape=(len(factor1_vec),len(lrValue_vec)))
#BestEpochNumberAccMat = np.zeros(shape=(len(factor1_vec),len(lrValue_vec)))
#
#for kk in range(0,len(factor1_vec)):
#    for jj in range(0,len(lrValue_vec)):
#        factor1 = factor1_vec[kk]
#        lrValue = lrValue_vec[jj]
#        ModelOut, BestEpochNumber, BestEpochNumberAcc = runLSTM(trainSource, testSource, batch_size, factor1,  dpValue, lrValue)
#        BestEpochNumberMat[kk,jj] = BestEpochNumber
#        BestEpochNumberAccMat[kk,jj] = BestEpochNumberAcc
#        print('Test:',factor1_vec[kk],lrValue_vec[jj], 'Best Acc:', BestEpochNumberAcc)
#        print(BestEpochNumberAccMat)
#i_ind,j_ind = np.unravel_index(BestEpochNumberAccMat.argmax(), BestEpochNumberAccMat.shape)
#print('LSTM Max Acc:',BestEpochNumberAccMat.max(),' index:',i_ind,j_ind)




### step 3 - use runLSTM2 to work on second layer
#batch_size = 256
#dpValue = 0.2
#factor1 = 1
#dpValue2_vec = [0, 0.1, 0.2, 0.3, 0.4]
#lrValue_vec= [0.0005,0.001, 0.0015,0.002,0.003,0.005,0.007,0.01] 
#factor2  = 1
#
#BestEpochNumberMat = np.zeros(shape=(len(dpValue2_vec),len(lrValue_vec)))
#BestEpochNumberAccMat = np.zeros(shape=(len(dpValue2_vec),len(lrValue_vec)))
#
#for kk in range(0,len(dpValue2_vec)):
#    for jj in range(0,len(lrValue_vec)):
#        dpValue2 = dpValue2_vec[kk]
#        lrValue = lrValue_vec[jj]
#        ModelOut, BestEpochNumber, BestEpochNumberAcc = runLSTM2(trainSource, testSource, batch_size, factor1,  dpValue, lrValue, factor2,  dpValue2)
#        BestEpochNumberMat[kk,jj] = BestEpochNumber
#        BestEpochNumberAccMat[kk,jj] = BestEpochNumberAcc
#        print('Test:',dpValue2_vec[kk],lrValue_vec[jj], 'Best Acc:', BestEpochNumberAcc)
#        print(BestEpochNumberAccMat)
#i_ind,j_ind = np.unravel_index(BestEpochNumberAccMat.argmax(), BestEpochNumberAccMat.shape)
#print('LSTM Max Acc:',BestEpochNumberAccMat.max(),' index:',i_ind,j_ind)



### best workinhg case
### step 4 - use runLSTM2 to work on second layer
#batch_size = 256
#dpValue = 0.2
#factor1 = 1
#dpValue2_vec = [0.3]
#lrValue_vec= [0.005] 
#factor2  = 1
#
#BestEpochNumberMat = np.zeros(shape=(len(dpValue2_vec),len(lrValue_vec)))
#BestEpochNumberAccMat = np.zeros(shape=(len(dpValue2_vec),len(lrValue_vec)))
#
#for kk in range(0,len(dpValue2_vec)):
#    for jj in range(0,len(lrValue_vec)):
#        dpValue2 = dpValue2_vec[kk]
#        lrValue = lrValue_vec[jj]
#        ModelOut, BestEpochNumber, BestEpochNumberAcc = runLSTM2(trainSource, testSource, batch_size, factor1,  dpValue, lrValue, factor2,  dpValue2)
#        BestEpochNumberMat[kk,jj] = BestEpochNumber
#        BestEpochNumberAccMat[kk,jj] = BestEpochNumberAcc
#        print('Test:',dpValue2_vec[kk],lrValue_vec[jj], 'Best Acc:', BestEpochNumberAcc)
#        print(BestEpochNumberAccMat)
#i_ind,j_ind = np.unravel_index(BestEpochNumberAccMat.argmax(), BestEpochNumberAccMat.shape)
#print('LSTM Max Acc:',BestEpochNumberAccMat.max(),' index:',i_ind,j_ind)
#



#
#batch_size = 256
#dpValue = 0.2
#factor1 = 1
#dpValue2_vec = [0, 0.1, 0.2, 0.25, 0.3, 0.4]
#lrValue_vec= [0.001,0.003,0.005,0.007,0.01] 
#factor2  = 1
#
#BestEpochNumberMat = np.zeros(shape=(len(dpValue2_vec),len(lrValue_vec)))
#BestEpochNumberAccMat = np.zeros(shape=(len(dpValue2_vec),len(lrValue_vec)))
#
#for kk in range(0,len(dpValue2_vec)):
#    for jj in range(0,len(lrValue_vec)):
#        dpValue2 = dpValue2_vec[kk]
#        lrValue = lrValue_vec[jj]
#        ModelOut, BestEpochNumber, BestEpochNumberAcc = runLSTM2(trainSource, testSource, batch_size, factor1,  dpValue, lrValue, factor2,  dpValue2)
#        BestEpochNumberMat[kk,jj] = BestEpochNumber
#        BestEpochNumberAccMat[kk,jj] = BestEpochNumberAcc
#        print('Test:',dpValue2_vec[kk],lrValue_vec[jj], 'Best Acc:', BestEpochNumberAcc)
#        print(BestEpochNumberAccMat)
#i_ind,j_ind = np.unravel_index(BestEpochNumberAccMat.argmax(), BestEpochNumberAccMat.shape)
#print('LSTM Max Acc:',BestEpochNumberAccMat.max(),' index:',i_ind,j_ind)
#
#
#
#
## MC runs for best model 
## parameters were found using random seed = 42
#batch_size = 256
#dpValue = 0.2
#factor1 = 1
#dpValue2 = 0.25
#lrValue= 0.005 
#factor2  = 1
#MC_runs = 10;
#AccRes = np.zeros(shape=(MC_runs,1))
#
#for kk in range(0,MC_runs):
#    ModelOut, BestEpochNumber, BestEpochNumberAcc = runLSTM2(trainSource, testSource, batch_size, factor1,  dpValue, lrValue, factor2,  dpValue2)
#    AccRes[kk] = BestEpochNumberAcc
#print(AccRes.mean())














# load data once instaed in LSTM or GRU
data = loadFiles(trainSource)
testData = loadFiles(testSource)
x_train, y_train = toLstmFormat(data)
x_test , y_test  = toLstmFormat(testData)

#
###################################
##check GRU performance and LSTM performance with a specific random seed
 #GRU example
#batch_size = 256
#dpValue = 0.2
#factor1 = 32
#dpValue2 = 0.25
#lrValue= 0.005 
#factor2  = 32
#ModelOut, BestEpochNumber, BestEpochNumberAcc = runGRU(x_train, y_train, x_test , y_test , batch_size, factor1,  dpValue, lrValue, factor2,  dpValue2)
#print(BestEpochNumberAcc)


#
#
###################################
## check Residual LSTM performance with a specific random seed
#batch_size = 256
#dpValue = 0.2
#factor1 = 1
#dpValue2 = 0.25
#lrValue= 0.005 
#factor2  = 1
#ModelOut, BestEpochNumber, BestEpochNumberAcc = runResLSTM(x_train, y_train, x_test , y_test , batch_size, factor1,  dpValue, lrValue, factor2,  dpValue2)
#

######################################
# Test CNN
window_size = 32
batch_size = 256
lrValue= 0.005 

padding='causal'
""" results in causal(dilated) convolutions, e.g.output[t] does not depend on input[t + 1:].
    Useful when modeling temporal data where the model should not violate the temporal order.
"""
def toCnnFormat(data, window_size=window_size):
    assert 0 < window_size < data.shape[0]
    xdata = np.asarray(data[sensor])
    ydata = np.asarray(data[mode])
    ydata = indices_to_one_hot(ydata, 4)

    x = np.array \
        ([xdata[start:start + window_size] for start in range(0, xdata.shape[0] - window_size)]) # np.atleast_3d(

    y = ydata[:len(x)]

    return x, y
def runCNN(data, testData):
    print('Load train data from : {}'.format(trainSource if trainSource != None else ' cloud ' ))
    x_train, y_train = toCnnFormat(data)
    x_test , y_test = toCnnFormat(testData)

    print('Build model...')
    model = Sequential()
    model.add(Conv1D(64, 3, padding=padding, input_shape=(x_train.shape[1], x_train.shape[2]), activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(32, 3, padding=padding, activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(64, 3, padding=padding, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(32, 3, padding=padding, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(16, 3, padding=padding, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(4, activation='softmax'))

  

    optimizer = Adamax(lr=lrValue, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)  # 'adam'


    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    print(model.summary())

   

    print('Train...')

    aCallback = callbacks.EarlyStopping(monitor='val_acc',
                              min_delta=0,
                              patience=5,
                              verbose=0, mode='auto')
    model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          callbacks = [aCallback] )
    BestEpochNumber = aCallback.stopped_epoch+1 - aCallback.patience
    BestEpochNumberAcc = aCallback.best
    
    return model , BestEpochNumber, BestEpochNumberAcc

#model , BestEpochNumber, BestEpochNumberAcc = runCNN(data, testData)
#####################################
batch_size = 256
dpValue = 0.2
#factor1_vec = [8, 16, 32, 48]
#dpValue2_vec = [0.1, 0.2, 0.25, 0.3, 0.4]
#lrValue_vec= [0.001,0.003,0.005,0.007,0.01] 

factor1_vec = [16, 32, 48, 64]
dpValue2_vec = [ 0.2, 0.25, 0.3]
lrValue_vec= [0.001,0.003,0.005,0.007,0.01] 



BestEpochNumberMat = np.zeros(shape=(len(factor1_vec),len(lrValue_vec),len(dpValue2_vec)))
BestEpochNumberAccMat = np.zeros(shape=(len(factor1_vec),len(lrValue_vec),len(dpValue2_vec)))
BestEpochNumberMatGRU = np.zeros(shape=(len(factor1_vec),len(lrValue_vec),len(dpValue2_vec)))
BestEpochNumberAccMatGRU = np.zeros(shape=(len(factor1_vec),len(lrValue_vec),len(dpValue2_vec)))

for kk in range(0,len(factor1_vec)):
    for jj in range(0,len(lrValue_vec)):
        for mm in range(0,len(dpValue2_vec)):
            factor1 = factor1_vec[kk]
            factor2 = factor1
            dpValue2 = dpValue2_vec[mm]
            lrValue = lrValue_vec[jj]
            ModelOut, BestEpochNumber, BestEpochNumberAcc = runLSTM2(x_train, y_train, x_test , y_test , batch_size, factor1,  dpValue, lrValue, factor2,  dpValue2)
            BestEpochNumberMat[kk,jj,mm] = BestEpochNumber
            BestEpochNumberAccMat[kk,jj,mm] = BestEpochNumberAcc
            print('LSTM Test:',factor1_vec[kk],lrValue_vec[jj],dpValue2_vec[mm], 'Best Acc:', BestEpochNumberAcc)
            print(BestEpochNumberAccMat)
            ModelOutGRU, BestEpochNumberGRU, BestEpochNumberAccGRU = runGRU(x_train, y_train, x_test , y_test , batch_size, factor1,  dpValue, lrValue, factor2,  dpValue2)
            BestEpochNumberMatGRU[kk,jj,mm] = BestEpochNumberGRU
            BestEpochNumberAccMatGRU[kk,jj,mm] = BestEpochNumberAccGRU
            print('GRU Test:',factor1_vec[kk],lrValue_vec[jj],dpValue2_vec[mm], 'Best Acc:', BestEpochNumberAccGRU)
            print(BestEpochNumberAccMatGRU)
i_ind,j_ind,k_ind = np.unravel_index(BestEpochNumberAccMat.argmax(), BestEpochNumberAccMat.shape)
print('LSTM Max Acc:',BestEpochNumberAccMat.max(),' index:',i_ind,j_ind,k_ind)
i_ind,j_ind,k_ind = np.unravel_index(BestEpochNumberAccMatGRU.argmax(), BestEpochNumberAccMatGRU.shape)
print('GRU Max Acc:',BestEpochNumberAccMatGRU.max(),' index:',i_ind,j_ind,k_ind)



model , BestEpochNumber, BestEpochNumberAccCNN = runCNN(data, testData)

