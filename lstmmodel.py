from __future__ import print_function
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import model_from_json
from keras.optimizers import Adamax , SGD
from keras import regularizers
from keras import callbacks
from keras.utils.vis_utils import plot_model

from sklearn.model_selection import train_test_split

from loadData import *

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
batch_size = 128
epochs=1000000
lrate = 0.005
beta_1=0.9
beta_2=0.999
momentum=0.9
decay=0.0
epsilon=None

model_image = 'lstm.png'

sensor = ['timestamp','gfx','gFy','gFz','wx','wy','wz']
mode   = ['devicemode']

# learning rate schedule
def step_decay(epoch):
	initial_lrate = 0.005
	drop = 0.5
	epochs_drop = 200.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate


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
    model.add(LSTM(256, dropout=0.02 , return_sequences=True,activation='relu', input_shape=(x_train.shape[1],x_train.shape[2])))
    model.add(LSTM(128 , dropout=0.02 , return_sequences=False,activation='relu'))
    model.add(Dense(4, activation='softmax')) #,kernel_regularizer=regularizers.l1(0.002)))
                                            # activity_regularizer=regularizers.l1(0.01)))

    optimizer = Adamax(lr=lrate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, decay=decay) # 'adam'
    ##optimizer = SGD(lr=lrate, momentum=momentum , decay=decay) # 'sgd'

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    print(model.summary())

    plot_model(model, to_file=model_image, show_shapes=True, show_layer_names=True)
    print('model image : {}'.format(model_image))

    print('Train...')

    # tensorboard callback :
    tbCallBack = callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=False)

    # learning schedule callback
    lratecb = callbacks.LearningRateScheduler(step_decay)

    model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          callbacks = [tbCallBack]) # ,lratecb] )

    score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)
    return model

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
import sys

if __name__ == '__main__':
    run(sys.argv)

