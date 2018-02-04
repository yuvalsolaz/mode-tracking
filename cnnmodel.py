
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import Conv1D
from keras.layers import MaxPooling1D

from keras.models import model_from_json
#from keras.optimizers import Adamax
#from numpy.core.numeric import full
from keras import callbacks

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



# TODO : consts
window_size = 32
batch_size = 128
epochs=1000

sensor = ['timestamp','gfx','gFy','gFz','wx','wy','wz']
mode   = ['devicemode']

def toCnnFormat(data, window_size=window_size):
    assert 0 < window_size < data.shape[0]
    xdata = data[sensor]
    ydata = data[mode]
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)

    x = np.array \
        ([xdata[start:start + window_size] for start in range(0, xdata.shape[0] - window_size)]) # np.atleast_3d(
    y = ydata[window_size:]
    return x, y


def runCNN(trainSource, testSource):
    print('Load train data from : {}'.format(trainSource if trainSource != None else ' cloud ' ))

    data = loadSensorData(trainSource)

    if testSource != None :
        print('Load test data from : {}'.format(testSource))
        testData = loadSensorData(testSource)
        x_train, y_train = toCNNFormat(data)
        x_test , y_test  = toCNNFormat(testData)
    else :
        print('split loaded data to train and test : ')
        train, test = train_test_split(data, test_size=0.2)
        x_train, y_train = toCnnFormat(train)
        x_test , y_test = toCnnFormat(test)

    print('Build model...')
    model = Sequential()
    model.add(Conv1D(10 ,3 ,input_shape=x_train.shape, activation='relu'))
    model.add(MaxPooling1D(2))  # Downsample the output of convolution by 2X.
    model.add(Conv1D(10, 3, activation='relu'))
    model.add(MaxPooling1D(2))  # Downsample the output of convolution by 2X.
    model.add(Dense(4, activation='softmax'))


    optimizer = 'adam' # Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    print(model.summary())

    print('Train...')

    tbCallBack = callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=False)
    model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          callbacks = [tbCallBack] )

    score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)
    return model




def run(argv):
    if len(argv) == 3:
        model = runCNN(argv[1], argv[2])
        return

    if len(argv) == 2 and argv[1] == 'cloud':
        runCNN(None, None)
        return

    print ('usage:')
    print ('for data from files: python ' , argv[0] , ' <train directory> <test directory>')
    print ('for data from cloud: python ' , argv[0] , ' cloud ')


## main
import sys

if __name__ == '__main__':
    run(sys.argv)

