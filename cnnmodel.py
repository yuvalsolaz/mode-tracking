
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.optimizers import Adamax,SGD
from keras.layers import Dropout
#from keras import regularizers

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



# TODO : consts
whights_file = os.path.join('./Model','cnn-best-weights.hdf5')  #-{epoch:02d}-{val_acc:.2f}
model_image = 'cnn.png'

window_size = 32

padding='causal'
""" results in causal(dilated) convolutions, e.g.output[t] does not depend on input[t + 1:].
    Useful when modeling temporal data where the model should not violate the temporal order.
"""

batch_size = 128
epochs=10000
lrate = 0.02
beta_1=0.9
beta_2=0.999
momentum=0.9
decay=0.0
epsilon=None

sensor = ['timestamp','gfx','gFy','gFz','wx','wy','wz']
mode   = ['devicemode']

def indices_to_one_hot(data, nb_classes):
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

def toCnnFormat(data, window_size=window_size):
    assert 0 < window_size < data.shape[0]
    xdata = np.asarray(data[sensor])
    ydata = np.asarray(data[mode])
    ydata = indices_to_one_hot(ydata, 4)

    x = np.array \
        ([xdata[start:start + window_size] for start in range(0, xdata.shape[0] - window_size)]) # np.atleast_3d(

    y = ydata[:len(x)]

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

    # load weights TODO if exists
    if os.path.exists(whights_file):
        print('loading whights from {}'.format (whights_file))
        model.load_weights(whights_file)

    optimizer = Adamax(lr=lrate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, decay=decay)  # 'adam'
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

    # checkpoint callback :
    chkpcb = callbacks.ModelCheckpoint(whights_file, monitor='val_acc', verbose=1, save_best_only=True,save_weights_only=False, mode='max',period=1)


    model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          callbacks = [tbCallBack,chkpcb] )

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

