## train :
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

## save model :
import pickle

# local modules
import consts
import loadData
import features

# train xgboost classifier and save model to file
def trainXgboost(x_train, y_train, outputModelFile = '/dev/null'):
    trainMatrix = xgb.DMatrix(x_train, label=y_train)
    bst = xgb.train(consts.xgboostParams, trainMatrix, consts.xgboostNum_round)

    ## save model :
    pickle.dump(bst, open(outputModelFile, "wb"))
    print ('xgboost model saved to : ', outputModelFile)

# train random forest classifier and save model to file
def trainRF(x_train, y_train, outputModelFile = '/dev/null'):
    fx_train = x_train.fillna(value=0, axis=0)
    rf = RandomForestClassifier()
    rf.fit(fx_train, y_train)
    ## save model
    pickle.dump(rf, open(outputModelFile, "wb"))
    print ('RF model saved to : ', outputModelFile)

def getTrainDataFromCloud():
    trainDataString = loadData.getTrainingData()
    print ( len(trainDataString) , ' samples loaded from fire base ')
    trainData = loadData.jsonToDataframe(trainDataString)

    # add features
    trainDataFeatures = features.addFeatures(trainData, g = 9.8)
    return trainDataFeatures

def run(argv):
    if len(argv) > 1:
        trainData = loadData.loadFiles(argv[1])
    else:
        trainData = getTrainDataFromCloud()

    x_train =  trainData[consts.FEATURES]
    y_train = trainData.devicemode

    trainXgboost(x_train, y_train, r'xgb-model.dat')
    trainRF(x_train, y_train, r'rf-model.dat')

## main
import sys

if __name__ == '__main__':
    run(sys.argv)




