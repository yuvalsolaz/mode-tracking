
## load model from model file and run accuracy tests on validation data  :
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import xgboost as xgb
import pickle

# local modules
import consts
import loadData

def test(x_test, y_test, modelFile, xgbModel=False):
    loaded_model = pickle.load(open(modelFile, "rb"))
    if xgbModel:
        dtest = xgb.DMatrix(x_test)
        preds = loaded_model.predict(dtest)
    else:
        preds = loaded_model.predict(x_test)
    best_preds = np.asarray([np.argmax(line) for line in preds]) if xgbModel else preds
    print (accuracy_score(y_test,best_preds))

    print(classification_report(y_pred=best_preds, y_true=y_test))
    conf_matrix = confusion_matrix(y_pred=best_preds, y_true=y_test)
    print(conf_matrix)


def run(argv):
    if len(argv) < 3:
        print ('usage: python ' , argv[0] , ' <validation data directory> <input model file>')
        return

    testData = loadData.loadFiles(argv[1])

    x_test = testData[consts.FEATURES]
    y_test = testData.devicemode

    modelFile = argv[2]
    xgbModel = True if 'xgb' in modelFile.lower() else False

    print('run accuracy tests with model : ' , modelFile , ' on data from : ' , argv[1] )

    test(x_test, y_test, modelFile , xgbModel)

## main
import sys

if __name__ == '__main__':
    run(sys.argv)