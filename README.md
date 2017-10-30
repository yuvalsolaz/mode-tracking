
mode-tracking : train test and online classifier for device mode classification

main utilities :
--------------------------------------------------------------------------------
train.py :
    1. train data from all csv files in the provided input directory
    2. if no input directory provided train on labeled data from the cloud
    3. save xgboost and random forest models to local files

    usage : python train.py <optional : input directory>

test.py :
    1. load provided model file
    2. run test on provided validation directory with csv files
    3. print accuracy report and confusion matrix

 mode-tracking.py :
    1. classify in real time data from the device application
    2. use loaded xgboost and random forest models from training


main modules :
--------------------------------------------------------------------------------
consts.py - window size , classifier parameters , features combination , etc

detect_peaks.py - count peaks in a given data window

features.py - all features calculation methods

loadData.py - data loading from files or cloud

mode-tracking - online classifier

data folders  :
--------------------------------------------------------------------------------
raw-data/train - folder for train data csv files

raw-data/test - folder for test data csv files

cloud api :
--------------------------------------------------------------------------------
loading all training data:
 https://us-central1-sensors-efc67.cloudfunctions.net/trainingData

delete all training data:
 https://us-central1-sensors-efc67.cloudfunctions.net/deleteTrainingData

get latest device data :
https://us-central1-sensors-efc67.cloudfunctions.net/latest















