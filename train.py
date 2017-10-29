import loadFiles

## consts :
train_data = r'raw-data/'
test_data = r'val-data/'

# train = loadFiles.loadFiles(train_data)
# test = loadFiles.loadFiles(test_data)

##  TODO : train xgb & rf and save model files

## TODO : load train from fire base
trainurl = loadFiles.loadFirebase()
print ( len(trainurl) , ' samples loaded from fire base ')