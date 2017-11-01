# local modules
import loadData

## usage python export <output csv file>>

def run(argv):
    if len(argv) < 2:
        print ('usage : python ' , argv[0] , ' <output csv file>')
        exit()

    trainDataString = loadData.getTrainingData()
    trainData = loadData.jsonToDataframe(trainDataString)
    distribute = trainData.groupby('devicemodeDescription').devicemode.count()

    print (distribute) #, ' ', distribute * 100.0 / len(distribute))

    trainData.to_csv(argv[1])
    print ('data saved to : ' , argv[1])

## main
import sys

if __name__ == '__main__':
    run(sys.argv)