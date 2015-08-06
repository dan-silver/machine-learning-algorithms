import sys
sys.path.append("..")
from knn import NearestNeighbor
from common import *

data = loadData('iris.data')

trainSet, testSet = splitData(data, 0.66)

trainY, trainX = extractColumn(trainSet, 'class')
testY, testX = extractColumn(testSet, 'class')

nn = NearestNeighbor()
nn.fit(trainX, trainY, k=4)

predictions = nn.predict(testX)

print predictions

accuracy = getAccuracy(testY, predictions)
print('Accuracy: ' + repr(accuracy) + '%')