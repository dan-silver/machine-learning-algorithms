import sys
sys.path.append("..")
from knn import NearestNeighbor
from testing import loadData, splitData, extractColumn, getAccuracy

data = loadData('iris.data')

trainSet, testSet = splitData(data, 0.66)

trainY, trainX = extractColumn(trainSet, 'class')
testY, testX = extractColumn(testSet, 'class')

predictions = NearestNeighbor().fit(trainX, trainY, k=4).predict(testX)

accuracy = getAccuracy(testY, predictions)
print('Accuracy: ' + repr(accuracy) + '%')