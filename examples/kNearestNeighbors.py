import sys
sys.path.append("..")
from knn import *
from common import *

data = loadData('iris.data')

trainSet, testSet = splitData(data, 0.66)

trainY, trainX = extractColumn(trainSet, 'class')
testY, testX = extractColumn(testSet, 'class')

nn = NearestNeighbor()
nn.fit(trainX, trainY, k=4)

# print nn.predictRow(testX.iloc[2])

predictions = nn.predict(testX)

print predictions

accuracy = getAccuracy(testY, predictions)
print('Accuracy: ' + repr(accuracy) + '%')