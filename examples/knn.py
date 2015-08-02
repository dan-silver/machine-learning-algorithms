import sys
sys.path.append("..")
from knn import NearestNeighbor
from common import *

data = loadData('iris.data')

trainSet, testSet = splitData(data, 0.66)

trainY, trainX = extractColumn(trainSet, 'class')
testY, testX = extractColumn(testSet, 'class')

nn = NearestNeighbor()
nn.fit(trainX, trainY)

a = nn.getNeighbors(testX.iloc[2], 5)
print nn.getResponse(a)
