import sys
sys.path.append("..")

from knn import NearestNeighbor
from naive_bayes import NaiveBayes
from common import loadData, splitData, extractColumn, gridSearch
import operator

data = loadData('iris.data')
trainSet, testSet = splitData(data, 0.66)
trainY, trainX = extractColumn(trainSet, 'class')
testY, testX = extractColumn(testSet, 'class')

classifiers = [
	NearestNeighbor().fit(trainX, trainY, k=4),
	NaiveBayes().fit(trainX, trainY)
]

for result in gridSearch(classifiers, testX, testY):
	print(str(result[0]) + ' ' + repr(result[1]) + '% accurate')