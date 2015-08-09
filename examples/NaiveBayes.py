import sys
sys.path.append("..")
from naive_bayes import NaiveBayes
from common import loadData, splitData, extractColumn, getAccuracy

data = loadData('iris.data')

trainSet, testSet = splitData(data, 0.66)

trainY, trainX = extractColumn(trainSet, 'class')
testY, testX = extractColumn(testSet, 'class')

cf = NaiveBayes()
predictions = cf.fit(trainX, trainY).predict(testX)

accuracy = getAccuracy(testY, predictions)
print('Accuracy: ' + repr(accuracy) + '%')