from __future__ import division
import sys
sys.path.append("..")
from naive_bayes import NaiveBayes
from common import *
import pandas as pd

data = loadData('iris.data')

trainSet, testSet = splitData(data, 0.66)

trainY, trainX = extractColumn(trainSet, 'class')
testY, testX = extractColumn(testSet, 'class')

cf = NaiveBayes()
cf.fit(trainX, trainY)

predictions = cf.predict(testX)

accuracy = getAccuracy(testY, predictions)
print('Accuracy: ' + repr(accuracy) + '%')