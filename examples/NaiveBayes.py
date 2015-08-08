from __future__ import division
import sys
sys.path.append("..")
from naive_bayes import NaiveBayes
from common import *
import pandas as pd

data = loadData('iris.data')

data = pd.DataFrame({
	'a':[1,2,3],
	'b':[20,21,22],
	'class':[1,0,0]
})
# print data
trainSet, testSet = splitData(data, 1)

trainY, trainX = extractColumn(trainSet, 'class')
testY, testX = extractColumn(testSet, 'class')

cf = NaiveBayes()
cf.fit(trainX, trainY)