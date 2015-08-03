# https://gist.github.com/boates/5127281

import pandas as pd
import math
import operator

def majority(dataset, targetAttribute):
	return dataset[targetAttribute].value_counts().idxmax()

def splitData(df, trainPercent):
	"""
	return: train, test
			(as pandas dataframes)
	params:
		df: pandas dataframe
		trainPercent: float | percentage of data for trainin set (default=0.6
	"""

	# create random list of indices
	from random import shuffle
	N = len(df)
	l = range(N)
	shuffle(l)

	# get splitting indicies
	trainLen = int(N*trainPercent)

	# get train and test sets
	train = df.ix[l[:trainLen]]
	test	 = df.ix[l[trainLen:]]
	return train, test

def loadData(filename, delimeter=','):
	return pd.read_csv(filename, sep=delimeter)

def extractColumn(df, col):
	colData = df.ix[:, col]
	otherColumns = df.drop([col], axis=1)
	return colData, otherColumns

def getTrainAndTestSets(df, targetCol, trainPercent=0.66):
	train, test = splitData(df, trainPercent)

	train_label, train_features = extractColumn(train, targetCol)
	test_label, test_features = extractColumn(test, targetCol)
	return train_label, train_features, test_label, test_features

def euclideanDistance(data1, data2):
	distance = 0
	for x in range(len(data1.keys())):
		distance += pow((data1[x] - data2[x]), 2)
	return math.sqrt(distance)


# ex: stats = {'a':1000, 'b':3000, 'c': 100}
#     returns b
def getMostCommonValue(dict):
	sortedVotes = sorted(dict.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]


def getAccuracy(testY, predictions):
	correct = 0
	for idx, item in enumerate(testY.values):
		if item == predictions[idx]:
			correct += 1
	return (correct/float(len(testY.values))) * 100.0