import pandas as pd
import math
import operator

def splitData(df, trainPercent):
    # https://gist.github.com/boates/5127281
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

def getAccuracy(testY, predictions):
	correct = 0
	for idx, item in enumerate(testY.values):
		if item == predictions[idx]:
			correct += 1
	return (correct/float(len(testY.values))) * 100.0

def gridSearch(classifiers, testX, testY):
	results = []
	for classifier in classifiers:
		predictions = classifier.predict(testX)
		accuracy = getAccuracy(testY, predictions)
		results.append((classifier, accuracy))

	return sorted(results, key=operator.itemgetter(1))