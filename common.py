# https://gist.github.com/boates/5127281

import pandas as pd

def splitData(df, trainPerc=0.66, testPerc=0.34):
	"""
	return: train, test
			(as pandas dataframes)
	params:
		df: pandas dataframe
		trainPerc: float | percentage of data for trainin set (default=0.6
		testPerc: float | percentage of data for test set (default=0.2)
	"""
	assert trainPerc + testPerc == 1.0
 
	# create random list of indices
	from random import shuffle
	N = len(df)
	l = range(N)
	shuffle(l)
 
	# get splitting indicies
	trainLen = int(N*trainPerc)
 
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

def getTrainAndTestSets(df, targetCol, trainPerc=None, testPerc=None):
	train, test = splitData(df)

	trainResult, trainData = extractColumn(train, targetCol)
	testResult, testData = extractColumn(test, targetCol)
	return trainResult, trainData, testResult, testData