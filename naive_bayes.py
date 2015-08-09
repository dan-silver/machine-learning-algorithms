# Converted the following to use pandas and follow the common api
# http://machinelearningmastery.com/naive-bayes-classifier-scratch-python/

from __future__ import division
import pandas as pd
import operator
from model import Model

class NaiveBayes(Model):

	# Given self.trainX, return a hash that has the unique trainY values with the indices of the coressponding trainX values
	def splitDataByResult(self):
		split = {}
		for index, item in self.trainY.iteritems():
			if item not in split:
				split[item] = []
			split[item].append(index)
		return split


	# Given a hash of the data split by class value (keys are class vals, values are row indices),
	# calc the mean and standard deviation for each attribute for each row set
	def stats(self, df):
		# loop over class values
		stats = {}
		for classValue, rowIndices in df.iteritems():
			stats[classValue] = {}
			rows  = self.trainX[self.trainX.index.isin(rowIndices)]
			means = rows.mean()
			stdev = rows.std()
			for attr, val in means.iteritems():
				stats[classValue][attr] = {"mean":val, "stdev":stdev[attr]}
		return stats

	def predictRow(self, testX):
		pass

	def fit(self, trainX, trainY, **options):
		super(NaiveBayes, self).fit(trainX, trainY, **options)
		dataByClassValue = self.splitDataByResult()
		a = self.stats(dataByClassValue)
		import pdb; pdb.set_trace()