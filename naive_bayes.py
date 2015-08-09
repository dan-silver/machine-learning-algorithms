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


	def stats(self, df):
		# loop over class values
		stats = {}
		for classValue, rowIndices in df.iteritems():
			stats[classValue] = {}
			means = self.trainX[self.trainX.index.isin(rowIndices)].mean()

			for attr in list(self.trainX.columns.values):
				stats[classValue][attr] = {"mean":means[attr], "stdev":None}
		return stats

	# def summarize(dataset):
	# 	summaries = [(mean(attribute), stdev(attribute)) for attribute in dataset)]
	# 	del summaries[-1]
	# 	return summaries

	def predictRow(self, testX):
		pass

	def fit(self, trainX, trainY, **options):
		super(NaiveBayes, self).fit(trainX, trainY, **options)
		dataByClassValue = self.splitDataByResult()
		a = self.stats(dataByClassValue)
		import pdb; pdb.set_trace()