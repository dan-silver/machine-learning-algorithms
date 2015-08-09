# Converted the following to use pandas and follow the common api
# http://machinelearningmastery.com/naive-bayes-classifier-scratch-python/

from __future__ import division
import pandas as pd
import operator
from model import Model
import math

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

	def calculateProbability(self, x, mean, stdev):
		exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
		return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

	def calculateClassProbabilities(self, testX):
		probabilities = {}
		for classValue, classSummaries in self.summaries.iteritems():
			probabilities[classValue] = 1
			for attr, stats in classSummaries.iteritems():
				x = testX[attr]
				probabilities[classValue] *= self.calculateProbability(x, stats['mean'], stats['stdev'])
		return probabilities


	def predictRow(self, testX):
		probabilities = self.calculateClassProbabilities(testX)
		bestLabel, bestProb = None, -1
		for classValue, probability in probabilities.iteritems():
			if bestLabel is None or probability > bestProb:
				bestProb = probability
				bestLabel = classValue
		return bestLabel

	def fit(self, trainX, trainY, **options):
		super(NaiveBayes, self).fit(trainX, trainY, **options)
		dataByClassValue = self.splitDataByResult()
		self.summaries = self.stats(dataByClassValue)
		return self