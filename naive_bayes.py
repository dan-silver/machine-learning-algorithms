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
			print item
			if item not in split:
				split[item] = []
			split[item].append(index)
		return split

	def predictRow(self, testX):
		pass