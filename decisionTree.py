# This is a port of existing code to use pandas (http://pandas.pydata.org/)
# http://www.csse.monash.edu.au/courseware/cse5230/2004/assets/decisiontreesTute.pdf
# https://github.com/NinjaSteph/DecisionTree/blob/master/src/DecisionTree.py
# http://www.jdxyw.com/?p=2095
# https://en.wikipedia.org/wiki/ID3_algorithm

from __future__ import division
import pandas as pd
import math
from model import Model

class DecisionTree(Model):
	def entropy(self, dataset, targetAttribute):
		e = 0.0
		length = len(dataset.index)

		for freq in dataset[targetAttribute].value_counts():
			e += (-freq/length) * math.log(freq/length, 2) 

		return e

	def gain(self, dataset, targetAttribute):
		valueFrequencies = dataset[targetAttribute].value_counts()

		total_value = sum(valueFrequencies.values)
		subsetEntropy = 0.0

		for key in valueFrequencies.keys():
			probability = valueFrequencies[key] / total_value
			subset = dataset[dataset[targetAttribute] == key]
			subsetEntropy += probability * self.entropy(subset, targetAttribute)

		return self.entropy(dataset, targetAttribute) - subsetEntropy


	def getBestFeatureToSplitOn(self, dataset):
		feature = dataset.columns[0]
		bestGain = 0
		for col in dataset.columns:
			colGain = self.gain(dataset, col)
			if colGain > bestGain:
				bestGain = colGain
				feature = col
		return feature

	def buildTree(self, trainX, trainY):
		if trainX.empty or len(trainX.columns) <= 1:
			return self.majority(trainX, self.outputCol)
		elif len(pd.unique(trainY)) == 1:
			return trainY.values[0]
		else:
			best = self.getBestFeatureToSplitOn(trainX)
			tree = {best:{}}
			for val in trainX[best]:
				# todo - select the corresponding values in trainY
				examplesX = trainX[trainX[best] == val]
				examplesX = examplesX.drop(best, 1)
				subtree = self.buildTree(examplesX)

				tree[best][val] = subtree
		self.model = tree

	def predictRow(self, row):
		pass