# This is a port of existing code to use numpy/pandas
# http://www.csse.monash.edu.au/courseware/cse5230/2004/assets/decisiontreesTute.pdf
# https://github.com/NinjaSteph/DecisionTree/blob/master/src/DecisionTree.py
# http://www.jdxyw.com/?p=2095

from __future__ import division
import pandas as pd
import math

class DecisionTree:
	def __init__(self, filename, outputCol):
		self.outputCol = outputCol
		self.data = pd.read_csv(filename)

	def entropy(self, dataset, targetAttribute):
		dataEntropy = 0.0
		length = len(dataset.index)

		for freq in dataset[targetAttribute].value_counts():
			dataEntropy += (-freq/length) * math.log(freq/length, 2) 

		return dataEntropy

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
			if col == self.outputCol:
				continue
			colGain = self.gain(dataset, col)
			if colGain > bestGain:
				bestGain = colGain
				feature = col
		return feature

	# @todo - make this faster
	def majority(self, dataset, targetAttribute):
		return dataset[targetAttribute].value_counts().keys()[0]

	def buildTree(self, dataset):
		vals = dataset[self.outputCol]
		defaultAttribute = self.majority(dataset, self.outputCol)

		if dataset.empty or len(dataset.columns) <= 1:
			return defaultAttribute
		elif len(pd.unique(vals)) == 1:
			return vals.values[0]
		else:
			best = self.getBestFeatureToSplitOn(dataset)
			tree = {best:{}}
			for val in dataset[best]:
				examples = dataset[dataset[best] == val]
				subtree = self.buildTree(examples)

				tree[best][val] = subtree
		return tree

	def createTree(self):
		return self.buildTree(self.data)