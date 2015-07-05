# This is a port of existing code to use pandas (http://pandas.pydata.org/)
# http://www.csse.monash.edu.au/courseware/cse5230/2004/assets/decisiontreesTute.pdf
# https://github.com/NinjaSteph/DecisionTree/blob/master/src/DecisionTree.py
# http://www.jdxyw.com/?p=2095
# https://en.wikipedia.org/wiki/ID3_algorithm

from __future__ import division
import pandas as pd
import math

class Model:
	def loadCSV(self, filename, outputCol):
		self.data = pd.read_csv(filename)
		self.outputCol = outputCol

	def majority(self, dataset, targetAttribute):
		return dataset[targetAttribute].value_counts().idxmax()

	def exportModel(self):
		return self.model

	def saveTree(self):
		pass

	def loadTree(self):
		pass

	def addData(self, data):
		pass

	def predict(self, data):
		predictions = []
		for row in data:
			predictions.append(self.predictRow(row))
		return predictions

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
			if col == self.outputCol:
				continue
			colGain = self.gain(dataset, col)
			if colGain > bestGain:
				bestGain = colGain
				feature = col
		return feature

	def buildTree(self, dataset):
		vals = dataset[self.outputCol]

		if dataset.empty or len(dataset.columns) <= 1:
			return self.majority(dataset, self.outputCol)
		elif len(pd.unique(vals)) == 1:
			return vals.values[0]
		else:
			best = self.getBestFeatureToSplitOn(dataset)
			tree = {best:{}}
			for val in dataset[best]:
				examples = dataset[dataset[best] == val]
				examples = examples.drop(best, 1)
				subtree = self.buildTree(examples)

				tree[best][val] = subtree
		return tree

	def createTree(self):
		self.model = self.buildTree(self.data)

	def predictRow(self, row):
		pass