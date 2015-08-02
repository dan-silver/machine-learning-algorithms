# This is a port of existing code to use pandas (http://pandas.pydata.org/)
# http://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

from __future__ import division
import pandas as pd
import operator
from model import Model
from common import euclideanDistance, getMostCommonValue

class NearestNeighbor(Model):
	def getNeighbors(self, testInstance, k):
		distances = []
		# get distance to each point
		for index, item in self.trainX.iterrows():
			dist = euclideanDistance(testInstance, item)
			distances.append((index, dist))
		distances.sort(key=operator.itemgetter(1)) #sort based on distance
		neighbors = [] #indexes
		for x in range(k):
			neighbors.append(distances[x][0])
		return neighbors #indexes

	def getResponse(self, neighbors):
		classVotes = {}
		for neighbor in neighbors:
			response = self.trainY.iloc[neighbor]
			if response in classVotes:
				classVotes[response] += 1
			else:
				classVotes[response] = 1
		return getMostCommonValue(classVotes)

	# def buildModel(self, trainingData, k=3):
	# 	for x in range(len(self.data)):
	# 		neighbors = getNeighbors(trainingSet, testSet[x], k)
	# 		result = getResponse(neighbors)
	# 		predictions.append(result)

	def predictRow(self, row):
		pass