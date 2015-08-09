import pandas as pd

class Model(object):
	def fit(self, trainX, trainY, **options):
		self.trainX = trainX
		self.trainY = trainY
		self.options = options

	def predict(self, testX):
		predictions = []
		for index, row in testX.iterrows():
			predictions.append(self.predictRow(row))
		return predictions

	def exportModel(self):
		return self.model

	def saveModel(self):
		pass

	def loadModel(self):
		pass

	def toJson(self):
		pass