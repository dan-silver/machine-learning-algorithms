import pandas as pd

class Model:
	def fit(self, trainX, trainY, **options):
		self.trainX = trainX
		self.trainY = trainY
		self.options = options

	def predict(self, testX):
		predictions = []
		[predictions.append(self.predictRow(row)) for row in testX]
		return predictions

	def exportModel(self):
		return self.model

	def saveModel(self):
		pass

	def loadModel(self):
		pass

	def toJson(self):
		pass