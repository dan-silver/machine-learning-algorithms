import pandas as pd

class Model:
	def fit(self, trainX, trainY):
		self.trainX = trainX
		self.trainY = trainY

	def predict(self, xs):
		predictions = []
		for row in xs:
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