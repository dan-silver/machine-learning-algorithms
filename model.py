import pandas as pd

class Model:
	def loadCSV(self, filename, outputCol, delimeter=','):
		self.data = pd.read_csv(filename, sep=delimeter)
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