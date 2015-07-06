import sys
sys.path.append("..") 
from decisionTree import DecisionTree

from common import *

data = loadData('iris.data')
trainResult, trainData, testResult, testData = getTrainAndTestSets(data, 'class')

x = DecisionTree()
x.buildTree(trainData, trainResult)

# print x.predict([[3,4,5]])

print x.exportModel()