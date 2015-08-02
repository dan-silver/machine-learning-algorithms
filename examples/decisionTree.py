import sys
sys.path.append("..") 
from decisionTree import DecisionTree

from common import *

data = loadData('iris.data')
train_label, train_features, test_label, test_features = getTrainAndTestSets(data, 'class')

x = DecisionTree()
x.buildTree(train_features, train_label)

# print x.predict([[3,4,5]])

print x.exportModel()
