import sys
sys.path.append("..") 
from knn import NearestNeighbor

from common import *

data = loadData('iris.data')
trainResult, trainData, testResult, testData = getTrainAndTestSets(data, 'class', trainPercent=0.8)

x = NearestNeighbor()
x.addData(trainData=trainData)



import pdb; pdb.set_trace()