import pandas as pd
import math
import operator

def majority(dataset, targetAttribute):
	return dataset[targetAttribute].value_counts().idxmax()

def euclideanDistance(data1, data2):
	distance = 0
	for x in range(len(data1.keys())):
		distance += pow((data1[x] - data2[x]), 2)
	return math.sqrt(distance)

# ex: stats = {'a':1000, 'b':3000, 'c': 100}
#     returns b
def getMostCommonValue(dict):
	sortedVotes = sorted(dict.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]