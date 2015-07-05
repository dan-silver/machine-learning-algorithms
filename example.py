from decisionTree import DecisionTree

x = DecisionTree()
x.loadCSV('data.csv', 'play')
x.createTree()

# print x.predict([[3,4,5]])

print x.exportModel()