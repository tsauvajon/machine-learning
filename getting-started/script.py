#!/usr/bin/python3

from sklearn import tree
from sklearn import ensemble
from sklearn import neighbors

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
'female', 'male', 'male']

dtClassifier = tree.DecisionTreeClassifier()
dtClassifier = dtClassifier.fit(X,Y)

rfClassifier = ensemble.RandomForestClassifier()
rfClassifier = rfClassifier.fit(X,Y)

nClassifier = neighbors.KNeighborsClassifier(n_neighbors=3)
nClassifier = nClassifier.fit(X,Y)

dtPrediction = dtClassifier.predict([[150, 55, 36]])
rfPrediction = rfClassifier.predict([[150, 55, 36]])
nPrediction = nClassifier.predict([[150, 55, 36]])

print(dtPrediction)
print(rfPrediction)
print(nPrediction)
