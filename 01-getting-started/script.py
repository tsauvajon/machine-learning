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

dt_classifier = tree.DecisionTreeClassifier()
dt_classifier = dt_classifier.fit(X,Y)

rf_classifier = ensemble.RandomForestClassifier()
rf_classifier = rf_classifier.fit(X,Y)

n_classifier = neighbors.KNeighborsClassifier(n_neighbors=3)
n_classifier = n_classifier.fit(X,Y)

dt_prediction = dt_classifier.predict([[150, 55, 36]])
rf_prediction = rf_classifier.predict([[150, 55, 36]])
n_prediction = n_classifier.predict([[150, 55, 36]])

print(dt_prediction)
print(rf_prediction)
print(n_prediction)
