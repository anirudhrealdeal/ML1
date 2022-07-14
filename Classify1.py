# KNN Classifier(K Nearest Neighbour)
# Loading modules
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# Loading Dataset
iris = datasets.load_iris()

# Printing descriptions and features
# print(iris.keys())
# print(iris.DESCR)
features = iris.data
labels = iris.target
# print(features[0],labels[0])

# Training Classifier
clf = KNeighborsClassifier()
clf.fit(features, labels)
label_pred = clf.predict([[31,1,1,2.1]])
print(label_pred)
# Output: [2] ie Virginica

