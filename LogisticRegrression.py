# Train a logistic regression classifier to predict whether flower is iris virginica or not
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
iris = datasets.load_iris()
# print(list(iris.keys()))
# print(iris.data)/print(iris['data'])
# print(iris['data'].shape)
# print(iris['target'])

# To explain logistic regression we will use only 1 feature
x = iris.data[:,3:] # We are trying to print only the last feature
# We want to make a classifier whether flower is iris virginica or not. binary classifier
y = (iris['target'] == True).astype(np.int)
# print(y)

# Train a logistic regression classifier
clf = LogisticRegression()
clf.fit(x,y)
y_pred = clf.predict([[3.6]])
print(y_pred)

# Using Matplotlib to plot the visualization
x_new = np.linspace(-10,20,100000).reshape(-1,1) # Reshapes it to  1D array
# Very easy function to introduce 1000 points between 0 and 3
y_prob = clf.predict_proba(x_new)
print(y_prob)
plt.plot(x_new, y_prob[:,1], "g-", label="virginica")
plt.show()