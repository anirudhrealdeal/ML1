import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

# Making our regressor
diabetes = datasets.load_diabetes()
# getting to know what's in the dataset
# dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])
print(diabetes.keys())
# print(diabetes.DESCR)
# print(diabetes.data)
diabetes_X = diabetes.data
# What feature was there in diabetes.data it gives the second data in column format. made it an array of arrays
# print(diabetes_X)
# Now performing the train/test splitting
diabetes_X_train = diabetes_X[:-30]  # We are taking the last 30 elements for training
diabetes_X_test = diabetes_X[-30:]  # We are taking the first 30 elements for training
'''
# Initialize list
Lst = [50, 70, 30, 20, 90, 10, 50]

# Display list
print(Lst[-7::1])


Output: [50, 70, 30, 20, 90, 10, 50]
'''
diabetes_Y_train = diabetes.target[:-30]
diabetes_Y_test = diabetes.target[-30:]

model = linear_model.LinearRegression()
model.fit(diabetes_X_train, diabetes_Y_train)
diabetes_Y_predict = model.predict(diabetes_X_test)
print("Mean squared Error:", mean_squared_error(diabetes_Y_test, diabetes_Y_predict))
# Mean squared Error: 3035.060115291269 Boy that's huge for an error XD but linear regression is like that. It's a simple one
# Now let's print the weights and intercepts
print("Weights:", model.coef_)
print("Intercept:", model.intercept_)

'''
Output:
dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])
Mean squared Error: 1826.4841712795037
Weights: [  -1.16678648 -237.18123633  518.31283524  309.04204042 -763.10835067
  458.88378916   80.61107395  174.31796962  721.48087773   79.1952801 ]
Intercept: 153.05824267739402
'''


