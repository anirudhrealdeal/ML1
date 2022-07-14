import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

# Making our regressor
diabetes = datasets.load_diabetes()
# np array because sklearn can work with only nparray
diabetes_X = np.array([[1], [2], [3]])
# Now performing the train/test splitting
diabetes_X_train = diabetes_X
diabetes_X_test = diabetes_X
diabetes_Y_train = np.array([3,2,4])
diabetes_Y_test = np.array([3,2,4])

model = linear_model.LinearRegression()
model.fit(diabetes_X_train, diabetes_Y_train)
diabetes_Y_predict = model.predict(diabetes_X_test)
print("Mean squared Error:", mean_squared_error(diabetes_Y_test, diabetes_Y_predict))

# Now let's print the weights and intercepts
print("Weights:", model.coef_)
print("Intercept:", model.intercept_)

'''
Output:
Mean squared Error: 0.5000000000000001
Weights: [0.5]
Intercept: 2.0
'''


