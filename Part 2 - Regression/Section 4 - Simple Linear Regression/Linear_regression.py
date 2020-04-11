# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset

dataset = pd.read_csv('Salary_Data.csv')
# print(dataset.describe())
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
# print(X,y)

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import  train_test_split
# from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# print(X_train)

#fitting the data into Linear Regresson
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)

#estimating the coeff
print("slope=",lr.coef_,"intersept",lr.intercept_)

#predicting the test data
y_pred=lr.predict(X_test)

#visualizing the training set result
plt.scatter(X_train,y_train,color = 'red')
plt.plot(X_train, lr.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
# print(lr.predict([[10.2]]))

#Evaluating the Model

from sklearn import metrics as mt
print("MAE=",mt.mean_absolute_error(y_test,y_pred))
print("MSE=",mt.mean_squared_error(y_test,y_pred))
print("Root Mean squared error ",np.sqrt(mt.mean_squared_error(y_test,y_pred)))
print("variance regression score function",mt.explained_variance_score(y_test,y_pred))
#More near to one is better
print("Maximum resudual error =",mt.max_error(y_test,y_pred))