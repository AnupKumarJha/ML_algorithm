# Data Preprocessing Template

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
print(dataset.describe())
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
ct = ColumnTransformer([("State", OneHotEncoder(), [3])],    remainder = 'passthrough')
X = ct.fit_transform(X)
print(X)


# Avoiding the Dummy Variable Trap
X = X[:, 1:]
print("X",X)

print(type(y),y)
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

print(type(X_train),type(y_train[0]))
from  sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)


# predicting the test model
y_pred=lr.predict(X_test)
#testing the model
from sklearn import metrics as mt
print("MAE=",mt.mean_absolute_error(y_test,y_pred))
print("MSE=",mt.mean_squared_error(y_test,y_pred))
print("Root Mean squared error ",np.sqrt(mt.mean_squared_error(y_test,y_pred)))
print("variance regression score function",mt.explained_variance_score(y_test,y_pred))
#More near to one is better
print("Maximum resudual error =",mt.max_error(y_test,y_pred))

#Building the optimal model using back ward elimination
# import statsmodels.formula.api as sm
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
# X_opt=X[:,[0,1,2,3,4,5]]
#
# regressor_OLS=sm.ols(endog=y,exog=X_opt).fit()
# print(regressor_OLS.summray())
#

import statsmodels.formula.api as sm


def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50, 6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.ols(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:, j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.ols(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:, [0, j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print(regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
            # regressor_OLS.summary()
    return x


SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)



