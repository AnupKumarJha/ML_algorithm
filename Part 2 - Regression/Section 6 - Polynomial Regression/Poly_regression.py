# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import  pandas as pd
import  numpy as np
# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
print(X,y)


"""# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#fitting linear model to polynomail regression
from sklearn.linear_model import LinearRegression
linear_regress=LinearRegression()
linear_regress.fit(X,y)

#plotting for linear regression
import matplotlib.pyplot as plt
plt.scatter(X,y,color="red")
plt.plot(X,linear_regress.predict(X),color='blue')
plt.xlabel("Experience ")
plt.ylabel("Salary")
plt.title("salary VS Position")
plt.show()

#Polynomaial regression
from sklearn.preprocessing import PolynomialFeatures
Poly_reg=PolynomialFeatures(degree=4)
X_poly=Poly_reg.fit_transform(X)
linear_regress2=LinearRegression()
# print(X_poly)
linear_regress2.fit(X_poly,y)

#ploting for polynomail regresssion
import matplotlib.pyplot as plt
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X,y,color="red")
plt.plot(X,linear_regress2.predict(Poly_reg.fit_transform(X)),color='blue')
plt.xlabel("Experience ")
plt.ylabel("Salary")
plt.title("salary VS Position")
plt.show()

#predicting the new result
print(linear_regress2.predict(Poly_reg.fit_transform([[6.5]])))

