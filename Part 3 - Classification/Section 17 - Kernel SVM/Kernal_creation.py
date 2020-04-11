# #importing the library
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import datasets,svm
#
# #importing datasets
# iris=datasets.load_breast_cancer()
# print(iris)
# X=iris.data[:,:2]
# Y=iris.target
# from sklearn import datasets
# iris = datasets.load_iris()
# digits = datasets.load_digits()
# print((digits.data))
# print("digits target=",digits.target)
# print(digits.images[0])
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
X = [['Male', 1], ['Female', 3], ['Female', 2]]
enc.fit(X)
enc.transform([['Female', 3], ['Male', 1]])
print("transformed data=",X[0][0])
print(enc.categories_)

#preprocessing tutorial
from sklearn import preprocessing
import numpy as np
X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])
X_scaled = preprocessing.scale(X_train)

print(X_scaled,"mean of scaled data is ",np.mean(X_scaled),"variance is =",np.var(X_scaled),"standard daviation is =",np.std(X_scaled))

#scaling the data
X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])
print("mean along vertical axis=0",np.mean(X_train,axis=0))
# min_max_scaler = preprocessing.MinMaxScaler()
# X_train_minmax = min_max_scaler.fit_transform(X_train)
# print(X_train_minmax)

#ordinal encoding
enc = preprocessing.OrdinalEncoder()
X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox']]
enc.fit(X)

print(enc.transform([['female', 'from US', 'uses Safari']]))
