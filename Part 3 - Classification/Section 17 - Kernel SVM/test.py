from sklearn import svm
import numpy as np

X = [[0,1,3,2,5], [1,2,5,6,2]]
y = [0, 1]
# np.reshape(X,(-1,1))
# print(X)
clf = svm.SVC()
clf.fit(X, y)
print(clf.predict([[1,2,4,5,1]]))