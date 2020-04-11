# KMeans algorithms # Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values
print(X)


# # Using the elbow method to find the optimal number of clusters
from  sklearn.cluster import KMeans
# wcss=[]
# for i in range(1,11):
#     kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=1000,random_state=12)
#     kmeans.fit(X)
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1,11),wcss)
# plt.title('The Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()

#fitting the dataset to kmeans algorithm
kmeans=KMeans(n_clusters=5,init='k-means++',max_iter=1000,random_state=12)
y_kmeans=kmeans.fit_predict(X)
print(y_kmeans)

#plotting the cluster

# plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='red',label='target')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='black',label='rich')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='green',label='kanjoosh')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c='blue',label='careless')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c='yellow',label='sensible')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

