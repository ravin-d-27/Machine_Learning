import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data Preprocessing

dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:,[3,4]].values

# Finding the number of clusters using the elbow method

from sklearn.cluster import KMeans

# Gonna Try different number of clusters

wcss = list()
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++',random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title("Elbow Method")
plt.xlabel('No. of Clusters')
plt.ylabel('Within-Cluster-Sum-of-Squares (WCSS)')
plt.show()

# Visualising the plots before clustering

for i in X:
    plt.scatter(i[0],i[1],color='black')
plt.title('Plots Before Clustering')
plt.xlabel('Annual Income (K$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

# Training the K-Means Model on the dataset

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters

plt.scatter(X[y_kmeans == 0,0],X[y_kmeans==0,1],s=30,c='red',label='Cluster1')
plt.scatter(X[y_kmeans == 1,0],X[y_kmeans==1,1],s=30,c='blue',label='Cluster2')
plt.scatter(X[y_kmeans == 2,0],X[y_kmeans==2,1],s=30,c='green',label='Cluster3')
plt.scatter(X[y_kmeans == 3,0],X[y_kmeans==3,1],s=30,c='cyan',label='Cluster4')
plt.scatter(X[y_kmeans == 4,0],X[y_kmeans==4,1],s=30,c='magenta',label='Cluster5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'yellow', label = 'Centroids')
plt.title('The Customer Cluster')
plt.xlabel('Annual Income (K$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()