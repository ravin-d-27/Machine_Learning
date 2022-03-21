import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:,[3,4]].values

# Visualising the dendrogram

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X,method = 'ward'))
plt.title("The Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distance")
plt.show()

# Now Training the Hierarchical Model

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc = hc.fit_predict(X)

# Visualising the clusters

plt.scatter(X[y_hc == 0,0],X[y_hc==0,1],s=30,c='red',label='Cluster1')
plt.scatter(X[y_hc == 1,0],X[y_hc==1,1],s=30,c='blue',label='Cluster2')
plt.scatter(X[y_hc == 2,0],X[y_hc==2,1],s=30,c='green',label='Cluster3')
plt.scatter(X[y_hc == 3,0],X[y_hc==3,1],s=30,c='cyan',label='Cluster4')
plt.scatter(X[y_hc == 4,0],X[y_hc==4,1],s=30,c='magenta',label='Cluster5')

plt.title('The Customer Cluster')
plt.xlabel('Annual Income (K$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()