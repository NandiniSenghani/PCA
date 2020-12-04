# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 23:16:21 2020

@author: Nandini senghani
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist 
from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch
from sklearn.cluster import	AgglomerativeClustering

wine=pd.read_csv("wine.csv")
wine.describe()
wine.head()
Q1=wine.quantile(0.25)
Q3=wine.quantile(0.75)
IQR=Q3-Q1
A=(wine<(Q1-1.5 * IQR))|(wine>(Q3+1.5*IQR))
A
wine_out= wine[~((wine<(Q1-1.5 * IQR))|(wine>(Q3+1.5*IQR))).any(axis=1)]
wine_out.shape
#outliers removed
# Normalizing the numerical data 
wine_normal = scale(wine_out)
pca = PCA()
pca_values = pca.fit_transform(wine_normal)
pca_values.shape
var = pca.explained_variance_ratio_
var
pca.components_[0]
# Cumulative variance 
var1 = np.cumsum(np.round(var,decimals = 4)*100)
var1
# Variance plot for PCA components obtained 
plt.plot(var1,color="red")
# plot between PCA1 and PCA2 
x = np.array(pca_values[:,0])
y = np.array(pca_values[:,1])
plt.plot(x,y,"ro")
plt.plot(np.arange(161),x,"ro")

df = pd.DataFrame(pca_values[:,0:11])
df3 = pd.DataFrame(pca_values[:,0:3])
#KMeans clusturing
k1 = list(range(2,10))
k1
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k1:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(df.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))
# Scree plot 
plt.plot(k1,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k1)
model2=KMeans(n_clusters=3) 
model2.fit(df)
model2.labels_ 

#HClusturing
z1 = linkage(df, method="complete",metric="euclidean")
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z1,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
h_clust= AgglomerativeClustering(n_clusters=4,linkage='complete',affinity = "euclidean").fit(df) 
h_clust.labels_

# Clustering with 3 pca components
#KMeans clusturing
k2 = list(range(2,10))
k2
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k2:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df3)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(df3.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df3.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))
# Scree plot 
plt.plot(k2,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k2)
model1=KMeans(n_clusters=3) 
model1.fit(df)
model1.labels_ 

#HClusturing
z = linkage(df3, method="complete",metric="euclidean")
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
h_clust1= AgglomerativeClustering(n_clusters=4,linkage='complete',affinity = "euclidean").fit(df) 
h_clust1.labels_
