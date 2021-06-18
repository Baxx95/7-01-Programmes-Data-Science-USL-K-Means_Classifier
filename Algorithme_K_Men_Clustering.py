# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 18:50:21 2021

@author: Zakaria
"""




import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans

from sklearn.datasets import load_iris


# Générer des données:
X, y = make_blobs(n_samples=100, centers = 3, cluster_std=0.5, random_state=0) #nb_features = 2 par défaut
plt.scatter(X[:,0], X[:, 1])


# Entrainer le modele de K-mean Clustering
model = KMeans(n_clusters=3)
model.fit(X)


#Visualiser les Clusters
predictions = model.predict(X)
plt.scatter(X[:,0], X[:,1], c=predictions)

#------------------------------------------
#Clustering des fleurs d'iris avec K-Mean
#------------------------------------------
iris = load_iris()

X = iris.data
y = iris.target

# Visualisation des donées
colormap=np.array(['Red','green','blue'])
plt.scatter(X[:,3], X[:,1], c = colormap[y]) 


# Entrainer le modele de K-mean Clustering
model = KMeans(n_clusters=3)
model.fit(X)


#Visualiser les Clusters
predictions = model.predict(X)
plt.scatter(X[:,3], X[:,1], c=predictions)

