import pandas as pd
import numpy as np
import os
import kmeans_bda
import matplotlib.pyplot as plt

X = pd.read_csv(os.path.join('Dataset 2', 'Dataset2.csv'), header=None).values



def computeInertia(X, centroids, idx):
    
    total = 0
    
    for i in range(len(X)):
        
        cluster_idx = idx[i]
        
        assigned_centroid = centroids[cluster_idx]
        
        squared_distance = np.sum((X[i] - assigned_centroid) ** 2)
        
        total += squared_distance
    
    return total
        
        

anomalies = []
K_range = range(1, 11)
iterations = 10

for k in K_range:
    
    centroids = kmeans_bda.kMeansInitCentroids(X, k)
    
    for i in range(iterations):
        idx = kmeans_bda.findClosestCentroid(X, centroids)
        centroids = kmeans_bda.computeCentroids(X, idx, k)
    
    inertia = computeInertia(X, centroids, idx)
    anomalies.append(inertia)
    
plt.figure(figsize=(8, 5))
plt.plot(K_range, anomalies, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.grid(True)
plt.show()
    
    