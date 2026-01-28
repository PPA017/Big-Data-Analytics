import pandas as pd
import numpy as np
import os

needed_cols = [0, 1, 2, 3]
X = pd.read_csv(os.path.join('Dataset 1','Iris-150.txt'), usecols=needed_cols, header=None).values

def kMeansInitCentroids(X, K):
    
    m, n = X.shape
    centroids = np.zeros((K, n))
    
    randidx = np.random.permutation(m)[:K]
    centroids = X[randidx, :]
    
    return centroids

def findClosestCentroid(X, centroids):
    
    K = centroids.shape[0]
    idx = np.zeros(X.shape[0], dtype=int)
    
    for i in range(X.shape[0]):
        
        squaredDistances = np.sum((centroids - X[i]) ** 2, axis=1)
        idx[i] = np.argmin(squaredDistances)
    
    return idx

def computeCentroids(X, idx, K):
    
    m, n = X.shape
    centroids = np.zeros((K,n))
    
    for k in range(K):
        
        points = X[idx == k]
        if len(points) > 0:
            centroids[k] = np.mean(points, axis=0)
        else:
            centroids[k] = np.zeros(n)
    
    return centroids
    

#print(X)
K = 3 # setosa, versicolor, virginica
max_iters = 10
initial_centroids = kMeansInitCentroids(X, K)
centroids = initial_centroids

for i in range(max_iters):
    idx = findClosestCentroid(X, centroids)
    centroids = computeCentroids(X, idx, K)
    

results = pd.read_csv(os.path.join('Dataset 1','Iris-150.txt'), header=None)
results['Cluster'] = idx
results.columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species', 'Cluster']
results.to_csv('iris_clustered.csv', index=False)

print("Final Centroids:\n", centroids)