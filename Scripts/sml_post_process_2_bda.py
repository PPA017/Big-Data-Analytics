import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import kmeans_bda
import sml_post_process_bda
import os

X = pd.read_csv(os.path.join('Dataset 2', 'Dataset2.csv'), header=None).values

K = 3 #from the elbow method
iterations = 10

centroids = kmeans_bda.kMeansInitCentroids(X, K)

for _ in range(iterations):
    
    idx = kmeans_bda.findClosestCentroid(X, centroids)
    centroids = kmeans_bda.computeCentroids(X, idx, K)
    

'========================================= Dataset Split (SC) ========================================= '
def computeSplitCriterion(X, idx, centroids):
    m = X.shape[0]
    num_centroids = centroids.shape[0]
    sc_values = np.zeros(m)
    
    for i in range(m):
        assigned_centroid = centroids[idx[i]]
        dist_assigned = np.linalg.norm(X[i] - assigned_centroid)
        
        other_distances = []
        for k in range(num_centroids):
            if k == idx[i]:
                continue 
            
            dist_other = np.linalg.norm(X[i] - centroids[k])
            other_distances.append(dist_other)
        
        dist_neighbor = min(other_distances)
        
        if dist_neighbor == 0:
            sc_values[i] = 0.0
        else:
            sc_values[i] = dist_assigned / dist_neighbor
            
    return sc_values

sc_scores = computeSplitCriterion(X, idx, centroids)

df = pd.DataFrame(X)
df['Cluster'] = idx
df['SC_Score'] = sc_scores

sc_value = 0.5

df['Misclassified'] = False
df.loc[df['SC_Score'] > sc_value, 'Misclassified'] = True

stable_data = df[df['Misclassified'] == False].copy()
unstable_data = df[df['Misclassified'] == True].copy()

if len(unstable_data) > 0:
    feature_cols = df.columns[:-3] 
    
    X_train = stable_data[feature_cols].values
    y_train = stable_data['Cluster'].values
    
    X_test = unstable_data[feature_cols].values
    
    classifier = RandomForestClassifier(n_estimators=200, random_state=42)
    classifier.fit(X_train, y_train)
    
    unstable_data['NewCluster'] = classifier.predict(X_test)
    
    num_changed = np.sum(unstable_data['Cluster'] != unstable_data['NewCluster'])

else:
    unstable_data['NewCluster'] = unstable_data['Cluster']

'========================================= SML Train on stable Data ========================================= '

stable_data['FinalCluster'] = stable_data['Cluster']
unstable_data['FinalCluster'] = unstable_data['NewCluster']

final_data = pd.concat([stable_data, unstable_data]).sort_index()

final_centroids = final_data.groupby('FinalCluster')[df.columns[:-3]].mean().values

final_data.to_csv('dataset2_km_sml_final.csv', index=False)
print("passed")


print(final_centroids)