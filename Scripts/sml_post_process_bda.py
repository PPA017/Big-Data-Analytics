import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('iris_clustered.csv')

features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
X = data[features].values
y = data['Cluster'].values
K = data['Cluster'].nunique()

centroids = (
    data.groupby('Cluster')[features]
    .mean()
    .values
)
'========================================= Dataset Split (SC) ========================================= '

def ComputeSplitCriterion(X, idx, centroids):
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
        
data['SC_Score'] = ComputeSplitCriterion(X, y, centroids)
sc_value = 0.5

data['Misclassified'] = False
data.loc[data['SC_Score'] > sc_value, 'Misclassified'] = True
stable_data = data[data['Misclassified'] == False].copy()
unstable_data = data[data['Misclassified'] == True].copy()
'========================================= SML Train on stable Data ========================================= '

if len(unstable_data) > 0:
    X_train = stable_data[features].values
    y_train = stable_data['Cluster'].values
    X_test = unstable_data[features].values

    classifier = RandomForestClassifier(n_estimators=200, random_state=42)
    classifier.fit(X_train, y_train)

    unstable_data['NewCluster'] = classifier.predict(X_test)
else:
    unstable_data['NewCluster'] = unstable_data['Cluster']

'========================================= Final Clusters ========================================= '

stable_data['FinalCluster'] = stable_data['Cluster']
unstable_data['FinalCluster'] = unstable_data['NewCluster']

final_data = pd.concat([stable_data, unstable_data]).sort_index()

final_centroids = final_data.groupby('FinalCluster')[features].mean().values

final_data.to_csv('iris_kmsml_final.csv', index=False)

print(final_centroids)