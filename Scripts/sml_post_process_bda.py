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

def distanceToAssignedCentroid(X, y, centroids):
    distances = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        distances[i] = np.sum((X[i] - centroids[y[i]]) ** 2)
    return distances

data['distance'] = distanceToAssignedCentroid(X, y, centroids)


'========================================= Dataset Split (SC) ========================================= '

data['misclassified'] = False


for k in range(K):
    cluster_distances = data[data['Cluster'] == k]['distance']
    threshold = np.percentile(cluster_distances, 80) 

    data.loc[
        (data['Cluster'] == k) &
        (data['distance'] > threshold),
        'misclassified'
    ] = True
    
stable_data = data[data['misclassified'] == False].copy()
unstable_data = data[data['misclassified'] == True].copy()


'========================================= SML Train on stable Data ========================================= '


X_train = stable_data[features].values
y_train = stable_data['Cluster'].values

X_test = unstable_data[features].values

classifier = RandomForestClassifier(n_estimators = 200, random_state = 42)

classifier.fit(X_train, y_train)

unstable_data['NewCluster'] = classifier.predict(X_test)


'========================================= Final Clusters ========================================= '

stable_data['FinalCluster'] = stable_data['Cluster']
unstable_data['FinalCluster'] = unstable_data['NewCluster']

final_data = pd.concat([stable_data, unstable_data])
final_centroids = (
    final_data.groupby('FinalCluster')[features].mean()
)

final_centroids.to_csv('iris_km_sml.csv', index=False)

print('passed')
print(final_centroids)