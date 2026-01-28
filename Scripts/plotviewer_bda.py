import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('iris_clustered.csv')

plt.figure(figsize=(10,6))

scatter = plt.scatter(data['PetalLength'], 
                      data['PetalWidth'], 
                      c=data['Cluster'], 
                      cmap='viridis', 
                      marker='o', 
                      edgecolor='k', 
                      s=50)

plt.title('K Means results')
plt.xlabel('Petal length (cm)')
plt.ylabel('Petal length (cm)')

legend1 = plt.legend(*scatter.legend_elements(), title="Clusters")
plt.gca().add_artist(legend1)

plt.grid(True, linestyle='--', alpha=0.6)
plt.show()