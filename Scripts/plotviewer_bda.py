import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('iris_kmsml_final.csv')

plt.figure(figsize=(10, 6))

colors = ['purple', 'teal', 'gold'] 

for i in range(3):
    stable = df[(df['FinalCluster'] == i) & (df['Misclassified'] == False)]
    plt.scatter(stable['PetalLength'], stable['PetalWidth'], 
                c=colors[i], 
                marker='o', 
                edgecolor='k', 
                s=60, 
                label=f'Cluster {i} (Stable)')
    
    refined = df[(df['FinalCluster'] == i) & (df['Misclassified'] == True)]
    plt.scatter(refined['PetalLength'], refined['PetalWidth'], 
                c=colors[i], 
                marker='x', 
                s=100, 
                linewidth=2, 
                label=f'Cluster {i} (Refined)')

plt.title('KM-SML Hybrid Results: Final Clusters', fontsize=14)
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Cluster Type")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.show()