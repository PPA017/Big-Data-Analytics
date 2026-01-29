import pandas as pd
import matplotlib.pyplot as plt

df_final = pd.read_csv('iris_km_sml.csv')

fig, ax = plt.subplots(figsize=(10, 6))

colors = ['#440154', '#21918c', '#fde725'] 

for i in range(3):
    stable = df_final[(df_final['Cluster'] == i) & (df_final['Stability'] == 'Stable')]
    ax.scatter(stable['PetalLength'], stable['PetalWidth'], 
               c=colors[i], marker='o', edgecolor='k', s=60, label=f'Cluster {i} (Stable)')
    
    unstable = df_final[(df_final['Cluster'] == i) & (df_final['Stability'] == 'Unstable')]
    ax.scatter(unstable['PetalLength'], unstable['PetalWidth'], 
               c=colors[i], marker='x', s=100, label=f'Cluster {i} (Refined)')

ax.set_title('KM-SML Hybrid Results: Post-Processed Clusters')
ax.set_xlabel('Petal Length (cm)')
ax.set_ylabel('Petal Width (cm)')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Refinement Status")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()