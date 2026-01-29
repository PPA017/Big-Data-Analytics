import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

try:
    df = pd.read_csv('dataset2_km_sml_final.csv')
except FileNotFoundError:
    print("Error: 'dataset2_km_sml_final.csv' not found. Please run the previous script first.")
    exit()

metadata_cols = ['Cluster', 'SC_Score', 'Misclassified', 'NewCluster', 'FinalCluster', 'Stability']
feature_cols = [c for c in df.columns if c not in metadata_cols]

X = df[feature_cols].values
final_clusters = df['FinalCluster'].values
misclassified = df['Misclassified'].values 
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plot_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
plot_df['Cluster'] = final_clusters
plot_df['Refined'] = misclassified

plt.figure(figsize=(10, 7))

colors = ['purple', 'teal', 'gold']

for k in range(3):
    stable_points = plot_df[(plot_df['Cluster'] == k) & (plot_df['Refined'] == False)]
    plt.scatter(stable_points['PC1'], stable_points['PC2'], 
                c=colors[k], 
                marker='o', 
                edgecolor='k', 
                s=60, 
                alpha=0.7,
                label=f'Cluster {k} (Stable)')

    refined_points = plot_df[(plot_df['Cluster'] == k) & (plot_df['Refined'] == True)]
    plt.scatter(refined_points['PC1'], refined_points['PC2'], 
                c=colors[k], 
                marker='x', 
                s=100, 
                linewidth=2, 
                label=f'Cluster {k} (Refined)')

plt.title('KM-SML Hybrid Results: Dataset 2 (PCA Projection)', fontsize=14)
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} Variance)')
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} Variance)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Cluster Type")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.show()

print(f"Total Variance Explained by 2D Projection: {sum(pca.explained_variance_ratio_):.2%}")