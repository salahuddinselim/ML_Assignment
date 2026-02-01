

import pandas as pd
import numpy as np
import os
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.preprocessing import StandardScaler


path = kagglehub.dataset_download("msjahid/bangladesh-districts-wise-population")
csv_path = os.path.join(path, 'city_population.csv')
df = pd.read_csv(csv_path)


X = df[['Population_2022', 'Area (km2)']]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


dbscan = DBSCAN(eps=0.5, min_samples=3)
df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)


hdbscan = HDBSCAN(min_cluster_size=3, min_samples=2)
df['HDBSCAN_Cluster'] = hdbscan.fit_predict(X_scaled)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))


sns.scatterplot(data=df, x='Population_2022', y='Area (km2)',
                hue='DBSCAN_Cluster', palette='viridis', ax=ax1, s=100)
ax1.set_title('DBSCAN Clustering\n(Fixed Density)')


sns.scatterplot(data=df, x='Population_2022', y='Area (km2)',
                hue='HDBSCAN_Cluster', palette='magma', ax=ax2, s=100)
ax2.set_title('HDBSCAN Clustering\n(Variable Density)')

plt.tight_layout()
plt.show()


outliers = df[df['HDBSCAN_Cluster'] == -1]
print(f"Detected {len(outliers)} outliers (noise points) using HDBSCAN:")
print(outliers[['Name', 'Population_2022', 'Area (km2)']].head())