
import kagglehub
path = kagglehub.dataset_download("jessicali9530/animal-crossing-new-horizons-nookplaza-dataset")

import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster


csv_path = os.path.join(path, 'villagers.csv')
df = pd.read_csv(csv_path)


features = ['Gender', 'Species', 'Personality']
X = df[features].copy()


le = LabelEncoder()
for col in features:
    X[col] = le.fit_transform(X[col])


linked = linkage(X, method='ward')


plt.figure(figsize=(12, 7))
dendrogram(linked,
           orientation='top',
           labels=df['Name'].values,
           distance_sort='descending',
           show_leaf_counts=True)

plt.title('Hierarchical Clustering Dendrogram (Villagers)')
plt.xlabel('Villager Name')
plt.ylabel('Euclidean Distance')
plt.xticks(rotation=90)
plt.show()



df['Hierarchical_Cluster'] = fcluster(linked, 6, criterion='maxclust')

print("Villagers grouped by Hierarchical Clustering:")
print(df[['Name', 'Species', 'Personality', 'Hierarchical_Cluster']].head(15))