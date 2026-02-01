

import kagglehub
path = kagglehub.dataset_download("jessicali9530/animal-crossing-new-horizons-nookplaza-dataset")

import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


csv_path = os.path.join(path, 'villagers.csv')
df = pd.read_csv(csv_path)


features = ['Gender', 'Species', 'Personality']
X = df[features].copy()



le = LabelEncoder()
for col in features:
    X[col] = le.fit_transform(X[col])


kmeans = KMeans(n_clusters=6, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X)


plt.figure(figsize=(10, 6))

plt.scatter(X['Personality'], X['Species'], c=df['Cluster'], cmap='rainbow', s=100, alpha=0.6)
plt.title('Animal Crossing Villager Clusters')
plt.xlabel('Personality (Encoded)')
plt.ylabel('Species (Encoded)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


print("Sample of Villager Groups:")
print(df[['Name', 'Species', 'Personality', 'Cluster']].head(10))

from google.colab import drive
drive.mount('/content/drive')