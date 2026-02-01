

import kagglehub


path = kagglehub.dataset_download("vjchoudhary7/customer-segmentation-tutorial-in-python")

print("Path to dataset files:", path)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os


csv_path = os.path.join(path, 'Mall_Customers.csv')
df = pd.read_csv(csv_path)


X = df.iloc[:, [3, 4]].values


kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_predict = kmeans.fit_predict(X)


df['Cluster'] = y_predict


plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']

for i in range(5):
    plt.scatter(X[y_predict == i, 0], X[y_predict == i, 1], s=50, c=colors[i], label=f'Cluster {i+1}')


plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='black', marker='X', label='Centroids')

plt.title('Customer Groups')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

print("Clustering complete. Here are the first few rows with their assigned clusters:")
print(df.head())