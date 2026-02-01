
!pip install minisom
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler


path = kagglehub.dataset_download("muratkokludataset/pumpkin-seeds-dataset")

df = pd.read_excel(f"{path}/Pumpkin_Seeds_Dataset/Pumpkin_Seeds_Dataset.xlsx")


data = df.iloc[:, 0:12].values
labels = df.iloc[:, 12].values


label_map = {label: i for i, label in enumerate(np.unique(labels))}
target = np.array([label_map[l] for l in labels])


scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)


som_grid_x, som_grid_y = 15, 15
som = MiniSom(som_grid_x, som_grid_y, 12, sigma=1.5, learning_rate=0.5)

som.pca_weights_init(data_scaled)
som.train_random(data_scaled, 1000, verbose=True)


plt.figure(figsize=(10, 10))
plt.pcolor(som.distance_map().T, cmap='bone_r') 
plt.colorbar()


markers = ['o', 's'] 
colors = ['C0', 'C1'] 

for i, x in enumerate(data_scaled):
    w = som.winner(x) 
    plt.plot(w[0] + 0.5, w[1] + 0.5, markers[target[i]],
             markerfacecolor='None', markeredgecolor=colors[target[i]],
             markersize=8, markeredgewidth=2)

plt.title('SOM U-Matrix: Pumpkin Seed Varieties')
plt.show()