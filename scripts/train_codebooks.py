# scripts/train_codebooks.py
import pickle
import numpy as np
from sklearn.cluster import KMeans
import os

# Cargar LPCs
with open('data/lpc_features.pkl', 'rb') as f:
    lpc_features = pickle.load(f)

codebooks = {}
n_clusters = 16  # Número de codevectors

# Organizar archivos por número
grouped = {}
for filename in lpc_features:
    num = int(filename.split('num')[1].split('_')[0])
    rep = int(filename.split('rep')[1].split('.')[0])
    if num not in grouped:
        grouped[num] = []
    if rep < 10:
        grouped[num].append(lpc_features[filename])

# Entrenar codebook para cada número
for num in grouped:
    data = np.vstack(grouped[num])
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
    codebooks[num] = kmeans.cluster_centers_

# Guardar codebooks
with open('../data/codebooks.pkl', 'wb') as f:
    pickle.dump(codebooks, f)

print("Entrenamiento de codebooks terminado.")
