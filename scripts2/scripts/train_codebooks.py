import pickle
import numpy as np
from sklearn.cluster import KMeans
import os

# Cargar LSFs
with open('data/lsf_features.pkl', 'rb') as f:
    lsf_features = pickle.load(f)

codebooks = {}
n_clusters = 32  # Número de codevectors

# Organizar archivos por número
grouped = {}
for filename in lsf_features:
    try:
        num = int(filename.split('num')[1].split('_')[0])
        rep = int(filename.split('rep')[1].split('.')[0])
    except (IndexError, ValueError):
        print(f"Nombre de archivo inesperado: {filename}")
        continue

    if num not in grouped:
        grouped[num] = []

    if rep < 10:  # Usar solo repeticiones 0-9 para entrenamiento
        grouped[num].append(lsf_features[filename])

# Entrenar codebook para cada número/palabra
for num in grouped:
    print(f"Entrenando codebook para num={num}, cantidad de muestras: {len(grouped[num])}")
    for i, lsf in enumerate(grouped[num]):
        print(f"  Muestra {i}: shape = {lsf.shape}")
    data = np.vstack(grouped[num])
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
    codebooks[num] = kmeans.cluster_centers_

# Guardar codebooks
with open('data/codebooks.pkl', 'wb') as f:
    pickle.dump(codebooks, f)

print("✅ Entrenamiento de codebooks con LSFs completado.")
