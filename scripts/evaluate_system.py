# scripts/evaluate_system.py
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix
import os

# Funciones
def lpc_distance(a, b):
    return np.sum((a - b)**2, axis=-1)

# Cargar datos
with open('data/lpc_features.pkl', 'rb') as f:
    lpc_features = pickle.load(f)

with open('data/codebooks.pkl', 'rb') as f:
    codebooks = pickle.load(f)

y_true = []
y_pred = []

# Organizar archivos por número
for filename in lpc_features:
    num = int(filename.split('num')[1].split('_')[0])
    rep = int(filename.split('rep')[1].split('.')[0])
    if rep >= 10:  # Sólo usar las repeticiones 10-14
        lpcs = lpc_features[filename]
        distances = []
        for target_num in codebooks:
            codebook = codebooks[target_num]
            dists = []
            for lpc in lpcs:
                dist = np.min(lpc_distance(codebook, lpc))
                dists.append(dist)
            distances.append(np.mean(dists))
        pred = np.argmin(distances) + 1
        y_true.append(num)
        y_pred.append(pred)

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred, labels=range(1,11))
print("Matriz de confusión:")
print(cm)
