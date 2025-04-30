# scripts/preprocess_audio.py
import numpy as np
import os
import soundfile as sf
import librosa
import pickle

# Funciones:
def pre_emphasis(signal, alpha=0.95):
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])

def framing(signal, frame_size=320, hop_size=128):
    frames = []
    for i in range(0, len(signal) - frame_size, hop_size):
        frames.append(signal[i:i+frame_size])
    return np.array(frames)

def apply_hamming(frames):
    window = np.hamming(frames.shape[1])
    return frames * window

def compute_energy(frames):
    return np.sum(frames ** 2, axis=1)

def detect_voiced_frames(energies, threshold_ratio=0.1):
    threshold = np.max(energies) * threshold_ratio
    return energies > threshold

def get_lpc_coefficients(frame, order=12):
    return librosa.lpc(frame, order)

# Preprocesar todas las grabaciones
input_dir = 'recordings/'
output_dir = 'data/'
os.makedirs(output_dir, exist_ok=True)

lpc_features = {}

for filename in os.listdir(input_dir):
    if filename.endswith('.wav'):
        filepath = os.path.join(input_dir, filename)
        signal, sr = sf.read(filepath)
        signal = pre_emphasis(signal.flatten())

        frames = framing(signal)
        frames = apply_hamming(frames)

        energies = compute_energy(frames)
        voiced_mask = detect_voiced_frames(energies)

        voiced_frames = frames[voiced_mask]

        lpcs = np.array([get_lpc_coefficients(frame) for frame in voiced_frames])
        lpc_features[filename] = lpcs

# Guardar LPCs
with open(os.path.join(output_dir, 'lpc_features.pkl'), 'wb') as f:
    pickle.dump(lpc_features, f)

print("Preprocesamiento terminado y LPCs guardados.")
