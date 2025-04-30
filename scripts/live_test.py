import sounddevice as sd
import soundfile as sf
import numpy as np
import scipy.signal as signal
import pickle
import os

# Funciones
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

# Implementación propia de LPC (usando autocorrelación)
def get_lpc_coefficients(frame, order=12):
    autocorr = np.correlate(frame, frame, mode='full')[len(frame)-1:]
    R = autocorr[:order+1]
    A = np.zeros(order+1)
    A[0] = 1.0
    for i in range(1, order+1):
        error = np.sum(R[:i] * A[i-1::-1])
        gain = (R[i] - error) / A[i-1]
        A[i] = gain
    return A[1:]  # Retorna solo los coeficientes LPC (sin el coeficiente de ganancia)

def lpc_distance(a, b):
    return np.sum((a - b) ** 2)  # Distancia de suma de cuadrados

# Cargar codebooks
with open('data/codebooks.pkl', 'rb') as f:
    codebooks = pickle.load(f)

# Grabación
fs = 16000  # 16 kHz
duration = 2  # 2 segundos

print("Habla un número del 1 al 10:")
recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
sd.wait()

# Procesamiento
signal = recording.flatten()
signal = pre_emphasis(signal)
frames = framing(signal)
frames = apply_hamming(frames)

energies = compute_energy(frames)
voiced_mask = detect_voiced_frames(energies)

voiced_frames = frames[voiced_mask]

lpcs = np.array([get_lpc_coefficients(frame) for frame in voiced_frames])

# Clasificación
distances = []
for target_num in codebooks:
    codebook = codebooks[target_num]
    dists = []
    for lpc in lpcs:
        dist = np.min([lpc_distance(codebook[i], lpc) for i in range(len(codebook))])
        dists.append(dist)
    distances.append(np.mean(dists))

predicted_number = np.argmin(distances) + 1
print(f"Reconocido: {predicted_number}")
