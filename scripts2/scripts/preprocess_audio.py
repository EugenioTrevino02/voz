import numpy as np
import os
import soundfile as sf
import librosa
import pickle
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

# ------------------ Funciones ------------------

# Filtro de pre-énfasis, que refuerza las altas frecuencias para mejorar la estimación LPC.
def pre_emphasis(signal, alpha=0.95):
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])

# Divide la señal en ventanas superpuestas de 320 muestras (20 ms a 16 kHz), desplazadas 128 muestras
def framing(signal, frame_size=320, hop_size=128):
    frames = []
    for i in range(0, len(signal) - frame_size, hop_size):
        frames.append(signal[i:i+frame_size])
    return np.array(frames)

# Aplica ventana de Hamming a cada frame, reduciendo el efecto de discontinuidades en los bordes
def apply_hamming(frames):
    window = np.hamming(frames.shape[1])
    return frames * window

# Calcula energía de cada frame
def compute_energy(frames):
    return np.sum(frames ** 2, axis=1)

# Filtra el ruido, marcando como voz los frames cuya energía supera un umbral del 10%
def detect_voiced_frames(energies, threshold_ratio=0.1):
    threshold = np.max(energies) * threshold_ratio
    return energies > threshold

# Convierte coeficientes LPC a LSF. Los LSF son más estables numéricamente, lo cual terminando siendo mejor para cuantización
def lpc_to_lsf(a):
    a = np.array(a, dtype=np.float64)
    if a[0] != 1.0:
        a = a / a[0]

    p = len(a) - 1  # Orden LPC

    A = a
    A_flip = a[::-1]

    P = A + A_flip
    Q = A - A_flip

    P_poly = Polynomial(P)
    Q_poly = Polynomial(Q)

    P_roots = P_poly.roots()
    Q_roots = Q_poly.roots()

    P_angles = np.angle(P_roots[np.abs(np.abs(P_roots) - 1) < 1e-6])
    Q_angles = np.angle(Q_roots[np.abs(np.abs(Q_roots) - 1) < 1e-6])

    lsf = np.sort(np.abs(np.concatenate((P_angles, Q_angles))))
    return lsf

# Calcula los coeficientes LPC de un frame
def get_lpc_coefficients(frame, order=12):
    return librosa.lpc(frame, order=order)

# Obtiene los LSF de un frame, utilizando los coeficientes LPC
def get_lsf_from_frame(frame, order=12):
    try:
        lpc = get_lpc_coefficients(frame, order)
        lsf = lpc_to_lsf(lpc)
        return lsf
    except Exception as e:
        print(f"Error al convertir a LSF: {e}")
        return None

# ------------------ Configuración ------------------

input_base_dir = 'recordings/'
output_dir = 'data/'
os.makedirs(output_dir, exist_ok=True)

lpc_features = {}
lsf_features = {}

# ------------------ Procesamiento ------------------

print(f"Leyendo archivos desde el directorio base: {input_base_dir}")
for number_dir_name in sorted(os.listdir(input_base_dir), key=lambda x: int(x) if x.isdigit() else float('inf')):
    number_dir_path = os.path.join(input_base_dir, number_dir_name)
    if os.path.isdir(number_dir_path) and number_dir_name.isdigit():
        print(f"Entrando a la carpeta: {number_dir_name}")
        for filename in os.listdir(number_dir_path):
            if filename.endswith('.wav'):
                filepath = os.path.join(number_dir_path, filename)
                print(f"Procesando archivo: {filepath}")
                try:
                    signal, sr = sf.read(filepath)
                    signal = signal.flatten()

                    # --- Preénfasis ---
                    emphasized = pre_emphasis(signal)

                    # --- Framing ---
                    frames = framing(emphasized)

                    # --- Aplicar ventana de Hamming ---
                    frames = apply_hamming(frames)

                    # --- Energía ---
                    energies = compute_energy(frames)
                    voiced_mask = detect_voiced_frames(energies)
                    voiced_frames = frames[voiced_mask]

                    lpcs = np.array([get_lpc_coefficients(frame) for frame in voiced_frames])
                    lsfs = np.array([get_lsf_from_frame(frame) for frame in voiced_frames])

                    print(f"{filename}: {voiced_frames.shape[0]} frames con voz")

                    lpc_features[filename] = lpcs
                    lsf_features[filename] = lsfs

                except Exception as e:
                    print(f"Error al procesar {filepath}: {e}")

# ------------------ Guardar ------------------

print("\nResumen de LPCs y LSFs:")
for filename in lpc_features.keys():
    print(f"  {filename}: {len(lpc_features[filename])} vectores LPC, {len(lsf_features[filename])} vectores LSF")

with open(os.path.join(output_dir, 'lpc_features.pkl'), 'wb') as f:
    pickle.dump(lpc_features, f)

with open(os.path.join(output_dir, 'lsf_features.pkl'), 'wb') as f:
    pickle.dump(lsf_features, f)

print("Preprocesamiento y visualización completados.")
