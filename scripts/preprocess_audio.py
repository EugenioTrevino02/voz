import numpy as np
import os
import soundfile as sf
import librosa
import pickle
import matplotlib.pyplot as plt

# ------------------ Funciones ------------------

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
    return librosa.lpc(frame, order=order)

# ------------------ Configuración ------------------

input_base_dir = 'recordings4/'  # Ajusta según tu carpeta
output_dir = 'data/'
os.makedirs(output_dir, exist_ok=True)

lpc_features = {}

# ------------------ Procesamiento ------------------

print(f"Leyendo archivos desde el directorio base: {input_base_dir}")
for number_dir_name in sorted(os.listdir(input_base_dir), key=lambda x: int(x) if x.isdigit() else float('inf')):
    number_dir_path = os.path.join(input_base_dir, number_dir_name)
    if os.path.isdir(number_dir_path) and number_dir_name.isdigit():
        print(f"  Entrando a la carpeta: {number_dir_name}")
        for filename in os.listdir(number_dir_path):
            if filename.endswith('.wav'):
                filepath = os.path.join(number_dir_path, filename)
                print(f"    Procesando archivo: {filepath}")
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
                    print(f"      {filename}: {voiced_frames.shape[0]} frames con voz")

                    lpc_features[filename] = lpcs

                except Exception as e:
                    print(f"    Error al procesar {filepath}: {e}")

# ------------------ Guardar ------------------

print("\nResumen de LPCs:")
for filename, lpcs in lpc_features.items():
    print(f"  {filename}: {len(lpcs)} vectores LPC")

with open(os.path.join(output_dir, 'lpc_features.pkl'), 'wb') as f:
    pickle.dump(lpc_features, f)

print("✅ Preprocesamiento y visualización completados.")
