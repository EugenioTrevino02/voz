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

def get_lpc_coefficients(frame, order=20):
    return librosa.lpc(frame, order=order)

# Preprocesar todas las grabaciones
input_base_dir = 'recordings/'  # Asegúrate de que esta ruta sea correcta
output_dir = 'data/'
os.makedirs(output_dir, exist_ok=True)

lpc_features = {}

print(f"Leyendo archivos desde el directorio base: {input_base_dir}")
for number_dir_name in sorted(os.listdir(input_base_dir)):
    number_dir_path = os.path.join(input_base_dir, number_dir_name)
    if os.path.isdir(number_dir_path) and number_dir_name.isdigit():
        print(f"  Entrando a la carpeta: {number_dir_name}")
        for filename in os.listdir(number_dir_path):
            if filename.endswith('.wav'):
                filepath = os.path.join(number_dir_path, filename)
                print(f"    Procesando archivo: {filepath}")
                try:
                    signal, sr = sf.read(filepath)
                    print(f"      Forma de la señal: {signal.shape}, Sample Rate: {sr}")
                    signal = pre_emphasis(signal.flatten())

                    frames = framing(signal)
                    print(f"      Número de frames: {frames.shape[0]}, Forma del primer frame: {frames.shape[1] if frames.ndim > 1 else 0}")
                    energies = compute_energy(frames)
                    voiced_mask = detect_voiced_frames(energies)
                    voiced_frames = frames[voiced_mask]

                    lpcs = np.array([get_lpc_coefficients(frame) for frame in voiced_frames])
                    print(f"      Número de frames con voz: {voiced_frames.shape[0]}, Forma del primer vector LPC: {lpcs[0].shape if lpcs.size > 0 else 'Sin frames con voz'}")
                    lpc_features[filename] = lpcs
                except Exception as e:
                    print(f"    Error al procesar {filepath}: {e}")

print("\nResumen de lpc_features:")
for filename, lpcs in lpc_features.items():
    print(f"  {filename}: Número de secuencias LPC = {len(lpcs)}, Forma de la primera secuencia = {lpcs[0].shape if len(lpcs) > 0 else 'Sin secuencias'}")

# Guardar LPCs
with open(os.path.join(output_dir, 'lpc_features.pkl'), 'wb') as f:
    pickle.dump(lpc_features, f)

print("Preprocesamiento terminado y LPCs guardados.")
