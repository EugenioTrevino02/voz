import sounddevice as sd
import numpy as np
import pickle
import librosa
from numpy.polynomial import Polynomial

'''
Resultados de palabras:
- Stop: 1
- Continue: 2
- Next: 3
- Lift: 4
- Drop: 5
- Start: 6
- Finish: 7
- One: 8
- Two: 9
- Three: 10
'''

# ------------------ Funciones (Adaptadas del primer script) ------------------

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

def lpc_to_lsf(a):
    a = np.array(a, dtype=np.float64)
    if a[0] != 1.0:
        a = a / a[0]
    p = len(a) - 1
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

def get_lpc_coefficients(frame, order=12):
    return librosa.lpc(frame, order=order)

def get_lsf_from_frame(frame, order=12):
    try:
        lpc = get_lpc_coefficients(frame, order)
        lsf = lpc_to_lsf(lpc)
        return lsf
    except Exception as e:
        print(f"Error al convertir a LSF: {e}")
        return None

def lsf_distance(a, b):
    return np.sum((a - b)**2, axis=-1)

# ------------------ Configuración para grabación ------------------

samplerate = 16000  # Frecuencia de muestreo
duration = 2       # Duración de la grabación en segundos
channels = 1       # Mono

# ------------------ Cargar Codebooks ------------------

try:
    with open('data/codebooks.pkl', 'rb') as f:
        codebooks = pickle.load(f)
    labels = sorted(codebooks.keys()) # Asumimos que las claves son los números/etiquetas
    print("Codebooks cargados correctamente.")
except FileNotFoundError:
    print("Error: No se encontró el archivo 'data/codebooks.pkl'. Asegúrate de haber entrenado los codebooks.")
    exit()

# ------------------ Función para reconocer la palabra en tiempo real ------------------

def recognize_word():
    print(f"Grabando durante {duration} segundos...")
    recording = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=channels, dtype='float32')
    sd.wait()  # Esperar a que termine la grabación
    print("Grabación finalizada")

    signal = recording.flatten()

    # --- Preprocesamiento ---
    emphasized = pre_emphasis(signal)
    frames = framing(emphasized)
    frames = apply_hamming(frames)
    energies = compute_energy(frames)
    voiced_mask = detect_voiced_frames(energies)
    voiced_frames = frames[voiced_mask]

    if voiced_frames.shape[0] == 0:
        print("No se detectó voz en la grabación.")
        return

    # --- Extracción de LSFs ---
    live_lsfs = np.array([get_lsf_from_frame(frame) for frame in voiced_frames if get_lsf_from_frame(frame) is not None])

    if live_lsfs.shape[0] == 0:
        print("No se pudieron extraer características LSF válidas.")
        return

    # --- Comparación con Codebooks ---
    distances = []
    for label in labels:
        codebook = codebooks[label]
        dists = []
        for lsf in live_lsfs:
            dist = np.min(lsf_distance(codebook, lsf))
            dists.append(dist)
        distances.append(np.mean(dists))

    predicted_index = np.argmin(distances)
    predicted_label = labels[predicted_index]
    confidence = 1 / (1 + distances[predicted_index]) # Una forma simple de "confianza"

    print(f"Palabra reconocida: {predicted_label} (confianza: {confidence:.2f})")

# ------------------ Bucle principal para la prueba en tiempo real ------------------

if __name__ == "__main__":
    print("¡Listo para el reconocimiento en tiempo real!")
    while True:
        input("Presiona Enter para grabar y reconocer una palabra...")
        recognize_word()