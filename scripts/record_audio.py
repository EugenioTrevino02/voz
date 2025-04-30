# scripts/record_audio.py
import sounddevice as sd
import soundfile as sf
import os

fs = 16000  # 16 kHz
duration = 2  # 2 segundos por grabación
base_dir = '../recordings/'

# Asegúrate que la carpeta base exista

os.makedirs(base_dir, exist_ok=True)


for number in range(1, 11):
    # Crear subcarpeta para cada número
    number_dir = os.path.join(base_dir, str(number))
    os.makedirs(number_dir, exist_ok=True)

    for repetition in range(15):
        input(f"\nPresiona ENTER y luego graba el número {number} (repetición {repetition + 1})...")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()

        filename = os.path.join(number_dir, f"num{number}_rep{repetition}.wav")
        sf.write(filename, recording, fs)
        print(f"✅ Grabado: {filename}")

