'''import sounddevice as sd

print(sd.query_devices())
print("\nMicrófono por defecto:")
print(sd.query_devices(sd.default.device[0]))'''

import librosa.core
print(dir(librosa.core))