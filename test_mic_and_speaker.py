import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import subprocess
import time

# Audio settings
duration = 3
mic_samplerate = 44100
channels = 1
mic_name = "K66"
output_file = "mic_test.wav"
volume_percent = 70

# Locate mic index
mic_index = None
for i, dev in enumerate(sd.query_devices()):
    if mic_name in dev['name']:
        mic_index = i
        break

if mic_index is None:
    print(f"Microphone with name containing '{mic_name}' not found.")
    exit(1)

print("Recording... Speak now.")
recording = sd.rec(int(duration * mic_samplerate), samplerate=mic_samplerate, channels=channels, dtype='int16', device=mic_index)
sd.wait()
print("Recording complete.")

wav.write(output_file, mic_samplerate, recording)
print(f"Saved recording to {output_file}")

# Set speaker volume to safe level (ALSA)
try:
    subprocess.run(["amixer", "sset", "Master", f"{volume_percent}%"], check=True)
except subprocess.CalledProcessError:
    print("Could not set volume. You may need to install or configure ALSA properly.")

# Playback
print("Playing back...")
subprocess.run(["aplay", output_file])

