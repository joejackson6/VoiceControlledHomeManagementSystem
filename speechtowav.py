import pyaudio
import wave
import threading

def list_input_devices(p):
    print("\nAvailable input devices:\n")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            print(f"[{i}] {info['name']}")
    print()

def record_audio(device_index, output_filename="output.wav", rate=44100, channels=1, chunk=1024):
    p = pyaudio.PyAudio()
    
    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=rate,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=chunk)
    
    print("\n Recording... Press Enter to stop.")
    frames = []
    recording = True

    def wait_for_enter():
        nonlocal recording
        input()
        recording = False

    stopper = threading.Thread(target=wait_for_enter)
    stopper.start()

    while recording:
        data = stream.read(chunk)
        frames.append(data)

    print("Recording stopped.\n")

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save to WAV
    wf = wave.open(output_filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f"Audio saved to '{output_filename}'\n")


if __name__ == "__main__":
    pa = pyaudio.PyAudio()
    list_input_devices(pa)

    device_id = int(input("Enter the device ID to use: ").strip())
    filename = input("Enter output filename (e.g., 'my_audio.wav'): ").strip()
    if not filename.endswith(".wav"):
        filename += ".wav"

    record_audio(device_id, filename)

