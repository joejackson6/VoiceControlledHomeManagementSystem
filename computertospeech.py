import pyaudio
from pydub import AudioSegment
import wave
import io
import pyttsx3

def list_output_devices(p):
    print("\nAvailable output devices:\n")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["maxOutputChannels"] > 0:
            print(f"[{i}] {info['name']}")
    print()

# Make the output quieter
def normalize_audio(sound, target_dBFS=-20.0):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


def play_audio_through_device(file_path, output_device_index):
    # Load and convert to WAV format in memory
    sound = AudioSegment.from_file(file_path)
    sound = sound.set_channels(2).set_frame_rate(44100).set_sample_width(2)  # stereo, 44.1kHz, 16-bit
    sound = normalize_audio(sound, target_dBFS=-25.0)  # Softer playback


    buffer = io.BytesIO()
    sound.export(buffer, format="wav")
    buffer.seek(0)

    # Use wave module to open buffer
    wf = wave.open(buffer, 'rb')

    # Setup PyAudio
    p = pyaudio.PyAudio()

    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True,
                    output_device_index=output_device_index)

    # Play audio
    data = wf.readframes(1024)
    while data:
        stream.write(data)
        data = wf.readframes(1024)

    stream.stop_stream()
    stream.close()
    p.terminate()

# FOR SENDING TEXT TO ROBOT VOICE
if __name__ == "__main__":

    engine = pyttsx3.init()
    engine.save_to_file("Hello, this is a test.", "output.mp3")
    engine.runAndWait()

    audio_file = "output.mp3"
	
    p = pyaudio.PyAudio()
    list_input_devices(p)
    list_output_devices(p)

    try:
        index = int(input("Enter output device index to use for playback: "))
        play_audio_through_device(audio_file, index)
    except Exception as e:
        print("Error:", e)
    finally:
        p.terminate()

'''
# FOR SENDING AUDIO TO THE SPEAKER
if __name__ == "__main__":
	audio_file = "test3.wav"  # Replace with your file path

    p = pyaudio.PyAudio()
    list_input_devices(p)
    list_output_devices(p)

    try:
        index = int(input("Enter output device index to use for playback: "))
        play_audio_through_device(audio_file, index)
    except Exception as e:
        print("Error:", e)
    finally:
        p.terminate()
'''
