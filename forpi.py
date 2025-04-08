import asyncio
import edge_tts
import pyaudio
import wave
import io
from pydub import AudioSegment

# --- TEXT TO SPEECH WITH EDGE-TTS ---
async def generate_speech(text, output_path="output.mp3", voice="en-US-AriaNeural"):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

# --- NORMALIZATION ---
def normalize_audio(sound, target_dBFS=-25.0):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

# --- LIST DEVICES ---
def list_output_devices(p):
    print("\nAvailable output devices:\n")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["maxOutputChannels"] > 0:
            print(f"[{i}] {info['name']}")
    print()

# --- PLAYBACK FUNCTION ---
def play_audio_through_device(file_path, output_device_index):
    # Load + normalize audio
    sound = AudioSegment.from_file(file_path)
    sound = normalize_audio(sound, target_dBFS=-25.0)
    sound = sound.set_channels(2).set_frame_rate(44100).set_sample_width(2)

    buffer = io.BytesIO()
    sound.export(buffer, format="wav")
    buffer.seek(0)
    wf = wave.open(buffer, 'rb')

    # Setup PyAudio playback
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True,
                    output_device_index=output_device_index)

    data = wf.readframes(1024)
    while data:
        stream.write(data)
        data = wf.readframes(1024)

    stream.stop_stream()
    stream.close()
    p.terminate()

# --- MAIN ---
if __name__ == "__main__":
    text = "Hello! This is your USB speaker test using Edge-TTS."

    # Step 1: Generate speech
    asyncio.run(generate_speech(text, output_path="output.mp3"))

    # Step 2: List output devices & choose one
    p = pyaudio.PyAudio()
    list_output_devices(p)
    index = int(input("Enter output device index for playback: "))
    p.terminate()

    # Step 3: Play audio through selected device
    play_audio_through_device("output.mp3", index)

