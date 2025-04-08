import asyncio
import edge_tts
import pyaudio
import wave
import io
import torchaudio
import time
import webrtcvad
import collections
import os
from pydub import AudioSegment
from speechbrain.pretrained import EncoderDecoderASR
from datetime import datetime

# Load ASR Model (do this once)
asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-transformer-transformerlm-librispeech", savedir="pretrained_models/asr")

# Text-to-Speech with Edge-TTS
async def generate_speech(text, output_path="response.mp3", voice="en-US-AriaNeural"):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

# Normalize and play audio through specified device
def normalize_audio(sound, target_dBFS=-25.0):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

def play_audio_through_device(file_path, output_device_index):
    sound = AudioSegment.from_file(file_path)
    sound = normalize_audio(sound)
    sound = sound.set_channels(2).set_frame_rate(44100).set_sample_width(2)

    buffer = io.BytesIO()
    sound.export(buffer, format="wav")
    buffer.seek(0)
    wf = wave.open(buffer, 'rb')

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

# Record with silence detection
def record_audio(filename="input.wav", timeout=5, aggressiveness=3):
    vad = webrtcvad.Vad(aggressiveness)
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=16000,
                        input=True,
                        frames_per_buffer=320)

    ring_buffer = collections.deque(maxlen=10)
    triggered = False
    voiced_frames = []

    print("🎙 Listening for speech...")

    start_time = time.time()
    while time.time() - start_time < timeout:
        frame = stream.read(320, exception_on_overflow=False)
        is_speech = vad.is_speech(frame, 16000)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.8 * ring_buffer.maxlen:
                triggered = True
                print("🎤 Recording started...")
                for f, _ in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.8 * ring_buffer.maxlen:
                break

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(voiced_frames))
    wf.close()

# Transcribe audio using SpeechBrain
def transcribe(filename="input.wav"):
    print("🧠 Transcribing...")
    return asr_model.transcribe_file(filename)

# Respond to user input
def generate_response(text):
    text = text.lower()
    if "time" in text:
        now = datetime.now()
        return f"The time is {now.strftime('%I:%M %p')}."
    elif "name" in text:
        return "My name is Pi Assistant."
    elif "weather" in text:
        return "Sorry, I can't check the weather right now."
    elif "stop" in text:
        return "Shutting down. Goodbye."
    else:
        return "I'm not sure how to respond to that."

# Wake word detection (basic)
def detect_wake_word():
    record_audio("wake.wav", timeout=3)
    text = transcribe("wake.wav")
    print("🗣 Wake attempt:", text)
    return "hey computer" in text.lower()

# Output device selection
def list_output_devices():
    p = pyaudio.PyAudio()
    print("\nAvailable output devices:\n")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["maxOutputChannels"] > 0:
            print(f"[{i}] {info['name']}")
    p.terminate()

def main_loop(output_device_index):
    print("👂 Waiting for wake word... Say 'Hey computer' to start.")
    while True:
        try:
            if detect_wake_word():
                print("✅ Wake word detected!")
                record_audio("command.wav")
                command_text = transcribe("command.wav")
                print("You said:", command_text)

                response_text = generate_response(command_text)
                print("💬 Responding:", response_text)

                asyncio.run(generate_speech(response_text))
                play_audio_through_device("response.mp3", output_device_index)

                if "shutting down" in response_text.lower():
                    break

        except KeyboardInterrupt:
            print("❌ Exiting...")
            break
        except Exception as e:
            print("⚠️ Error:", e)

# Run
if __name__ == "__main__":
    list_output_devices()
    device_index = int(input("Choose output device index (e.g., USB speaker): "))
    main_loop(device_index)

