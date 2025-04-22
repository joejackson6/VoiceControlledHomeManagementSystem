import os
import torch
import sounddevice as sd
import soundfile as sf
import torchaudio
import pyttsx3
import speech_recognition as sr
from speechbrain.inference import SpeakerRecognition

RECORD_SECONDS = 7
SAMPLE_RATE = 16000
THRESHOLD = 0.60
EMBEDDINGS_PATH = "embeddings.pt"
AUDIO_PATH = "test_auth.wav"

def speak(text):
	engine = pyttsx3.init()
	engine.say(text)
	engine.runAndWait()

def load_embeddings(path):
	if not os.path.exists(path):
		print("embeddings.pt not found.")
		speak("User data not found. Please enroll first.")
		return None
	return torch.load(path)

def record_audio_file(filename):
	rec = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
	sd.wait()
	sf.write(filename, rec, SAMPLE_RATE)

def transcribe_audio(filename):
	r = sr.Recognizer()
	with sr.AudioFile(filename) as source:
		audio = r.record(source)
	try:
		return r.recognize_google(audio)
	except sr.UnknownValueError:
		print("Could not understand audio.")
		speak("Sorry, I didn't catch that.")
	except sr.RequestError as e:
		print(f"Speech Recognition error: {e}")
		speak("Speech recognition is not available.")
	return None

def live_authenticate(filename, verification, embeddings):
	waveform, _ = torchaudio.load(filename)
	test_embedding = verification.encode_batch(waveform).squeeze(0).detach()

	scores = {}
	for name, emb in embeddings.items():
		score = verification.similarity(test_embedding, emb).item()
		scores[name] = score

	most_likely = max(scores, key=scores.get)
	highest_score = scores[most_likely]

	print("\nAuthentication Scores:")
	for name, score in scores.items():
		print(f"{name}: {score:.4f}")

	if highest_score >= THRESHOLD:
		return most_likely
	else:
		print(f"Speaker not recognized. Best match: {most_likely} (score: {highest_score:.4f})")
		return None

def respond(text):
	if "weather" in text:
		reply = "I can't check the weather yet, but I'm working on it."
	elif "your name" in text:
		reply = "I'm your assistant."
	else:
		reply = f"You said: {text}"
	print("Assistant:", reply)
	speak(reply)

def listen_and_authenticate():
	verification = SpeakerRecognition.from_hparams(
		source="speechbrain/spkrec-ecapa-voxceleb",
		savedir="pretrained_models/spkrec-ecapa-voxceleb"
	)
	embeddings = load_embeddings(EMBEDDINGS_PATH)
	if embeddings is None:
		return

	while True:
		try:
			print("Waiting for command starting with 'Hey Homer' (press Ctrl+C to exit)...")
			print("Recording full input...")
			record_audio_file(AUDIO_PATH)

			text = transcribe_audio(AUDIO_PATH)
			if text and text.lower().startswith("hey homer"):
				command = text[len("hey homer"):].strip()
				user = live_authenticate(AUDIO_PATH, verification, embeddings)
				if user:
					print(f"Welcome, {user}!")
					speak(f"Hello, {user}")
					respond(command)
				else:
					speak("Sorry, I don't recognize you.")
		except KeyboardInterrupt:
			print("Exiting...")
			break

if __name__ == "__main__":
	listen_and_authenticate()

