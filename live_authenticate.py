import torch
import sounddevice as sd
import soundfile as sf
import torchaudio
from speechbrain.inference import SpeakerRecognition

RECORD_SECONDS = 7
THRESHOLD = 0.60
SAMPLE_RATE = 16000
FILENAME = "test.wav"

print("Recording... Speak now!")
recording = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
sd.wait()
sf.write(FILENAME, recording, SAMPLE_RATE)
print("Saved to test.wav")

verification = SpeakerRecognition.from_hparams(
	source="speechbrain/spkrec-ecapa-voxceleb",
	savedir="pretrained_models/spkrec-ecapa-voxceleb"
)

user_embeddings = torch.load("embeddings.pt")
waveform, sample_rate = torchaudio.load(FILENAME)
test_embedding = verification.encode_batch(waveform).squeeze(0).detach()

scores = {}
for name, emb in user_embeddings.items():
	score = verification.similarity(test_embedding, emb).item()
	scores[name] = score

most_likely = max(scores, key=scores.get)
highest_score = scores[most_likely]

print("\nAuthentication Scores:")
for name, score in scores.items():
	print(f"{name}: {score:.4f}")

if highest_score < 0.5:
	print(f"\nSpeaker not recognized. All scores too low (best: {most_likely}, score: {highest_score:.4f})")
else:
	print(f"\nRecognized as: {most_likely} (score: {highest_score:.4f})")

